"""
    ProblemParameters(integrator, system, Q, R, Qf, N, Δt)

Contains the parameters for a trajectory optimization problem with an assumed non-time-varying quadratic objective.
"""
struct ProblemParameters
    integrator::ImplicitIntegrator
    objective::TrajectoryCost
    dims::PrimalDimensions
    idx::PrimalIndices
    Δt::Union{Nothing, AbstractFloat}
    Δtlb::Union{Nothing, AbstractFloat}
    function ProblemParameters(
        integrator::Integrator,
        system::HybridSystem,
        stage_cost::Function,
        terminal_cost::Function,
        N::Int;
        Δt::Union{Nothing, AbstractFloat} = nothing,
        Δtlb::Union{Nothing, AbstractFloat} = nothing
    )::ProblemParameters
        if isnothing(Δt)
            nt = 1
            Δtlb = isnothing(Δt) & isnothing(Δtlb) ? 1e-3 : Δtlb
        else
            nt = 0
        end
        dims = PrimalDimensions(N, system.nx, system.nu, nt)
        idx = PrimalIndices(dims)
        obj = TrajectoryCost(dims, idx, stage_cost, terminal_cost)
        if typeof(integrator) == ExplicitIntegrator
            integrator = ImplicitIntegrator(integrator)
        end
        return new(integrator, obj, dims, idx, Δt, Δtlb)
    end
end

"""
    assert_timings(params, sequence)

Raises an error if a transition time step is not at least 2 more than the first time step or previous transition time step. Also raises an error if the final transition time step is >= the horizon length N.
"""
function assert_timings(
    params::ProblemParameters,
    sequence::Vector{TransitionTiming}
)::Nothing
    k = 1
    for timing = sequence
        timing.k < k+2 ? error(
            "Please have at least 2 time steps between transitions!") : nothing
        k = timing.k
    end
    sequence[end].k >= params.dims.N ? error(
        "Final transition time step should be < horizon length N!") : nothing
    return
end

"""
    SolverCallbacks(params, sequence, term_guard, xrefs, urefs, xic, xgc)

Contains solver-agnostic callback functions, constraint jacobian and Lagrangian hessian sparsity patterns, and dual variable dimensions.
"""
struct SolverCallbacks
    f::Function
    Lc::Function
    g::Function
    h::Function
    c::Function
    f_grad::Function
    g_jac::Function
    h_jac::Function
    c_jac::Function
    f_hess::Function
    Lc_hess::Function
    c_jac_sp::SparsityPattern
    L_hess_sp::SparsityPattern
    dims::DualDimensions
    gauss_newton::Bool
    function SolverCallbacks(
        params::ProblemParameters,
        sequence::Vector{TransitionTiming},
        term_guard::Function,
        xrefs::Vector{<:AbstractFloat},
        urefs::Vector{<:AbstractFloat},
        xic::Vector{<:AbstractFloat},
        xgc::Union{Nothing, Vector{<:AbstractFloat}} = nothing;
        gauss_newton::Bool = false
    )::SolverCallbacks
        # Enforce transition sequence timing rules
        assert_timings(params, sequence)

        # Define objective
        yref = compose_trajectory(params.dims, params.idx, xrefs, urefs)
        f = y -> params.objective(yref, y)

        # Compose inequality constraints
        keepout = y -> guard_keepout(params, sequence, term_guard, y)
        if isnothing(params.idx.Δt)
            g = y -> keepout(y)
        else
            timestep = y -> timestep_size(params, y)
            g = y -> [keepout(y); timestep(y)]
        end

        # Compose equality constraints
        ic = y -> initial_condition(params, xic, y)
        defect = y -> dynamics_defect(params, sequence, y, params.Δt)
        touchdown = y -> guard_touchdown(params, sequence, y)
        if isnothing(xgc)
            h = y -> [ic(y); defect(y); touchdown(y)]
        else
            gc = y -> goal_condition(params, xgc, y)
            h = y -> [ic(y); defect(y); touchdown(y); gc(y)]
        end

        # Compose all constraints
        c = y -> [g(y); h(y)]

        # Define constraint component of Lagrangian
        Lc = (y, λ) -> λ' * c(y)

        # Get constraint / dual variable dimensions
        yinf = fill(Inf, params.dims.ny)
        ng = length(g(yinf))
        nh = length(h(yinf))
        dims = DualDimensions(ng, nh)

        # Autodiff all callbacks
        println("forward-diffing...")
        f_grad = y -> ForwardDiff.gradient(f, y)
        g_jac = y -> ForwardDiff.jacobian(g, y)
        h_jac = y -> ForwardDiff.jacobian(h, y)
        c_jac = y -> ForwardDiff.jacobian(c, y)
        f_hess = y -> ForwardDiff.hessian(f, y)
        if gauss_newton
            Lc_hess = (y, λ) -> zeros(params.dims.ny, params.dims.ny)
        else
            Lc_hess = (y, λ) -> ForwardDiff.hessian(dy -> Lc(dy, λ), y)
        end

        # Get constraint jacobian and Lagrangian hessian sparsity patterns
        println("getting sparsity patterns...")
        λinf = fill(Inf, dims.nc)
        c_jac_sp = SparsityPattern(c_jac(yinf))
        L_hess_sp = SparsityPattern(f_hess(yinf) + Lc_hess(yinf, λinf))
        return new(
            f, Lc, g, h, c,
            f_grad, g_jac, h_jac, c_jac,
            f_hess, Lc_hess,
            c_jac_sp, L_hess_sp,
            dims, gauss_newton
        )
    end
end

"""
    IpoptCallbacks(cb)

Contains callbacks for the Julia-wrapped Ipop C interface according to the
solver's documentation: https://github.com/jump-dev/Ipopt.jl/tree/master.
"""
struct IpoptCallbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h::Function
    function IpoptCallbacks(
        cb::SolverCallbacks
    )::IpoptCallbacks
        function eval_f(        # Objective evaluation
            x::Vector{Float64}
        )::Float64
            return cb.f(x)
        end

        function eval_g(        # Constraint evaluation
            x::Vector{Float64},
            g::Vector{Float64}
        )::Nothing
            g .= cb.c(x)
            return
        end

        function eval_grad_f(   # Objective gradient
            x::Vector{Float64},
            grad_f::Vector{Float64}
        )::Nothing
            grad_f .= cb.f_grad(x)
            return
        end

        function eval_jac_g(    # Constraint jacobian
            x::Vector{Float64},
            rows::Vector{Cint},
            cols::Vector{Cint},
            values::Union{Nothing, Vector{Float64}}
        )::Nothing
            if isnothing(values)
                rows .= cb.c_jac_sp.row_coords
                cols .= cb.c_jac_sp.col_coords
            else
                #values .= sparse(cb.c_jac(x)).nzval
                c_jac = cb.c_jac(x)
                @inbounds @simd for i = 1:cb.c_jac_sp.nzvals
                    values[i] = c_jac[
                        cb.c_jac_sp.row_coords[i],
                        cb.c_jac_sp.col_coords[i]
                    ]
                end
            end
            return
        end

        function eval_h(        # Lagrangian hessian
            x::Vector{Float64},
            rows::Vector{Cint},
            cols::Vector{Cint},
            obj_factor::Float64,
            lambda::Vector{Float64},
            values::Union{Nothing, Vector{Float64}}
        )::Nothing
            if isnothing(values)
                rows .= cb.L_hess_sp.row_coords
                cols .= cb.L_hess_sp.col_coords
            else
                L_hess = obj_factor * cb.f_hess(x) + cb.Lc_hess(x, lambda)
                @inbounds @simd for i = 1:cb.L_hess_sp.nzvals
                    values[i] = L_hess[
                        cb.L_hess_sp.row_coords[i],
                        cb.L_hess_sp.col_coords[i]
                    ]
                end
            end
            return
        end
        return new(eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    end
end

"""
    ipopt_solve(params, cb, y0, print_level; approx_hessian)

Sets up and solves a trajectory optimization using Ipopt given a set of problem parameters and solver callback functions.
"""
function ipopt_solve(
    params::ProblemParameters,
    cb::SolverCallbacks,
    y0::Vector{<:AbstractFloat};
    gauss_newton::Bool = false,
    print_level::Int = 5,
    max_iter::Int = 1000,
    tol::AbstractFloat = 1e-8
)::IpoptProblem
    # Define primal and constraint bounds
    cb_ipopt = IpoptCallbacks(cb)
    ylb = fill(-Inf, params.dims.ny)
    yub = fill(Inf, params.dims.ny)
    clb = [fill(-Inf, cb.dims.ng); zeros(cb.dims.nh)]
    cub = zeros(cb.dims.nc)

    @show params.dims.ny

    # Create Ipopt problem
    prob = Ipopt.CreateIpoptProblem(
        params.dims.ny,
        ylb,
        yub,
        cb.dims.nc,
        clb,
        cub,
        cb.c_jac_sp.nzvals,
        cb.L_hess_sp.nzvals,
        cb_ipopt.eval_f,
        cb_ipopt.eval_g,
        cb_ipopt.eval_grad_f,
        cb_ipopt.eval_jac_g,
        cb_ipopt.eval_h
    )

    # Add solver options
    if gauss_newton | cb.gauss_newton
        Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")
    end
    Ipopt.AddIpoptIntOption(prob, "print_level", print_level)
    Ipopt.AddIpoptIntOption(prob, "max_iter", max_iter)
    Ipopt.AddIpoptNumOption(prob, "tol", tol)

    # Warm-start and solve
    println("starting ipopt...")
    prob.x = y0
    status = Ipopt.IpoptSolve(prob)
    return prob
end
