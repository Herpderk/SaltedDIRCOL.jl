"""
    ProblemParameters(integrator, system, Q, R, Qf, N, Δt)

Contains the parameters for a trajectory optimization problem with an assumed non-time-varying quadratic objective.
"""
struct ProblemParameters
    integrator::Function
    objective::Function
    dims::PrimalDimensions
    idx::PrimalIndices
    Δt::Union{Nothing, Real}
    function ProblemParameters(
        integrator::Function,
        system::HybridSystem,
        Q::Union{Matrix, Diagonal},
        R::Union{Matrix, Diagonal},
        Qf::Union{Matrix, Diagonal},
        N::Int,
        Δt::Real = nothing
    )::ProblemParameters
        nx = system.nx
        nu = system.nu
        nt = isnothing(Δt) ? 1 : 0
        dims = PrimalDimensions(N, nx, nu, nt)
        idx = PrimalIndices(dims)
        obj = init_quadratic_cost(dims, idx, Q, R, Qf)
        return new(integrator, obj, dims, idx, Δt)
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
    function SolverCallbacks(
        params::ProblemParameters,
        sequence::Vector{TransitionTiming},
        term_guard::Function,
        xrefs::Vector,
        urefs::Vector,
        xic::Vector,
        xgc::Union{Nothing, Vector} = nothing
    )::SolverCallbacks
        # Enforce transition sequence timing rules
        assert_timings(params, sequence)

        # Define objective
        f = y -> params.objective(xrefs, urefs, y)

        # Compose inequality constraints
        keepout = y -> guard_keepout(params, sequence, term_guard, y)
        g = y -> keepout(y)

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

        # Autodiff all callbacks
        f_grad = y -> ForwardDiff.gradient(f, y)
        jacs = [y -> ForwardDiff.jacobian(func, y) for func = (g, h, c)]
        f_hess = y -> ForwardDiff.hessian(f, y)
        Lc_hess = (y, λ) -> ForwardDiff.hessian(dy -> Lc(dy, λ), y)

        # Get constraint / dual variable dimensions
        yinf = fill(Inf, params.dims.ny)
        ng = length(g(yinf))
        nh = length(h(yinf))
        dims = DualDimensions(ng, nh)

        # Get constraint jacobian and Lagrangian hessian sparsity patterns
        λinf = fill(Inf, dims.nc)
        c_jac = jacs[end]
        c_jac_sp = SparsityPattern(c_jac(yinf))
        L_hess_sp = SparsityPattern(f_hess(yinf) + Lc_hess(yinf, λinf))
        return new(
            f, Lc, g, h, c,
            f_grad, jacs...,
            f_hess, Lc_hess,
            c_jac_sp, L_hess_sp,
            dims
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
            x::Vector
        )::Float64
            return Float64(cb.f(x))
        end

        function eval_g(        # Constraint evaluation
            x::Vector,
            g::Vector
        )::Nothing
            g .= cb.c(x)
            return
        end

        function eval_grad_f(   # Objective gradient
            x::Vector,
            grad_f::Vector
        )::Nothing
            grad_f .= cb.f_grad(x)
            return
        end

        function eval_jac_g(    # Constraint jacobian
            x::Vector,
            rows::Vector{Cint},
            cols::Vector{Cint},
            values::Union{Nothing, Vector}
        )::Nothing
            if isnothing(values)
                rows .= cb.c_jac_sp.row_idx
                cols .= cb.c_jac_sp.col_idx
            else
                #values .= sparse(cb.c_jac(x)).nzval
                c_jac = cb.c_jac(x)
                @inbounds @simd for i = 1:cb.c_jac_sp.nzvals
                    values[i] = c_jac[
                        cb.c_jac_sp.row_idx[i],
                        cb.c_jac_sp.col_idx[i]
                    ]
                end
            end
            return
        end

        function eval_h(        # Lagrangian hessian
            x::Vector,
            rows::Vector{Cint},
            cols::Vector{Cint},
            obj_factor::Real,
            lambda::Vector,
            values::Union{Nothing, Vector}
        )::Nothing
            if isnothing(values)
                rows .= cb.L_hess_sp.row_idx
                cols .= cb.L_hess_sp.col_idx
            else
                L_hess = obj_factor * cb.f_hess(x) + cb.Lc_hess(x, lambda)
                @inbounds @simd for i = 1:cb.L_hess_sp.nzvals
                    values[i] = L_hess[
                        cb.L_hess_sp.row_idx[i],
                        cb.L_hess_sp.col_idx[i]
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
    y0::Vector,
    print_level::Int = 5;
    approx_hessian::Bool = true
)::IpoptProblem
    # Define primal and constraint bounds
    cb_ipopt = IpoptCallbacks(cb)
    ylb = fill(-Inf, params.dims.ny)
    yub = fill(Inf, params.dims.ny)
    clb = [fill(-Inf, cb.dims.ng); zeros(cb.dims.nh)]
    cub = zeros(cb.dims.nc)

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
    if approx_hessian
        Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")
    end
    Ipopt.AddIpoptIntOption(prob, "print_level", print_level)

    # Warm-start and solve
    prob.x = y0
    status = Ipopt.IpoptSolve(prob)
    return prob
end
