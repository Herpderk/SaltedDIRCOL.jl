"""
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
    function SolverCallbacks(
        params::ProblemParameters,
        sequence::Vector{TransitionTiming},
        term_guard::Function,
        xrefs::Value,
        urefs::Value,
        xic::Value,
        xgc::Union{Nothing, Value} = nothing
    )::SolverCallbacks
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
        return new(
            f, Lc, g, h, c,
            f_grad, jacs...,
            f_hess, Lc_hess
        )
    end
end

"""
"""
function get_sparsity_pattern(
    A::Matrix
)::Tuple{Vector{Int}, Vector{Int}}
    rows, cols, vals = findnz(sparse(A))
    return rows, cols
end

"""
"""
struct IpoptCallbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h::Function
    function IpoptCallbacks(
        params::ProblemParameters,
        cb::SolverCallbacks
    )::IpoptCallbacks
        # Get constraint jacobian and Lagrangian hessian sparsity patterns
        yinf = Inf * ones(params.dims.ny)
        λinf = Inf * ones(length(cb.c(yinf)))
        c_rows, c_cols = get_sparsity_pattern(cb.c_jac(yinf))
        L_rows, L_cols = get_sparsity_pattern(
            cb.f_hess(yinf) + cb.Lc_hess(yinf, λinf)
        )

        # Objective evaluation
        function eval_f(
            x::Vector{Float64}
        )::Float64
            return cb.f(x)
        end

        # Constraint evaluation
        function eval_g(
            x::Vector{Float64},
            g::Vector{Float64}
        )::Nothing
            g = cb.c(x)
        end

        # Objective gradient
        function eval_grad_f(
            x::Vector{Float64},
            grad_f::Vector{Float64}
        )::Nothing
            grad_f = cb.f_grad(x)
        end

        # Constraint jacobian
        function eval_jac_g(
            x::Vector{Float64},
            rows::Vector{Cint},
            cols::Vector{Cint},
            values::Union{Nothing, Vector{Float64}}
        )::Nothing
            if isnothing(values)
                rows = c_rows
                cols = c_cols
            else
                values = sparse(cb.c_jac(x)).nzval
            end
        end

        # Lagrangian hessian
        function eval_h(
            x::Vector{Float64},
            rows::Vector{Cint},
            cols::Vector{Cint},
            obj_factor::Float64,
            lambda::Float64,
            values::Union{Nothing, Vector{Float64}}
        )::Nothing
            if isnothing(values)
                rows = L_rows
                cols = L_cols
            else
                Lhess = obj_factor * cb.f_hess(x) + cb.Lc_hess(x, lambda)
                values = sparse(Lhess).nzval
            end
        end
        return new(eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    end
end
