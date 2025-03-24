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
    g::Function
    h::Function
    c::Function
    fgrad::Function
    gjac::Function
    hjac::Function
    cjac::Function
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

        # TODO: Add Lagrangian function

        # Autodiff all callbacks
        fgrad = y -> ForwardDiff.gradient(f, y)
        gjac = y -> ForwardDiff.jacobian(g, y)
        hjac = y -> ForwardDiff.jacobian(h, y)
        cjac = y -> ForwardDiff.jacobian(c, y)
        return new(f, g, h, c, fgrad, gjac, hjac, cjac)
    end
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
        callbacks::SolverCallbacks
    )::IpoptCallbacks
        # Get constraint jacobian sparsity pattern
        sp_cjac = sparse(cjac(Inf * ones(params.dims.ny)))
        crows, ccols, cvals = findnz(sp_cjac)

        # Objective evaluation
        function eval_f(
            x::Vector{Float64}
        )::Float64
            return callbacks.f(x)
        end

        # Constraint evaluation
        function eval_g(
            x::Vector{Float64},
            g::Vector{Float64}
        )::Nothing
            g = callbacks.c(x)
        end

        # Objective gradient
        function eval_grad_f(
            x::Vector{Float64},
            grad_f::Vector{Float64}
        )::Nothing
            grad_f = callbacks.fgrad(x)
        end

        # Constraint jacobian
        function eval_jac_g(
            x::Vector{Float64},
            rows::Vector{Cint},
            cols::Vector{Cint},
            values::Union{Nothing, Vector{Float64}}
        )::Nothing
            if isnothing(values)
                rows = crows
                cols = ccols
            else
                values = sparse(callbacks.cjac(x)).nzval
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
        end
        return new(eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    end
end
