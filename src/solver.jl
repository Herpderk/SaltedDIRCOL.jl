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
    fgrad::Function
    gjac::Function
    hjac::Function
    function SolverCallbacks(
        params::ProblemParameters,
        sequence::Vector{TransitionTiming},
        term_guard::Function,
        xrefs::RealValue,
        urefs::RealValue,
        xic::RealValue,
        xgc::Union{Nothing, Real} = nothing
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

        # Autodiff all callbacks
        fgrad = y -> ForwardDiff.gradient(f, y)
        gjac = y -> ForwardDiff.jacobian(g, y)
        hjac = y -> ForwardDiff.jacobian(h, y)
        return new(f, g, h,fgrad, gjac, hjac)
    end
end
