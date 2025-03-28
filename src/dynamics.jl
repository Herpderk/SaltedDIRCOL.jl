"""
    SaltationMatrix(flow_I, flow_J, guard, reset)

Derives the saltation matrix function for a given hybrid transition.
"""
struct SaltationMatrix
    flow_I::Function
    flow_J::Function
    guard::Function
    reset::Function
end

function (transition::SaltationMatrix)(
    x::Vector{<:DiffFloat},
    u::Vector{<:DiffFloat}
)::Matrix{<:DiffFloat}
    xJ = transition.reset(x)
    g_grad = ForwardDiff.gradient(transition.guard, x)
    R_jac = ForwardDiff.jacobian(transition.reset, x)
    return (
        R_jac
        + (transition.flow_J(xJ, u) - R_jac * transition.flow_I(x, u))
        * g_grad' / (g_grad' * transition.flow_I(x,u))
    )
end

"""
    Transition(flow_I, flow_J, guard, reset)

Contains all hybrid system objects pertaining to a hybrid transition.
"""
mutable struct Transition
    flow_I::Function
    flow_J::Function
    guard::Function
    reset::Function
    saltation::SaltationMatrix
    next_transition::Union{Nothing, Transition}
    function Transition(
        flow_I::Function,
        flow_J::Function,
        guard::Function,
        reset::Function,
        next_transition::Union{Nothing, Transition} = nothing
    )::Transition
        saltation = SaltationMatrix(flow_I, flow_J, guard, reset)
        return new(flow_I, flow_J, guard, reset, saltation, next_transition)
    end
end

"""
    TransitionTiming(k, transition)

Contains the time step for which the given hybrid transition occurs at the beginning of.
"""
struct TransitionTiming
    k::Int
    transition::Transition
end

"""
    HybridSystem(
        nx, nu, transitions;
        stage_ineq_constr=nothing, stage_eq_constr=nothing,
        term_ineq_constr=nothing, term_eq_constr=nothing
    )

Contains all hybrid system objects in addition to the system's state and input dimensions. Assumes the following forms for stage and terminal constraints:

stage_constr(x,u) (<)= 0

term_constr(x) (<)= 0
"""
mutable struct HybridSystem
    nx::Int
    nu::Int
    stage_ng::Int
    stage_nh::Int
    term_ng::Int
    term_nh::Int
    stage_ineq_constr::Union{Nothing, Function}
    stage_eq_constr::Union{Nothing, Function}
    term_ineq_constr::Union{Nothing, Function}
    term_eq_constr::Union{Nothing, Function}
    transitions::Dict{Symbol, Transition}
    function HybridSystem(
        nx::Int,
        nu::Int,
        transitions::Dict{Symbol, Transition};
        stage_ineq_constr::Union{Nothing, Function} = nothing,
        stage_eq_constr::Union{Nothing, Function} = nothing,
        term_ineq_constr::Union{Nothing, Function} = nothing,
        term_eq_constr::Union{Nothing, Function} = nothing
    )::HybridSystem
        # Pack stage and terminal constraints together
        stage_constrs = (stage_ineq_constr, stage_eq_constr)
        term_constrs = (term_ineq_constr, term_eq_constr)

        # Get dimensions of constraint functions
        xtest = zeros(nx)
        utest = zeros(nu)
        stage_dims = [
            isnothing(constr) ? 0 : length(constr(xtest, utest))
            for constr in stage_constrs
        ]
        term_dims = [
            isnothing(constr) ? 0 : length(constr(xtest))
            for constr in term_constrs
        ]

        return new(
            nx, nu,
            stage_dims..., term_dims...,
            stage_constrs..., term_constrs...,
            transitions
        )
    end
end

"""
"""
function roll_out(
    integrator::ExplicitIntegrator,
    system::HybridSystem,
    N::Int,
    Δt::AbstractFloat,
    us::Vector{<:AbstractFloat},
    x0::Vector{<:AbstractFloat},
    init_transition::Symbol
)::Vector{<:AbstractFloat}
    # Init loop variables
    u_idx = [(1:system.nu) .+ (k-1)*system.nu for k = 1:N-1]
    xs = [zeros(system.nx) for k = 1:N]
    xs[1] = x0
    curr_trans = system.transitions[init_transition]

    # Roll out over time horizon
    for k = 1:N-1
        xk = xs[k]
        # Reset if guard is hit
        if curr_trans.guard(xk) <= 0.0
            xk = curr_trans.reset(xk)
            curr_trans = curr_trans.next_transition
        end
        # Integrate smooth dynamics
        xs[k+1] = integrator(curr_trans.flow_I, xk, us[u_idx[k]], Δt)
    end
    return vcat(xs...)
end
