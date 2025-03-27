"""
    derive_saltation_matrix(flow_I, flow_J, guard, reset)

Derives the saltation matrix function for a given hybrid transition.
"""
function derive_saltation_matrix(
    flow_I::Function,
    flow_J::Function,
    guard::Function,
    reset::Function
)::Function
    function g_grad(x::DiffVector)
        return ForwardDiff.gradient(guard, x)
    end
    function R_jac(x::DiffVector)
        return ForwardDiff.jacobian(reset, x)
    end
    function salt_mat(x::DiffVector, u::DiffVector)
        return (
            R_jac(x)
            + (flow_J(reset(x),u) - R_jac(x) * flow_I(x,u)) * g_grad(x)'
            / (g_grad(x)' * flow_I(x,u))
        )
    end
    return salt_mat
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
    saltation::Function
    next_transition::Union{Nothing, Transition}
    function Transition(
        flow_I::Function,
        flow_J::Function,
        guard::Function,
        reset::Function,
        next_transition::Union{Nothing, Transition} = nothing
    )::Transition
        salt_expr = derive_saltation_matrix(flow_I, flow_J, guard, reset)
        return new(flow_I, flow_J, guard, reset, salt_expr, next_transition)
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
    HybridSystem(nx, nu, transitions)

Contains all hybrid system objects in addition to the system's state and input dimensions.
"""
struct HybridSystem
    nx::Int
    nu::Int
    transitions::Dict{Symbol, Transition}
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
    u_idx = [1:system.nu .+ (k-1)*system.nu  for k = 1:N-1]
    xs = [zeros(system.nx) for k = 1:N]
    xs[1] = x0
    curr_trans = system.transitions[init_transition]
    for k = 1:N-1
        xk = xs[k]
        if curr_trans.guard(xk) <= 0.0
            xk = curr_trans.reset(xk)
            curr_trans = curr_trans.next_transition
        end
        xs[k+1] = integrator(curr_trans.flow_I, xk, us[u_idx[k]], Δt)
    end
    return vcat(xs...)
end
