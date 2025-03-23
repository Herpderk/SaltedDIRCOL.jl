"""
    derive_saltation_matrix(flow_I, flow_J, guard, reset)

Derives the function for computing the saltation matrix for a given hybrid transition.
"""
function derive_saltation_matrix(
    flow_I::Function,
    flow_J::Function,
    guard::Function,
    reset::Function
)::Function
    g_grad = x -> FD.gradient(guard, x)
    R_jac = x -> FD.jacobian(reset, x)
    return (x,u) -> (
        R_jac(x)
        + (flow_J(reset(x),u) - R_jac(x) * flow_I(x,u)) * g_grad(x)'
        / (g_grad(x)' * flow_I(x,u))
    )
end

"""
    Transition(flow_I, flow_J, guard, reset)

Contains all hybrid system objects pertaining to a hybrid transition.
"""
struct Transition
    flow_I::Function
    flow_J::Function
    guard::Function
    reset::Function
    saltation::Function
    function Transition(
        flow_I::Function,
        flow_J::Function,
        guard::Function,
        reset::Function
    )
        salt_expr = derive_saltation_matrix(flow_I, flow_J, guard, reset)
        return new(flow_I, flow_J, guard, reset, salt_expr)
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
    transitions::Dict{String, Transition}
end
