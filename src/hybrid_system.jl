"""
Derives the function for computing the saltation matrix for a given transition.
"""
function derive_saltation_matrix(
    prev_flow::Function,
    next_flow::Function,
    guard::Function,
    reset::Function
)::Function
    g_grad = x -> FD.gradient(guard, x)
    R_jac = x -> FD.jacobian(reset, x)
    return (x,u) -> (
        R_jac(x)
        + (next_flow(reset(x),u) - R_jac(x) * prev_flow(x,u)) * g_grad(x)'
        / (g_grad(x)' * prev_flow(x,u))
    )
end

"""
Contains the hybrid system objects pertaining to a hybrid transition.

Input:
    prev_mode - HybridMode
    next_mode - HybridMode
    guard - Float64 Function of the state x
    reset - Vector{FLoat64} Function of the state x

Output:
    transition - Transition struct containing prev_mode, next_mode, guard,
                 reset, and saltation matrix expression
"""
struct Transition
    prev_mode
    next_mode
    guard::Function
    reset::Function
    saltation::Function
    function Transition(
        prev_mode,
        next_mode,
        guard::Function,
        reset::Function
    )
        salt_expr = derive_saltation_matrix(
            prev_mode.flow,
            next_mode.flow,
            guard,
            reset
        )
        new(prev_mode, next_mode, guard, reset, salt_expr)
    end
end

"""
Contains the hybrid system objects pertaining to a hybrid mode.

Input:
    flow - Vector{Float64} Function of the state x and input u
    transitions - optional Vector{Transition}

Output:
    hybrid mode - HybridMode struct containing flow and transitions
"""
mutable struct HybridMode
    flow::Function
    transitions::Union{Vector{Transition}, Nothing}
    function HybridMode(
        flow::Function,
        transitions::Union{Vector{Transition}, Nothing} = nothing
    )
        return new(flow, transitions)
    end
end
