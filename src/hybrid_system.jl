"""
Derives the function for computing the saltation matrix for a given transition.
Input:
    prev_flow - Vector Function of Vector state x
    next_flow - Vector Function of Vector state x
    guard - scalar Function of Vector state x
    reset - Vector Function of Vector state x
Output:
    saltation_expr - Matrix Function of Vector state x and input u
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
    guard - scalar Function of Vector state x
    reset - Vector Function of Vector state x
Output:
    transition - Transition struct containing prev_mode, next_mode, guard,
                 reset, and saltation matrix expression
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
Contains all hybrid system objects as well as the state and input dimensions.
Input:
    key_mode_pairs - Vector{Tuple{String, HybridMode}}
    nx - Int
    nu - Int
Output:
    hybrid_system - HybridSystem
"""
struct HybridSystem
    nx::Int
    nu::Int
    transitions::Dict{String, Transition}
    function HybridSystem(
        nx::Int,
        nu::Int,
        transitions::Dict{String, Transition}
    )
        return new(nx, nu, transitions)
    end
end
