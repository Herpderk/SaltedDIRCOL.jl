"""
Derives the function for computing the saltation matrix for a given transition.
Input:
    prev_flow - Vector{Float64} Function of Vector{Float64} state x
    next_flow - Vector{Float64} Function of Vector{Float64} state x
    guard - Float64 Function of Vector{Float64} state x
    reset - Vector{Float64} Function of Vector{Float64} state x
Output:
    saltation_expr - Matrix{Float64} Function of Vector{Float64} state x
                     and Vector{Float64} input u
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
    guard - Float64 Function of Vector{Float64} state x
    reset - Vector{Float64} Function of Vector{Float64} state x
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
        return new(prev_mode, next_mode, guard, reset, salt_expr)
    end
end

"""
Contains the hybrid system objects pertaining to a hybrid mode.
Input:
    flow - Vector{Float64} Function of Vector{Float64} state x and input u
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

"""
Constructs a Dict that maps keys to modes given pairs of keys and modes.
Input:
    key_mode_pairs - Vector{Tuple{String, HybridMode}}
Output:
    key_mode_dict - Dict{String, HybridMode}
"""
function generate_key_mode_dict(
    key_mode_pairs::Vector{Tuple{String, HybridMode}}
)::Dict{String, HybridMode}
    key_mode_dict = Dict()
    for (key, mode) = key_mode_pairs
        key_mode_dict[key] = mode
    end
    return key_mode_dict
end

"""
Contains all hybrid system objects as well as the state and input dimensions.
Input:
    key_mode_pairs - Vector{Tuple{String, HybridMode}}
    nx - Int64
    nu - Int64
Output:
    model - HybridSystem
"""
struct HybridSystem
    modes::Dict{String, HybridMode}
    nx::Int64
    nu::Int64
    function HybridSystem(
        key_mode_pairs::Vector{Tuple{String, HybridMode}},
        nx::Int64,
        nu::Int64
    )
        modes = generate_key_mode_dict(key_mode_pairs)
        return new(modes, nx, nu)
    end
end
