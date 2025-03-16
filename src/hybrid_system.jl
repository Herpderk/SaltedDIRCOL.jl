using JuMP
import ForwardDiff as FD

function derive_saltation_matrix(
    prev_flow::Function,
    next_flow::Function,
    guard::Function,
    reset::Function
)::Function
    g_grad = x -> FD.gradient(guard, x)
    R_jac = x -> FD.jacobian(reset, x)
    salt = (x,u) -> (
        R_jac(x) + (next_flow(x,u) - R_jac(x) * prev_flow(x,u)) * g_grad(x)'
        / (g_grad(x)' * prev_flow(x,u))
    )
    return salt
end

abstract type T end

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
        salt = derive_saltation_matrix(
            prev_mode.flow,
            next_mode.flow,
            guard,
            reset
        )
        new(prev_mode, next_mode, guard, reset, salt)
    end
end

mutable struct HybridMode
    flow::Function
    transitions::Union{Vector{Transition}, Nothing}
    function HybridMode(
        flow::Function,
        transitions::Union{Vector{Transition}, Nothing} = nothing
    )
        new(flow, transitions)
    end
end
