"""
Explicit integrator using the fourth order Runge-Kutta method.
Input:
    dynamics - Vector Function of state x and input u
    x - Vector representing the system's state at a given time
    u - Vector representing the system's input at a given time
    h - Flo representing the integration time step
Output:
    next x - Vector representing the next state after the time step
"""
function rk4(
    dynamics::Function,
    x::Vector,
    u::Vector,
    h::Union{Float64, VariableRef}
)::Vector
    k1 = dynamics(x, u)
    k2 = dynamics(x + h/2 * k1, u)
    k3 = dynamics(x + h/2 * k2, u)
    k4 = dynamics(x + h * k3, u)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
end
