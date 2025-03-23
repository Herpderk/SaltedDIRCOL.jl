const DecisionVar = Union{Real, Vector{Real}}

"""
    rk4(dynamics, x, u, h)

Explicit integrator using the fourth order Runge-Kutta method.
"""
function rk4(
    dynamics::Function,
    x::DecisionVar,
    u::DecisionVar,
    h::Real
)::DecisionVar
    k1 = dynamics(x, u)
    k2 = dynamics(x + h/2 * k1, u)
    k3 = dynamics(x + h/2 * k2, u)
    k4 = dynamics(x + h * k3, u)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
end

"""
    hermite_simpson(dynamics, x0, u0, x1, h)

Implicit integrator using the Hermite-Simpson method, which is accurate to third order.
"""
function hermite_simpson(
    dynamics::Function,
    x0::DecisionVar,
    u0::DecisionVar,
    x1::DecisionVar,
    h::Real
)::DecisionVar
    f0 = dynamics(x0, u0)
    f1 = dynamics(x1, u0)
    xc = 1/2*(x0 + x1) + h/8*(f0 - f1)
    fc = dynamics(xc, u0)
    return x0 - x1 + h/6*(f0 + 4*fc + f1)
end
