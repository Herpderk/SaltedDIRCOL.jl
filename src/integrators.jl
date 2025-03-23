const DecisionVar = Union{Real, Vector{Real}}

"""
    rk4(dynamics, x0, u0, x1, h)

Explicit integrator using the fourth order Runge-Kutta method. Written as an equality constraint.
"""
function rk4(
    dynamics::Function,
    x0::DecisionVar,
    u0::DecisionVar,
    x1::DecisionVar,
    h::Real
)::DecisionVar
    k1 = dynamics(x0, u0)
    k2 = dynamics(x0 + h/2*k1, u0)
    k3 = dynamics(x0 + h/2*k2, u0)
    k4 = dynamics(x0 + h*k3, u0)
    return x0 - x1 + h/6*(k1 + 2*k2 + 2*k3 + k4)
end

"""
    hermite_simpson(dynamics, x0, u0, x1, h)

Implicit integrator using the Hermite-Simpson method.
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
