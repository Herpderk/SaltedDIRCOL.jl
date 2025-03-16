function rk4(
    dynamics::Function,
    h::Float64
)::Function
    k1 = dynamics(x, u)
    k2 = dynamics(x + h/2 * k1, u)
    k3 = dynamics(x + h/2 * k2, u)
    k4 = dynamics(x + h * k3, u)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
end
