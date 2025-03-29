module Explicit
    using ..SaltedDIRCOL: DiffFloat

    """
        rk4(dynamics, x0, u0, x1, Δt)

    Explicit integrator using the fourth order Runge-Kutta method. Written as an equality constraint.
    """
    function rk4(
        dynamics::Function,
        x0::Vector{<:DiffFloat},
        u0::Vector{<:DiffFloat},
        Δt::DiffFloat
    )::Vector{<:DiffFloat}
        k1 = dynamics(x0, u0)
        k2 = dynamics(x0 + Δt/2*k1, u0)
        k3 = dynamics(x0 + Δt/2*k2, u0)
        k4 = dynamics(x0 + Δt*k3, u0)
        return x0 + Δt/6*(k1 + 2*k2 + 2*k3 + k4)
    end

    """
    """
    function rk3(
        dynamics::Function,
        x0::Vector{<:DiffFloat},
        u0::Vector{<:DiffFloat},
        Δt::DiffFloat
    )::Vector{<:DiffFloat}
        k1 = dynamics(x0, u0)
        k2 = dynamics(x0 + Δt/2*k1, u0)
        k3 = dynamics(x0 - Δt*k1 + 2*Δt*k2, u0)
        return x0 + Δt/6*(k1 + 4*k2 + k3)
    end
end


module Implicit
    using ..SaltedDIRCOL: DiffFloat

    """
        hermite_simpson(dynamics, x0, u0, x1, Δt)

    Implicit integrator using the Hermite-Simpson method.
    """
    function hermite_simpson(
        dynamics::Function,
        x0::Vector{<:DiffFloat},
        u0::Vector{<:DiffFloat},
        x1::Vector{<:DiffFloat},
        Δt::DiffFloat
    )::Vector{<:DiffFloat}
        f0 = dynamics(x0, u0)
        f1 = dynamics(x1, u0)
        xc = 1/2*(x0 + x1) + Δt/8*(f0 - f1)
        fc = dynamics(xc, u0)
        return x0 - x1 + Δt/6*(f0 + 4*fc + f1)
    end
end


"""
"""
struct ExplicitIntegrator
    method::Function
end
ExplicitIntegrator(method_name::Symbol) = ExplicitIntegrator(
    get_module_function(Explicit, method_name)
)

"""
"""
struct ImplicitIntegrator
    method::Function
end
ImplicitIntegrator(method_name::Symbol) = ImplicitIntegrator(
    get_module_function(Implicit, method_name)
)
ImplicitIntegrator(explicit::ExplicitIntegrator) = ImplicitIntegrator(
    (
        dynamics::Function,
        x0::Vector{<:DiffFloat},
        u0::Vector{<:DiffFloat},
        x1::Vector{<:DiffFloat},
        Δt::DiffFloat
    ) -> explicit(dynamics, x0, u0, Δt) - x1
)

const Integrator = Union{ExplicitIntegrator, ImplicitIntegrator}

"""
"""
function (integrator::Integrator)(
    dynamics::Function,
    primals::Vararg{Union{Vector{<:DiffFloat}, DiffFloat}}
)::Vector{<:DiffFloat}
    return integrator.method(dynamics, primals...)
end
