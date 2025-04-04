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
    ExplicitIntegrator(method_name)

Callable struct that instantiates its integration method based on the corresponding integrator in the `Explicit` module. When called on, returns the forward-simulated state given a set of primal variables.
"""
struct ExplicitIntegrator
    method::Function
end

ExplicitIntegrator(method_name::Symbol) = ExplicitIntegrator(
    get_module_function(Explicit, method_name)
)

"""
    ImplicitIntegrator(method_name)

Callable struct that instantiates its integration method based on the corresponding integrator in the `Implicit` module. When called on, returns the defect residuals between a set of primal variables and the future state. If given an `ExplicitIntegrator`, a defect residual function will be instantiated using said explicit integrator.
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
        Δt::AbstractFloat
    ) -> explicit(dynamics, x0, u0, Δt) - x1
)

const Integrator = Union{ExplicitIntegrator, ImplicitIntegrator}

"""
    integrator(dynamics, primals)

Callable struct method for the `ExplicitIntegrator` and `ImplicitIntegrator` structs. Returns either the forward-simulated state or defect residuals between states at adjacent time steps, respectively.
"""
function (integrator::Integrator)(
    dynamics::Function,
    primals::Vararg{Union{Vector{<:DiffFloat}, DiffFloat}}
)::Vector{<:DiffFloat}
    return integrator.method(dynamics, primals...)
end
