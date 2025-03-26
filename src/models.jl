"""
    bouncing_ball(e=1.0, g=9.81)

Returns the hybrid system model containing the modes and transitions of a planar (in)elastic bouncing ball.
"""
function bouncing_ball(
    e::Real = 1.0,
    g::Real = 9.81
)::HybridSystem
    # Model ballistic dynamics with thrust
    # State space: x, y, xdot, ydot
    nx = 4
    nu = 1
    function ballistic_flow(x::DiffVector, u::DiffVector)::DiffVector
        return [x[3:4]; 0.0; u[1] - g]
    end

    # Define apex transition
    function g_apex(x::DiffVector)::DiffFloat
        return x[4]
    end
    function R_apex(x::DiffVector)::DiffVector
        return x
    end
    apex = Transition(ballistic_flow, ballistic_flow, g_apex, R_apex)

    # Define impact transition
    function g_impact(x::DiffVector)::DiffFloat
        return x[2]
    end
    function R_impact(x::DiffVector)::DiffVector
        return [x[1:3]; -e * x[4]]
    end
    impact = Transition(ballistic_flow, ballistic_flow, g_impact, R_impact)

    # Link transitions
    apex.next_transition = impact
    impact.next_transition = apex

    # Create hybrid system
    transitions = Dict(
        "apex" => apex,
        "impact" => impact
    )
    return HybridSystem(nx, nu, transitions)
end
