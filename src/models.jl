"""
    bouncing_ball(e=1.0, g=9.81)

Returns the hybrid system model containing the modes and transitions of a planar (in)elastic bouncing ball.
"""
function bouncing_ball(
    e::Float64 = 1.0,
    g::Float64 = 9.81
)::HybridSystem
    # Model ballistic dynamics with thrust
    # State space: x, y, xdot, ydot
    nx = 4
    nu = 1
    ballistic_flow = (x,u) -> [x[3:4]; 0.0; u[1] - g]

    # Define apex transition
    g_apex = x -> x[4]
    R_apex = x -> x
    apex = Transition(ballistic_flow, ballistic_flow, g_apex, R_apex)

    # Define impact transition
    g_impact = x -> x[2]
    R_impact = x -> [x[1:3]; -e * x[4]]
    impact = Transition(ballistic_flow, ballistic_flow, g_impact, R_impact)

    # Create hybrid system
    transitions = Dict(
        "apex" => apex,
        "impact" => impact
    )
    return HybridSystem(nx, nu, transitions)
end
