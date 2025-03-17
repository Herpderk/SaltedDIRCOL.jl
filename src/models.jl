"""
Models the modes and transitions of a 2D elastic bouncing ball system.
Input:
    e - Float64 representing the coefficient of restitution
    g - Float64 representing the acceleration due to gravity
Output:
    model - HybridSystem with "upwards" and "downwards" mode keys
"""
function bouncing_ball(
    e::Float64 = 1.0,
    g::Float64 = 9.81
)::HybridSystem
    # Model ballistic dynamics with thrust
    # State space: x, y, xdot, ydot
    nx = 4
    nu = 1
    ballistic_flow = (x,u) -> [x[3:4]; 0.0; u - g]
    up_mode = HybridMode(ballistic_flow)
    down_mode = HybridMode(ballistic_flow)

    # Define apex transition
    g_apex = x -> x[4]
    R_apex = x -> x
    apex = Transition(up_mode, down_mode, g_apex, R_apex)
    up_mode.transitions = [apex]

    # Define impact transition
    g_impact = x -> x[2]
    R_impact = x -> [x[1:3], -e * x[4]]
    impact = Transition(down_mode, up_mode, g_impact, R_impact)
    down_mode.transitions = [impact]

    # Create hybrid system with labels
    key_mode_pairs = [
        ("upwards", up_mode),
        ("downwards", down_mode)
    ]
    return HybridSystem(key_mode_pairs, nx, nu)
end
