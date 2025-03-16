"""
Contains various pre-defined models of hybrid dynamical systems in the form of
Dict{String, HybridMode}.
"""
function generate_model(
    key_mode_pairs::Vector{Tuple{String, HybridMode}}
)::Dict{String, HybridMode}
    model = Dict()
    for (key, mode) = key_mode_pairs
        model[key] = mode
    end
    return model
end

function bouncing_ball(
    e::Float64 = 1.0,
    g::Float64 = 9.81
)::Dict{String, HybridMode}
    # Model ballistic dynamics with thrust
    # State space: x, y, xdot, ydot
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
    return generate_model([
        ("upwards", up_mode),
        ("downwards", down_mode)
    ])
end
