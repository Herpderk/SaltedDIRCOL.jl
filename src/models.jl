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
    ballistic_flow = (x,u) -> [x[3:4]; 0.0; u[1] - g]

    # Define apex transition
    g_apex = x -> x[4]
    R_apex = x -> x
    apex = Transition(ballistic_flow, ballistic_flow, g_apex, R_apex)

    # Define impact transition
    g_impact = x -> x[2]
    R_impact = x -> [x[1:3]; -e * x[4]]
    impact = Transition(ballistic_flow, ballistic_flow, g_impact, R_impact)

    # Link transitions
    apex.next_transition = impact
    impact.next_transition = apex

    # Create hybrid system
    transitions = Dict(
        :apex => apex,
        :impact => impact
    )
    return HybridSystem(nx, nu, transitions)
end

"""
"""
function hopper(
    m1::Real = 5.0,    # body mass
    m2::Real = 1.0,    # foot mass
    e::Real = 0.0,     # coefficient of restitution of foot
    g::Real = 9.81     # acceleration due to gravity
)::HybridSystem
    # State space:
    #   body x, body y, foot x, foot y,
    #   body xdot, body ydot, foot xdot, foot ydot
    nx = 8
    nu = 2
    M = Diagonal([m1 m1 m2 m2])

    function get_unit_lengths(
        x::DiffVector
    )::Tuple{DiffVector, DiffVector}
        r1 = x[1:2]
        r2 = x[3:4]
        l1 = (r1[1]-r2[1]) / norm(r1-r2)
        l2 = (r1[2]-r2[2]) / norm(r1-r2)
        return (l1, l2)
    end

    function B_flight(
        x::DiffVector
    )::DiffMatrix
        l1, l2 = get_unit_lengths(x)
        return [l1  l2; l2 -l1; -l1 -l2; -l2  l1]
    end

    function B_stance(
        x::DiffVector
    )::DiffMatrix
        l1, l2 = get_unit_lengths(x)
        return [l1  l2; l2 -l1; zeros(2,2)]
    end

    function generalized_flow(
        control_allocation::Function,
        gravity::Vector{<:Real},
        x::DiffVector,
        u::DiffVector
    )::DiffVector
        B = control_allocation(x)
        vdot = gravity + M\(B*u)
        v = x[5:8]
        return [v; vdot]
    end

    # Define flows
    grav_flight = [0; -g; 0; -g]
    flight_flow = (x,u) -> generalized_flow(B_flight, grav_flight, x, u)
    grav_stance =  [0; -g; 0; 0]
    stance_flow = (x,u) -> generalized_flow(B_stance, grav_stance, x, u)

    # Define liftoff transition
    g_liftoff = x -> -x[4]              # Flipped vertical position of foot
    R_liftoff = x -> x                  # Identity reset
    liftoff = Transition(stance_flow, flight_flow, g_liftoff, R_liftoff)

    # Define impact transition
    g_impact = x -> x[4]                # Vertical position of foot
    R_impact = x -> [x[1:6]; e*x[7:8]]  # (In)elastic collision
    impact = Transition(flight_flow, stance_flow, g_impact, R_impact)

    # Create hybrid system
    transitions = Dict(
        :liftoff => liftoff,
        :impact => impact
    )
    return HybridSystem(nx, nu, transitions)
end
