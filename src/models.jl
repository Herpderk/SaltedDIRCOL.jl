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
    dynamics = (x,u) -> [x[3:4]; 0.0; u[1] - g]

    # Define apex transition
    g_apex = x -> x[4]
    R_apex = x -> x
    apex = Transition(dynamics, dynamics, g_apex, R_apex)

    # Define impact transition
    g_impact = x -> x[2]
    R_impact = x -> [x[1:3]; -e * x[4]]
    impact = Transition(dynamics, dynamics, g_impact, R_impact)

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
    g::Real = 9.81,    # acceleration due to gravity
    Llb::Real = 0.5,
    Lub::Real = 1.5
)::HybridSystem
    # State space:
    #   body x, body y, foot x, foot y,
    #   body xdot, body ydot, foot xdot, foot ydot
    nx = 8
    nu = 2
    M = Diagonal([m1 m1 m2 m2])

    function get_length_vector(
        x::DiffVector
    )::DiffVector
        return x[1:2] - x[3:4]
    end

    function get_unit_length(
        x::DiffVector
    )::DiffVector
        L = get_length_vector(x)
        return L / norm(L)
    end

    function B_flight(
        x::DiffVector
    )::DiffMatrix
        L1, L2 = get_unit_length(x)
        return [L1  L2; L2 -L1; -L1 -L2; -L2  L1]
    end

    function B_stance(
        x::DiffVector
    )::DiffMatrix
        L1, L2 = get_unit_length(x)
        return [L1  L2; L2 -L1; zeros(2,2)]
    end

    function dynamics(
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

    # Define dynamics for each mode
    grav_flight = [0; -g; 0; -g]
    grav_stance =  [0; -g; 0; 0]
    flight_dynamics = (x,u) -> dynamics(B_flight, grav_flight, x, u)
    stance_dynamics = (x,u) -> dynamics(B_stance, grav_stance, x, u)

    # Define liftoff transition
    g_liftoff = x -> -x[4]              # Flipped vertical position of foot
    R_liftoff = x -> x                  # Identity reset
    liftoff = Transition(stance_dynamics, flight_dynamics, g_liftoff, R_liftoff)

    # Define impact transition
    g_impact = x -> x[4]                # Vertical position of foot
    R_impact = x -> [x[1:6]; e*x[7:8]]  # (In)elastic collision
    impact = Transition(flight_dynamics, stance_dynamics, g_impact, R_impact)

    # Link transitions
    liftoff.next_transition = impact
    impact.next_transition = liftoff

    # Create hybrid system
    transitions = Dict(
        :liftoff => liftoff,
        :impact => impact
    )

    # Define length inequality constraint functions
    glb = x -> Llb - norm(get_length_vector(x))
    gub = x -> -Lub + norm(get_length_vector(x))
    g = x -> [glb(x); gub(x)]
    g_stage = (x, u) -> g(x)
    g_term = x -> g(x)

    return HybridSystem(
        nx, nu, transitions;
        stage_ineq_constr=g_stage, term_ineq_constr=g_term
    )
end
