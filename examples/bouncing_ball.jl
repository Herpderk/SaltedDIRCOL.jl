using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using JuMP
using Ipopt
using SaltedDIRCOL

# Define hybrid system model
hybrid_sys = SaltedDIRCOL.bouncing_ball()
nx = hybrid_sys.nx
nu = hybrid_sys.nu

# Define horizon parameters
Δt = 0.01
N = 100

# Init optimizer and decision variables
model = Model(Ipopt.Optimizer)
@variables(model, begin
    xs[1 : N*nx]
    us[1 : (N-1)*nu]
end)

# Define reference trajectory and initial conditions
xgoal = [0.5; 0.2; 0.0; 0.0]
xrefs = repeat(xgoal, N)
urefs = zeros((N-1) * nu)
xic = [0.0; 0.1; 1.0; 0.0]

# Fix initial state
for (k, val) in enumerate(xic)
    fix(xs[k], val)
end

# Define objective function
Q = 1e-6 * I(nx)
R = 1e0 * I(nu)
Qf = 1e6 * Q
J = @objective(model, Min, SaltedDIRCOL.quadratic_cost(
    Q, R, Qf, xrefs, urefs, xs, us, N
))

# Define mode sequence
sequence = [
    (25, hybrid_sys.modes["upwards"].transitions[1]),
    (50, hybrid_sys.modes["downwards"].transitions[1]),
    (75, hybrid_sys.modes["upwards"].transitions[1]),
    (N, hybrid_sys.modes["downwards"].transitions[1])
]

# Transcribe constraints via multiple shooting
for k = 1 : N-1
    curr_transition_time = nothing
    curr_transition = nothing
    for (transition_time, transition) in sequence
        curr_transition_time = transition_time
        curr_transition = transition
        if k > transition_time
            continue
        else
            break
        end
    end

    # Index current state, input, and next state
    x0 = xs[1 + (k-1)*nx : k*nx]
    u0 = us[1 + (k-1)*nu : k*nu]
    x1 = xs[1 + k*nx : (k+1)*nx]

    # Discretize dynamics function
    x1c = SaltedDIRCOL.rk4(curr_transition.prev_mode.flow, x0, u0, Δt)
    if k != N & k == curr_transition_time
        # Insert saltation matrix here
        x1c = curr_transition.reset(x1c)
    end

    # Set defect constraint
    constr = @constraint(model, x1c - x1 == zeros(nx))
end

# Solve OCP
optimize!(model)
@show value.(us)
