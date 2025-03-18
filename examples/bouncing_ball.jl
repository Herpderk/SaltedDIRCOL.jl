using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using Nonconvex
Nonconvex.@load NLopt
using SaltedDIRCOL

# Define horizon parameters
Δt = 0.01
N = 100

# Define hybrid system model
hybrid_sys = SaltedDIRCOL.bouncing_ball()
nx = hybrid_sys.nx
nu = hybrid_sys.nu

# Define reference trajectory and initial conditions
xgoal = [0.5; 0.2; 0.0; 0.0]
xrefs = repeat(xgoal, N)
urefs = zeros((N-1) * nu)
xic = [0.0; 0.1; 1.0; 0.0]

# Init objective function
Q = 1e-6 * I(nx)
R = 1e0 * I(nu)
Qf = 1e6 * Q
J = SaltedDIRCOL.init_quadratic_cost_function(N, xrefs, urefs, Q, R, Qf)

# Init optimization model and decision variables
ny = N*nx + (N-1)*nu
model = Model(J)
addvar!(model, -Inf*ones(ny), Inf*ones(ny))

# Define mode sequence
sequence = [
    (25, hybrid_sys.transitions["apex"]),
    (50, hybrid_sys.transitions["impact"]),
    (75, hybrid_sys.transitions["apex"]),
    (N, hybrid_sys.transitions["impact"])
]

# Transcribe constraints via multiple shooting
for k = 1 : N-1
    transition_time = nothing
    transition = nothing
    for (transition_time, transition) in enumerate(sequence)
        if k > transition_time
            continue
        else
            break
        end
    end

    # Index current state, input, and next state
    x0_idx = 1 + (k-1)*nx : k*nx
    u0_idx = 1 + (k-1)*nu : k*nu
    x1_idx = 1 + k*nx : (k+1)*nx

    # Choose constraints based on current mode
    if k != N && k == transition_time
        #@constraint(model, transition.guard(x0) == 0.0)
        # Insert saltation matrix here
        #@operator(model, reset_map, 1, transition.reset, transition.saltation)
        #xJ = reset_map(x0)
        #x1c = SaltedDIRCOL.rk4(transition.flow_I, xJ, u0, Δt)
    else
        #@constraint(model, transition.guard(x0) >= 0.0)
        #x1c = SaltedDIRCOL.rk4(transition.flow_J, x0, u0, Δt)
    end
    #@constraint(model, x1c - x1 == zeros(nx))
end

# Solve OCP
