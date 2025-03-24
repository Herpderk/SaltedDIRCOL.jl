using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using Revise
using SaltedDIRCOL

# Define hybrid system model
system = SaltedDIRCOL.bouncing_ball()

# Define quadratic cost matrices
Q = 1e-6 * I(system.nx)
R = 1e0 * I(system.nu)
Qf = 1e6 * Q

# Define horizon parameters
N = 100
Δt = 0.01

# Define trajopt problem parameters
params = SaltedDIRCOL.ProblemParameters(
    SaltedDIRCOL.hermite_simpson, system, Q, R, Qf, N, Δt
)

# Define transition sequence and terminal guard
apex = system.transitions["apex"]
impact = system.transitions["impact"]
sequence = [
    SaltedDIRCOL.TransitionTiming(25, apex),
    SaltedDIRCOL.TransitionTiming(50, impact),
    SaltedDIRCOL.TransitionTiming(75, apex),
    SaltedDIRCOL.TransitionTiming(N-1, impact),
]
term_guard = impact.guard

# Define reference trajectory and initial conditions
xic = [0.0; 0.1; 1.0; 0.0]
xgc = [0.5; 0.2; 0.0; 0.0]
xrefs = repeat(xgc, N)
urefs = zeros((N-1) * system.nu)

# Define solver callback functions and gradients/jacobians
callbacks = SaltedDIRCOL.SolverCallbacks(
    params, sequence, term_guard, xrefs, urefs, xic
)

@time size(callbacks.fgrad(zeros(params.dims.ny)))
@time size(callbacks.gjac(zeros(params.dims.ny)))
@time size(callbacks.hjac(zeros(params.dims.ny)))
