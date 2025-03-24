using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using SparseArrays
using Revise
using SaltedDIRCOL

# Define hybrid system model
system = SaltedDIRCOL.bouncing_ball()

# Define quadratic cost matrices
Q = 1e-6 * I(system.nx)
R = 1e0 * I(system.nu)
Qf = 1e6 * Q

# Define horizon parameters
N = 50
Δt = 0.01

# Define trajopt problem parameters
params = SaltedDIRCOL.ProblemParameters(
    SaltedDIRCOL.hermite_simpson, system, Q, R, Qf, N, Δt
)

# Define transition sequence and terminal guard
apex = system.transitions["apex"]
impact = system.transitions["impact"]
sequence = [
    SaltedDIRCOL.TransitionTiming(10, apex),
    SaltedDIRCOL.TransitionTiming(20, impact),
    SaltedDIRCOL.TransitionTiming(30, apex),
    SaltedDIRCOL.TransitionTiming(N-1, impact),
]
term_guard = impact.guard

# Define reference trajectory and initial conditions
xic = [0.0; 0.1; 1.0; 0.0]
xgc = [0.5; 0.2; 0.0; 0.0]
xrefs = repeat(xgc, N)
urefs = zeros((N-1) * system.nu)

# Define solver callbacks
callbacks = SaltedDIRCOL.SolverCallbacks(
    params, sequence, term_guard, xrefs, urefs, xic
)
ipopt_cb = SaltedDIRCOL.IpoptCallbacks(params, callbacks)

# Test callbacks
ytest = ones(params.dims.ny)
λtest = ones(length(callbacks.c(ytest)))
@time callbacks.Lc_hess(ytest, λtest)
