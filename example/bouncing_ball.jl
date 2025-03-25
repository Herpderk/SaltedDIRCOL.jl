using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using Revise
using SaltedDIRCOL

# Define hybrid system model
system = SaltedDIRCOL.bouncing_ball()

# Define trajopt parameters
Q = 1e0 * diagm([1.0, 1.0, 0.0, 0.0])
R = 1e-6 * I(system.nu)
Qf = 1e3 * Q
N = 100
Δt = 0.01
params = SaltedDIRCOL.ProblemParameters(
    SaltedDIRCOL.hermite_simpson, system, Q, R, Qf, N, Δt
)

# Define transition sequence and terminal guard
impact = system.transitions["impact"]
sequence = [
    SaltedDIRCOL.TransitionTiming(20, impact),
    SaltedDIRCOL.TransitionTiming(40, impact),
    SaltedDIRCOL.TransitionTiming(60, impact),
    SaltedDIRCOL.TransitionTiming(80, impact),
]
term_guard = impact.guard

# Define reference trajectory and initial conditions
xic = [0.0; 10.0; 10.0; 0.0]
xgc = [10.0; 5.0; 0.0; 0.0]
xrefs = repeat(xgc, N)
urefs = zeros((N-1) * system.nu)

# Define solver callbacks
cb = SaltedDIRCOL.SolverCallbacks(
    params, sequence, term_guard, xrefs, urefs, xic;
    gauss_newton=true
)

# Solve using Ipopt
y0 = zeros(params.dims.ny)
sol = SaltedDIRCOL.ipopt_solve(params, cb, y0)

# Visualize
SaltedDIRCOL.plot_2d_trajectory(params.dims, params.idx, (1,2), sol.x)
nothing
