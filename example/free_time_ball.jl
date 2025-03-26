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
N = 50
Δtlb = 1e-3
params = SaltedDIRCOL.ProblemParameters(
    SaltedDIRCOL.hermite_simpson, system, Q, R, Qf, N; Δtlb = Δtlb
)

# Define transition sequence and terminal guard
impact = system.transitions["impact"]
sequence = [
    SaltedDIRCOL.TransitionTiming(10, impact),
    SaltedDIRCOL.TransitionTiming(20, impact),
    SaltedDIRCOL.TransitionTiming(30, impact),
    SaltedDIRCOL.TransitionTiming(40, impact),
]
term_guard = impact.guard

# Define reference trajectory and initial conditions
xic = [0.0; 10.0; 10.0; 0.0]
xgc = [5.0; 5.0; 0.0; 0.0]
xrefs = repeat(xgc, N)
urefs = zeros((N-1) * system.nu)

# Define solver callbacks
cb = SaltedDIRCOL.SolverCallbacks(
    params, sequence, term_guard, xrefs, urefs, xic;
    gauss_newton=true
)

# Solve using Ipopt
#xs = SaltedDIRCOL.roll_out()
xs = ones(params.dims.ny)
y0 = [repeat([xgc; 0.0; Δt], N-1); xgc]
sol = SaltedDIRCOL.ipopt_solve(params, cb, y0)

# Visualize
SaltedDIRCOL.plot_2d_trajectory(
    params, (1,2), sol.x;
    xlim = (0, 20),
    ylim = (0, 20)
)
nothing
