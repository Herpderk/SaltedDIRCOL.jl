using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using Revise
using SaltedDIRCOL

# Define elast bouncing ball model
system = bouncing_ball()

# Define stage and terminal cost functions
Q = 1e0 * diagm([1.0, 1.0, 0.0, 0.0])
R = 1e-6 * I(system.nu)
Qf = 1e3 * Q
stage = (x,u) -> x'*Q*x + u'*R*u
terminal = x -> x'*Qf*x

# Define trajopt parameters
N = 50
Δtlb = 01e-3
hs = ImplicitIntegrator(:hermite_simpson)
params = ProblemParameters(hs, system, stage, terminal, N; Δtlb = Δtlb)

# Define transition sequence and terminal guard
impact = system.transitions["impact"]
sequence = [
    TransitionTiming(10, impact),
    TransitionTiming(20, impact),
    TransitionTiming(30, impact),
    TransitionTiming(40, impact),
]
term_guard = impact.guard

# Define reference trajectory and initial conditions
xic = [0.0; 10.0; 10.0; 0.0]
xgc = [5.0; 5.0; 0.0; 0.0]
xrefs = repeat(xgc, N)
urefs = zeros((N-1) * system.nu)

# Define solver callbacks
cb = SolverCallbacks(
    params, sequence, term_guard, xrefs, urefs, xic, xgc;
    gauss_newton=false
)

# Solve using Ipopt
#xs = roll_out()
xs = ones(params.dims.ny)
y0 = [repeat([xgc; 0.0; Δt], N-1); xgc]
sol = ipopt_solve(params, cb, y0; gauss_newton=false, max_iter=3000)

# Visualize
plot_2d_trajectory(
    params, (1,2), sol.x;
    xlim = (0, 20),
    ylim = (0, 20)
)
nothing
