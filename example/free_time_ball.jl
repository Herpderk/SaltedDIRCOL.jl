using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using Revise
using SaltedDIRCOL

# Define elast bouncing ball model
system = bouncing_ball()

# Define stage and terminal cost functions
Q = 1e-3 * diagm([1.0, 1.0, 0.0, 0.0])
R = 1e-3 * I(system.nu)
Qf = 1e3 * Q
stage = (x,u) -> x'*Q*x + u'*R*u
terminal = x -> x'*Qf*x

# Define trajopt parameters
N = 40
Δtlb = 1e-3
Δtub = 1e-1
hs = ImplicitIntegrator(:hermite_simpson)
params = ProblemParameters(
    system, hs, stage, terminal, N; Δtlb=Δtlb, Δtub=Δtub
)

# Define reference trajectory and initial conditions
xic = [0.0; 2.0; 2.0; 5.0]
xgc = [5.0; 2.0; 0.0; 0.0]
xrefs = repeat(xgc, N)
urefs = zeros((N-1) * system.nu)

# Initial guess
Δt = 0.05
Δts = fill(Δt, N-1)
us = urefs
rk4 = ExplicitIntegrator(:rk4)
xs = roll_out(rk4, system, N, Δt, us, xic, :impact)
y0 = compose_trajectory(params.dims, params.idx, xs, us, Δts)
plot_2d_trajectory(params, (1,2), y0; title="Initial Guess")

# Define transition sequence and terminal guard
impact = system.transitions[:impact]
sequence = [
    TransitionTiming(5, impact),
    TransitionTiming(10, impact),
    TransitionTiming(15, impact),
]
term_guard = impact.guard

# Define solver callbacks and solve with Ipopt
cb = SolverCallbacks(
    params, sequence, term_guard, xrefs, urefs, xic;
    gauss_newton=true, salted=true
)
sol = ipopt_solve(params, cb, y0; max_iter=1000, tol=1e-6)

# Visualize
plot_2d_trajectory(params, (1,2), sol.x;)
nothing
