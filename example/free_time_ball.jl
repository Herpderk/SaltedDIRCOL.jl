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
Δtlb = 0.8e-2
Δtub = 1.2e-2
hs = ImplicitIntegrator(:hermite_simpson)
params = ProblemParameters(hs, system, stage, terminal, N; Δtlb = Δtlb)

# Define transition sequence and terminal guard
impact = system.transitions[:impact]
sequence = [
    #TransitionTiming(10, impact),
    #TransitionTiming(20, impact),
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
    params, sequence, term_guard, xrefs, urefs, xic;
    gauss_newton=true
)

# Initial guess
Δt = 0.01
Δts = fill(Δt, N-1)
us = urefs
rk4 = ExplicitIntegrator(:rk4)
xs = roll_out(rk4, system, N, Δt, us, xic, :impact)
y0 = compose_trajectory(params.dims, params.idx, xs, us, Δts)
sol = ipopt_solve(params, cb, y0; max_iter=10000)

# Visualize
plot_2d_trajectory(params, (1,2), sol.x;)
nothing
