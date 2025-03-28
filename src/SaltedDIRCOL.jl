module SaltedDIRCOL

using LinearAlgebra
using SparseArrays
using ForwardDiff
using Ipopt
using Plots

export
        ProblemParameters,
        SolverCallbacks,
        ipopt_solve,
        ExplicitIntegrator,
        ImplicitIntegrator,
        TransitionTiming,
        plot_2d_trajectory,
        roll_out,
        compose_trajectory,
        decompose_trajectory,
        bouncing_ball,
        hopper

include("utils.jl")
include("indexing.jl")
include("integrators.jl")
include("dynamics.jl")
include("models.jl")
include("objectives.jl")
include("solver.jl")
include("constraints.jl")
include("plotting.jl")

end # module SaltedDIRCOL
