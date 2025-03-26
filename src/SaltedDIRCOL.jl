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
        bouncing_ball

include("utils/types.jl")
include("utils/indexing.jl")
include("utils/utils.jl")
include("integrators.jl")
include("dynamics.jl")
include("models.jl")
include("objectives.jl")
include("constraints.jl")
include("solver.jl")
include("plotting.jl")

end # module SaltedDIRCOL
