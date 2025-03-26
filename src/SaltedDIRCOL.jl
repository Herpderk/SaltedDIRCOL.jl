module SaltedDIRCOL

using LinearAlgebra
using SparseArrays
using ForwardDiff
using Ipopt
using Plots

include("utils.jl")
include("integrators.jl")
include("hybrid_system.jl")
include("models.jl")
include("objectives.jl")
include("solver.jl")
include("constraints.jl")
include("plotting.jl")

end # module SaltedDIRCOL
