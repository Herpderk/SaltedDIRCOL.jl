module SaltedDIRCOL

using LinearAlgebra
using SparseArrays
using ForwardDiff
using Ipopt
using Plots

include("utils.jl")
include("hybrid_system.jl")
include("integrators.jl")
include("objectives.jl")
include("models.jl")
include("solver.jl")
include("constraints.jl")

end # module SaltedDIRCOL
