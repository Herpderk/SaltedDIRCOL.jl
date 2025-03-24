module SaltedDIRCOL

using LinearAlgebra
using ForwardDiff
using Ipopt

include("utils.jl")
include("hybrid_system.jl")
include("integrators.jl")
include("objectives.jl")
include("models.jl")
include("solver.jl")
include("constraints.jl")

end # module SaltedDIRCOL
