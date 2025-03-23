module SaltedDIRCOL

import ForwardDiff as FD
using LinearAlgebra
using Ipopt

include("hybrid_system.jl")
include("models.jl")
include("integrators.jl")
include("objectives.jl")
include("constraints.jl")
include("utils.jl")

end # module SaltedDIRCOL
