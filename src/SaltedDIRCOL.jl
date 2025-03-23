module SaltedDIRCOL

import ForwardDiff as FD
using LinearAlgebra
using Ipopt

include("utils.jl")
include("hybrid_system.jl")
include("models.jl")
include("integrators.jl")
include("objectives.jl")
include("constraints.jl")

end # module SaltedDIRCOL
