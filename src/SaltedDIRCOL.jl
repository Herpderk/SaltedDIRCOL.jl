module SaltedDIRCOL

import ForwardDiff as FD
using LinearAlgebra
using Nonconvex

include("hybrid_system.jl")
include("models.jl")
include("integrators.jl")
include("objectives.jl")

end # module SaltedDIRCOL
