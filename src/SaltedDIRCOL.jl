module SaltedDIRCOL

import ForwardDiff as FD
using JuMP
using Ipopt

export
    Transition,
    HybridMode,
    bouncing_ball

include("hybrid_system.jl")
include("models.jl")
include("integrators.jl")

end # module SaltedDIRCOL
