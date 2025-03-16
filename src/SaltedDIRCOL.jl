module SaltedDIRCOL

using ForwardDiff
using JuMP
using Ipopt

export
    Transition,
    HybridMode,
    bouncing_ball

include("hybrid_system.jl")
include("models.jl")

end # module SaltedDIRCOL
