"""
"""
function quadratic_cost(
    Q::Union{Matrix, Diagonal},
    R::Union{Matrix, Diagonal},
    Qf::Union{Matrix, Diagonal},
    xrefs::Vector,
    urefs::Vector,
    xs::Union{Vector, VariableRef},
    us::Union{Vector, VariableRef},
    N::Int
)
    Nmat = I(N-1)
    Qs = kron(Nmat, Q)
    Rs = kron(Nmat, R)

    us_err = us - urefs
    all_xs_err = xs - xrefs

    xs_err = all_xs_err[1 : Int(end - length(xs)/N)]
    xf_err = all_xs_err[Int(end + 1 - length(xs)/N) : end]
    return xs_err'*Qs*xs_err + us_err'*Rs*us_err + xf_err'*Qf*xf_err
end
