"""
"""
function init_quadratic_cost_function(
    N::Int,
    xrefs::Vector,
    urefs::Vector,
    Q::Union{Matrix, Diagonal},
    R::Union{Matrix, Diagonal},
    Qf::Union{Matrix, Diagonal}
)::Function
    Nmat = I(N-1)
    Qs = kron(Nmat, Q)
    Rs = kron(Nmat, R)
    nxs = length(xrefs)
    nx = Int(nxs / N)
    ny = nxs + length(urefs)
    function cost_function(y::Vector)
        xs = y[1 : nxs]
        xs_err = xs[1 : end - nx]
        xf_err = xs[end + 1 - nx : end]
        us = y[1+nxs : ny]
        us_err = us - urefs
        return xs_err'*Qs*xs_err + us_err'*Rs*us_err + xf_err'*Qf*xf_err
    end
    return cost_function
end
