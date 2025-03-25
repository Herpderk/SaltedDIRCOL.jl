"""
    init_quadratic_cost(dims, idx, Q, R, Qf, xrefs, urefs)

Returns a quadratic cost function given a set of cost matrices.
"""
function init_quadratic_cost(
    dims::PrimalDimensions,
    idx::PrimalIndices,
    Q::Union{Matrix, Diagonal},
    R::Union{Matrix, Diagonal},
    Qf::Union{Matrix, Diagonal}
)::Function
    Nmat = I(dims.N - 1)
    Qs = kron(Nmat, Q)
    Rs = kron(Nmat, R)
    function quadratic_cost(
        xrefs::Vector,
        urefs::Vector,
        y::Vector
    )::Real
        # Get stage state errors
        xs = vcat([y[i] for i = idx.x[1 : end-1]]...)
        xs_err = xs - xrefs[1 : end-dims.nx]
        # Get terminal state error
        xf = y[idx.x[end]]
        xf_err = xf - xrefs[end-dims.nx+1 : end]
        # Get stage input errors
        us = vcat([y[i] for i = idx.u]...)
        us_err = us - urefs
        return xs_err'*Qs*xs_err + us_err'*Rs*us_err + xf_err'*Qf*xf_err
    end
    return quadratic_cost
end
