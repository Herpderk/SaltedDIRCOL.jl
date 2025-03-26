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
        xrefs::Vector{Float64},
        urefs::Vector{Float64},
        y::Vector
    )::DiffFloat64
        xs, us = decompose_trajectory(idx, y)
        xerrs = xs[1 : end-dims.nx] - xrefs[1 : end-dims.nx]
        xf_err = xs[end-dims.nx+1 : end] - xrefs[end-dims.nx+1 : end]
        uerrs = us - urefs
        return (
            xerrs' * Qs * xerrs
            + uerrs' * Rs * uerrs
            + xf_err' * Qf * xf_err
        )
    end
    return quadratic_cost
end
