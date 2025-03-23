"""
    init_quadratic_cost(idx, dims, Q, R, Qf, xrefs, urefs)

Returns a quadratic cost function given a set of cost matrices.
"""
function init_quadratic_cost(
    idx::VariableIndices,
    Q::Union{Matrix, Diagonal},
    R::Union{Matrix, Diagonal},
    Qf::Union{Matrix, Diagonal},
    xrefs::RealValue,
    urefs::RealValue
)::Function
    nx, nu = idx.dims.nx, idx.dims.nu
    Nmat = I(idx.dims.N - 1)
    Qs = kron(Nmat, Q)
    Rs = kron(Nmat, R)
    function quadratic_cost(y::RealValue)
        xs = vcat([y[i] for i = idx.x]...)
        us = vcat([y[i] for i = idx.u]...)
        xs_err = xs[1 : end-nx] - xrefs[1 : end-nx]
        xf_err = xs[end-nx+1 : end] - xrefs[end-nx+1 : end]
        us_err = us[1 : end-nu] - urefs[1 : end-nu]
        return xs_err'*Qs*xs_err + us_err'*Rs*us_err + xf_err'*Qf*xf_err
    end
    return quadratic_cost
end
