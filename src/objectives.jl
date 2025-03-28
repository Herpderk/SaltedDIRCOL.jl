"""
"""
struct TrajectoryCost
    dims::PrimalDimensions
    idx::PrimalIndices
    stage_cost::Function
    terminal_cost::Function
end

"""
"""
function (params::TrajectoryCost)(
    yref::Vector{<:AbstractFloat},
    y::Vector{<:DiffFloat}
)::DiffFloat
    J = 0.0
    for k in 1 : params.dims.N-1
        J += params.stage_cost(
            y[params.idx.x[k]] - yref[params.idx.x[k]],
            y[params.idx.u[k]] - yref[params.idx.u[k]]
        )
    end
    J += params.terminal_cost(
        y[params.idx.x[params.dims.N]] - yref[params.idx.x[params.dims.N]]
    )
    return J
end
