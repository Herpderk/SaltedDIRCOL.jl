"""
"""
struct Dimensions
    N::Int
    nx::Int
    nu::Int
    nh::Int
    ny::Int
    function Dimensions(
        N::Int,
        nx::Int,
        nu::Int,
        nh::Int
    )
        ny = N*nx + (N-1)*(nu + nh)
        return new(N, nx, nu, nh, ny)
    end
end

"""
"""
function get_indices(
    dims::Dimensions,
    Δstart::Int,
    Δstop::Int
)::Vector{UnitRange{Int}}
    ny_per_step = dims.nx + dims.nu + dims.nh
    return [(1+Δstart : nx+Δstop) .+ (k-1)*(ny_per_step) for k = 1:N]
end

"""
"""
struct VariableIndices
    x::Vector{UnitRange{Int}}
    u::Vector{UnitRange{Int}}
    h::Union{nothing, Vector{UnitRange{Int}}}
    function VariableIndices(
        dims::Dimensions
    )
        nx, nu, nh = dims.nx, dims.nu, dims.nh
        x_idx = get_indices(dims, 0, 0)
        u_idx = get_indices(dims, nx, nu)
        if nh == 1
            h_idx = get_indices(dims, nx+nu, nx+nu+nh)
        elseif nh == 0
            h_idx = nothing
        else
            error("dimensions of t cannot be greater than 1!")
        end
        return new(x_idx, u_idx, h_idx)
    end
end

