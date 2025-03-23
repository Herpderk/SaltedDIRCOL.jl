const RealValue = Union{Real, Vector}

"""
    Dimensions(N, nx, nu, nh)

Contains dimensions corresponding to the number of horizon time steps, states, inputs, time, and total decision variables. Time with 0 dimensions implies that the time step durations are fixed.
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
        if !(nh in (0, 1))
            error("Dimension of h must be 0 or !")
        end
        ny = N*nx + (N-1)*(nu + nh)
        return new(N, nx, nu, nh, ny)
    end
end

"""
    get_indices(dims, Δstart, Δstop)

Returns a range of indices given the [x, u, h] order of decision variables.
"""
function get_indices(
    dims::Dimensions,
    Δstart::Int,
    Δstop::Int
)::Vector{UnitRange{Int}}
    ny_per_step = dims.nx + dims.nu + dims.nh
    return [(1+Δstart : dims.nx+Δstop) .+ (k-1)*(ny_per_step) for k = 1:dims.N]
end

"""
    VariableIndices(dims)

Contains ranges of indices for getting instances of x, u, or h given the [x, u, h] order of decision variables.
"""
struct VariableIndices
    dims::Dimensions
    x::Vector{UnitRange{Int}}
    u::Vector{UnitRange{Int}}
    h::Union{Nothing, Vector{UnitRange{Int}}}
    function VariableIndices(
        N::Int,
        nx::Int,
        nu::Int,
        nh::Int
    )
        dims = Dimensions(N, nx, nu, nh)
        x_idx = get_indices(dims, 0, 0)
        u_idx = get_indices(dims, nx, nu)
        if nh == 1
            h_idx = get_indices(dims, nx+nu, nx+nu+nh)
        elseif nh == 0
            h_idx = nothing
        end
        return new(dims, x_idx, u_idx, h_idx)
    end
end

