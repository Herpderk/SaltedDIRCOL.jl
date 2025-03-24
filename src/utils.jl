const Value = Union{Real, Vector}

"""
    PrimalDimensions(N, nx, nu, nt)

Contains dimensions corresponding to the number of horizon time steps, states, inputs, time, and total decision variables. Time with 0 dimensions implies that the time step durations are fixed.
"""
struct PrimalDimensions
    N::Int
    nx::Int
    nu::Int
    nt::Int
    ny::Int
    function PrimalDimensions(
        N::Int,
        nx::Int,
        nu::Int,
        nt::Int
    )::PrimalDimensions
        if !(nt in (0, 1))
            error("Dimension of Δt must be 0 or 1!")
        end
        ny = N*nx + (N-1)*(nu + nt)
        return new(N, nx, nu, nt, ny)
    end
end

"""
    get_indices(dims, Δstart, Δstop)

Returns a range of indices given the [x, u, h] order of decision variables.
"""
function get_indices(
    dims::PrimalDimensions,
    N::Int,
    Δstart::Int,
    Δstop::Int
)::Vector{UnitRange{Int}}
    ny_per_step = dims.nx + dims.nu + dims.nt
    return [(1+Δstart : dims.nx+Δstop) .+ (k-1)*(ny_per_step) for k = 1:N]
end

"""
    PrimalIndices(dims)

Contains ranges of indices for getting instances of x, u, or Δt given the [x, u, Δt] order of decision variables.
"""
struct PrimalIndices
    x::Vector{UnitRange{Int}}
    u::Vector{UnitRange{Int}}
    Δt::Union{Nothing, Vector{UnitRange{Int}}}
    function PrimalIndices(
        dims::PrimalDimensions
    )::PrimalIndices
        x_idx = get_indices(dims, dims.N, 0, 0)
        u_idx = get_indices(dims, dims.N-1, dims.nx, dims.nu)
        if dims.nt == 1
            Δt_idx = get_indices(
                dims, dims.N-1, dims.nx+dims.nu, dims.nx+dims.nu+dims.nt)
        elseif dims.nt == 0
            Δt_idx = nothing
        end
        return new(x_idx, u_idx, Δt_idx)
    end
end

