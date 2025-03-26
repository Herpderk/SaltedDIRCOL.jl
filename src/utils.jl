const DiffFloat64 = Union{Float64, ForwardDiff.Dual}

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
        !(nt in (0, 1)) ? error("Dimension of Δt must be 0 or 1!") : nothing
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

"""
    DualDimensions(ng, nh)

Contains dimensions of the problem constraints/dual variables. The value of ng corresponds to the inequality constraints, while nh is for the equality constraints. The value of nc is the sum of ng and nh.
"""
struct DualDimensions
    ng::Int
    nh::Int
    nc::Int
    function DualDimensions(
        ng::Int,
        nh::Int
    )::DualDimensions
        nc = ng + nh
        return new(ng, nh, nc)
    end
end

"""
    SparsityPattern(A)

Contains the number of non-zero values and their row/column coordinates for a given matrix A.
"""
struct SparsityPattern
    nzvals::Int
    row_coords::Vector{Int}
    col_coords::Vector{Int}
    function SparsityPattern(
        A::Matrix
    )::SparsityPattern
        row_idx, col_idx, vals = findnz(sparse(A))
        return new(length(vals), row_idx, col_idx)
    end
end

"""
"""
function decompose_trajectory(
    idx::PrimalIndices,
    y::Vector
)::Tuple{Vector, Vector}
    xs = vcat([y[i] for i = idx.x[1 : end]]...)
    us = vcat([y[i] for i = idx.u[1 : end]]...)
    return xs, us
end
