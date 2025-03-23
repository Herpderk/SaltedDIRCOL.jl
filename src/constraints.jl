"""
    assert_final_time(idx, sequence)

Raises an error if the final time step in the transition sequence is greater than the horizon length N.
"""
function assert_final_time(
    idx::VariableIndices,
    sequence::Vector{TransitionTiming}
)::nothing
    if sequence[end].k > idx.dims.N
        error("Final time step should be >= the horizon length N!")
    end
end

"""
    get_variables(idx, k, y, h=nothing)

Returns a tuple of (decision) variables for computing dynamics defect residuals for a given time step.
"""
function get_variables(
    idx::VariableIndices,
    k::Int,
    y::RealValue,
    h::Union{nothing, RealValue}
)::Tuple{RealValue}
    x0 = y[idx.x[k]]
    u0 = y[idx.u[k]]
    x1 = y[idx.x[k+1]]
    hval = isnothing(h) ? y[idx.h[k]] : h
    return x0, u0, x1, hval
end

"""
    dynamics_defect(idx, integrator, sequence, y, h=nothing)

Computes dynamics defect residuals for a given integration scheme and transition sequence.
"""
function dynamics_defect(
    idx::VariableIndices,
    integrator::Function,
    sequence::Vector{TransitionTiming},
    y::RealValue,
    h::Union{nothing, Real} = nothing
)::RealValue
    assert_final_time(idx, sequence)

    # Init defect residuals and time step counter
    c = [zeros(idx.dims.nx) for k = 1:idx.dims.N-1]
    k_start = 0

    # Iterate over each hybrid mode and time step
    for timing = sequence
        for k = k_start : timing.k
            # Get states, inputs and time step duration
            x0, u0, x1, hval = get_variables(idx, k, y, h)
            # Evaluate defects; assume reset occurs at the start of time steps
            xJ = k == timing.k ? timing.transition.reset(x0) : x0
            if k == timing.k1
                xJ = timing.transition.reset(x0)
                c[k] = integrator(timing.transition.flow_J, xJ, u0, x1, hval)
            else
                c[k] = integrator(timing.transition.flow_I, x0, u0, x1, hval)
            end
        end
        # Update starting time step of next mode
        k_start = timing.k + 1
    end

    # Integrate over remaining time steps
    if k_start < idx.dims.N
        for k = k_start : idx.dims.N-1
            x0, u0, x1, hval = get_variables(idx, k, y, h)
            c[k] = integrator(sequence[end].transition.flow_J, x0, u0, x1, hval)
        end
    end
    return vcat(c...)
end

"""
    guard_touchdown(idx, sequence, y)

Computes guard "touchdown" residuals at time steps right before transitions.
"""
function guard_touchdown(
    idx::VariableIndices,
    sequence::Vector{TransitionTiming},
    y::RealValue
)::RealValue
    assert_final_time(idx, sequence)
    c = zeros(length(sequence))

    # Evaluate touchdown residuals right before transitions
    for (i, timing) in enumerate(sequence)
        c[i] = timing.transition.guard(y[idx.x[timing.k-1]])
    end
    return c
end

"""
    guard_keepout(idx, terminal_guard, sequence, y)

Computes guard "keep-out" residuals at every time step without touchdown. Requires an additional terminal guard for keep-out after the last transition.
"""
function guard_keepout(
    idx::VariableIndices,
    terminal_guard::Function,
    sequence::Vector{TransitionTiming},
    y::RealValue
)::RealValue
    assert_final_time(idx, sequence)

    # Init keepout residuals
    c = zeros(idx.dims.N - length(sequence))
    k_start = 0

    # Iterate over each hybrid mode and time step
    for timing = sequence
        for k = k_start : timing.k-2
            c[k] = timing.transition.guard(y[idx.x[k]])
        end
        # Update starting time step; skip time step with touchdown constraint
        k_start = timing.k
    end

    # Evaluate final guard residuals over remaining time steps
    if k_start < idx.dims.N
        for k = k_start : idx.dims.N
            c[k] = terminal_guard(y[idx.x[k]])
        end
    end
    return c
end
