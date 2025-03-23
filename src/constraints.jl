"""
    assert_timings(idx, sequence)

Raises an error if a transition time step is not at least 2 more than the first time step or previous transition time step. Also raises an error if the final transition time step is >= the horizon length N.
"""
function assert_timings(
    idx::VariableIndices,
    sequence::Vector{TransitionTiming}
)::Nothing
    k = 1
    for timing = sequence
        timing.k < k+2 ? error(
            "Please have at least 2 time steps between transitions!") : nothing
        k = timing.k
    end
    sequence[end].k >= idx.dims.N ? error(
        "Final transition time step should be < horizon length N!") : nothing
end

"""
    get_variables(idx, k, y, h=nothing)

Returns a tuple of (decision) variables for computing dynamics defect residuals for a given time step.
"""
function get_variables(
    idx::VariableIndices,
    k::Int,
    y::RealValue,
    h::Union{Nothing, Real}
)::Tuple{RealValue, RealValue, RealValue, Union{Nothing, Real}}
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
    h::Union{Nothing, Real} = nothing
)::RealValue
    assert_timings(idx, sequence)

    # Init defect residuals and time step counter
    c = [zeros(idx.dims.nx) for k = 1:idx.dims.N-1]
    k_start = 1

    # Iterate over each hybrid mode and time step
    for timing = sequence
        for k = k_start : timing.k
            # Get states, inputs and time step duration
            x0, u0, x1, hval = get_variables(idx, k, y, h)
            if k == timing.k
                # Assume reset occurs at start of time step and integrate after
                xJ = timing.transition.reset(x0)
                c[k] = integrator(timing.transition.flow_J, xJ, u0, x1, hval)
            else
                # Integrate smooth dynamics
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
    assert_timings(idx, sequence)
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
    assert_timings(idx, sequence)

    # Init keepout residuals
    c = zeros(idx.dims.N - length(sequence))
    k_start = 1
    i = 1

    # Iterate over each hybrid mode and time step
    for timing = sequence
        for k = k_start : timing.k-2
            c[i] = timing.transition.guard(y[idx.x[k]])
            i += 1
        end
        # Update starting time step; skip time step with touchdown constraint
        k_start = timing.k
    end

    # Evaluate terminal guard residuals over remaining time steps
    if k_start < idx.dims.N
        for k = k_start : idx.dims.N
            c[i] = terminal_guard(y[idx.x[k]])
            i += 1
        end
    end
    return c
end
