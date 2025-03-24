"""
    assert_timings(params, sequence)

Raises an error if a transition time step is not at least 2 more than the first time step or previous transition time step. Also raises an error if the final transition time step is >= the horizon length N.
"""
function assert_timings(
    params::ProblemParameters,
    sequence::Vector{TransitionTiming}
)::Nothing
    k = 1
    for timing = sequence
        timing.k < k+2 ? error(
            "Please have at least 2 time steps between transitions!") : nothing
        k = timing.k
    end
    sequence[end].k >= params.dims.N ? error(
        "Final transition time step should be < horizon length N!") : nothing
end

"""
    get_primals(params, k, y, Δt)

Returns a tuple of (decision) variables for computing dynamics defect residuals for a given time step.
"""
function get_primals(
    params::ProblemParameters,
    k::Int,
    y::RealValue,
    Δt::Union{Nothing, Real}
)::Tuple{RealValue, RealValue, RealValue, Union{Nothing, Real}}
    x0 = y[params.idx.x[k]]
    u0 = y[params.idx.u[k]]
    x1 = y[params.idx.x[k+1]]
    Δt = isnothing(Δt) ? y[params.idx.Δt[k]] : Δt
    return x0, u0, x1, Δt
end

"""
    dynamics_defect(params, integrator, sequence, y, Δt=nothing)

Computes dynamics defect residuals for a given integration scheme and transition sequence.
"""
function dynamics_defect(
    params::ProblemParameters,
    sequence::Vector{TransitionTiming},
    y::RealValue,
    Δt::Union{Nothing, Real} = nothing
)::RealValue
    assert_timings(params, sequence)

    # Init defect residuals and time step counter
    c = [zeros(eltype(y), params.dims.nx) for k = 1:params.dims.N-1]
    k_start = 1

    # Iterate over each hybrid mode and time step
    for timing = sequence
        for k = k_start : timing.k
            # Get states, inputs and time step duration
            x0, u0, x1, hval = get_primals(params, k, y, Δt)
            if k == timing.k
                # Assume reset occurs at start of time step and integrate after
                xJ = timing.transition.reset(x0)
                c[k] = params.integrator(
                    timing.transition.flow_J, xJ, u0, x1, hval)
            else
                # Integrate smooth dynamics
                c[k] = params.integrator(
                    timing.transition.flow_I, x0, u0, x1, hval)
            end
        end
        # Update starting time step of next mode
        k_start = timing.k + 1
    end

    # Integrate over remaining time steps
    if k_start < params.dims.N
        for k = k_start : params.dims.N-1
            x0, u0, x1, hval = get_primals(params, k, y, Δt)
            c[k] = params.integrator(
                sequence[end].transition.flow_J, x0, u0, x1, hval)
        end
    end
    return vcat(c...)
end

"""
    guard_keepout(params, term_guard, sequence, y)

Computes guard "keep-out" residuals at every time step without touchdown. Requires an additional terminal guard for keep-out after the last transition.
"""
function guard_keepout(
    params::ProblemParameters,
    sequence::Vector{TransitionTiming},
    term_guard::Function,
    y::RealValue
)::RealValue
    assert_timings(params, sequence)

    # Init keepout residuals
    c = zeros(eltype(y), params.dims.N - length(sequence))
    k_start = 1
    i = 1

    # Iterate over each hybrid mode and time step
    for timing = sequence
        for k = k_start : timing.k-2
            # Flip the guard to adhere to NLP convention: g(x) <= 0
            c[i] = -timing.transition.guard(y[params.idx.x[k]])
            i += 1
        end
        # Update starting time step; skip time steps with touchdown constraint
        k_start = timing.k
    end

    # Evaluate terminal guard residuals over remaining time steps
    if k_start < params.dims.N
        for k = k_start : params.dims.N
            c[i] = term_guard(y[params.idx.x[k]])
            i += 1
        end
    end
    return c
end

"""
    guard_touchdown(params, sequence, y)

Computes guard "touchdown" residuals at time steps right before transitions.
"""
function guard_touchdown(
    params::ProblemParameters,
    sequence::Vector{TransitionTiming},
    y::RealValue
)::RealValue
    assert_timings(params, sequence)
    return [
        timing.transition.guard(y[params.idx.x[timing.k-1]])
        for timing in sequence
    ]
end

"""
    initial_condition(params, xic, y)

Computes initial condition residuals.
"""
function initial_condition(
    params::ProblemParameters,
    xic::RealValue,
    y::RealValue
)::RealValue
    return y[params.idx.x[1]] - xic
end

"""
    goal_condition(params, xg, y)

Computes goal condition residuals.
"""
function goal_condition(
    params::ProblemParameters,
    xg::RealValue,
    y::RealValue
)::RealValue
    return y[params.idx.x[params.dims.N]] - xg
end
