"""
"""
function plot_2d_trajectory(
    params::ProblemParameters,
    vis_state_idx::Tuple{Int, Int},
    y::Vector;
    animate::Bool = false,
    title::String = "System Trajectory",
    xlabel::String = "x",
    ylabel::String = "y",
    xlim::Tuple{Real, Real} = (0.0, 10.0),
    ylim::Tuple{Real, Real} = (0.0, 10.0),
    markershape::Symbol = :none,
    markercolor::Symbol = :blue,
    markersize::Real = 2.0,
    linecolor::Symbol = :blue,
    linewidth::Real = 2.0
)::Nothing
    for i = vis_state_idx
        !(i in 1:params.dims.nx) ? error("invalid state index!") : nothing
    end
    vis_states = [
        [y[params.idx.x[k]][vis_state_idx[i]]
        for k = 1:params.dims.N] for i = 1:2
    ]
    if !animate
        plt = plot(
            vis_states...,
            xlim = xlim,
            ylim = ylim,
            title = title,
            markershape = markershape,
            markercolor = markercolor,
            markersize = markersize,
            linecolor = linecolor,
            linewidth = linewidth,
            legend = false,
        )
        display(plt)
    else
    end
    return
end
