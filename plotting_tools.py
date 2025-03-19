import altair as alt
import numpy as np
import pandas as pd


def build_mh_charts(
    chain_df, 
    i, 
    p_true, 
    n_steps,
    burn_in=0, 
    std_dev=0.05, 
    animate_dist=False,
    dist_scale=0.5
):
    """
    Builds a chart (or charts) for:
      1) Histogram of accepted p-values (left),
      2) MCMC trace (right),
      3) A normal distribution "laid on its side" at the current iteration i 
         if animate_dist=True (overlaid on the trace).

    Args:
        chain_df (pd.DataFrame): M-H chain with columns 
            [iteration, state, candidate, accepted].
        i (int): Current iteration to display.
        p_true (float): True p-value for reference.
        n_steps (int): Total M-H steps (for x-axis domain).
        burn_in (int): Number of burn-in iterations to exclude.
        std_dev (float): Proposal distribution std. dev.
        animate_dist (bool): Whether to show the proposal distribution 
                             around the current state at iteration i.
        dist_scale (float): Factor to scale the pdf horizontally 
                            so it doesn't blow out the chart near iteration i.

    Returns:
        alt.Chart: A horizontal concatenation of 
                   (1) the histogram and 
                   (2) the layered trace + optional distribution.
    """
    # Filter data up to iteration i so the line grows over time
    filtered_df = chain_df[chain_df['iteration'] <= i].copy()
    
    # Add a column to identify burn-in vs post-burn-in points
    filtered_df['phase'] = 'Post Burn-in'
    if burn_in > 0:
        filtered_df.loc[filtered_df['iteration'] <= burn_in, 'phase'] = 'Burn-in'
    
    # Histogram of accepted proposal states - ONLY post-burn-in
    accepted_df = filtered_df[(filtered_df['accepted']) & (filtered_df['iteration'] > burn_in)]
    hist = alt.Chart(accepted_df).mark_bar(
        orient="horizontal",
        size=10,               # Thicker bars
        stroke='black',        # Outline color
        strokeWidth=1          # Outline thickness
    ).encode(
        y=alt.Y(
            'state:Q',
            bin=alt.Bin(maxbins=40, extent=[0, 1]),
            title='p',
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(values=[x/20 for x in range(21)], labelOverlap=False)
        ),
        x=alt.X('count()', title='Count'),
    ).properties(width=200, height=300, title="Histogram of Accepted p-values")

    # Red rule for the true p-value
    rule_df = pd.DataFrame({'p_true': [p_true]})
    true_p_rule = alt.Chart(rule_df).mark_rule(
        color='red',
        strokeDash=[6, 4]  # Creates a dashed line pattern: 6 pixels on, 4 pixels off
    ).encode(
        y=alt.Y('p_true:Q', scale=alt.Scale(domain=[0, 1]))
    )
    hist_with_rule = hist + true_p_rule
    
    # Chart trace
    fixed_x_domain = [1, n_steps]  # Start from 1 instead of burn_in+1
    
    # Split data for accepted and rejected points
    accepted_pts = filtered_df[filtered_df['accepted']]
    rejected_pts = filtered_df[~filtered_df['accepted']]
    
    # Split accepted points into burn-in and post-burn-in
    burn_in_pts = accepted_pts[accepted_pts['phase'] == 'Burn-in']
    post_burn_in_pts = accepted_pts[accepted_pts['phase'] == 'Post Burn-in']
    
    # Get the burn-in endpoint state if burn-in is active
    burn_in_end_state = None
    if burn_in > 0 and i >= burn_in:
        burn_in_end_state = chain_df[chain_df['iteration'] == burn_in]['state'].iloc[0]
    
    # Line for burn-in phase
    burn_in_line = alt.Chart(burn_in_pts).mark_line().encode(
        x=alt.X("iteration:Q", title="Iteration", scale=alt.Scale(domain=fixed_x_domain)),
        y=alt.Y("state:Q", title="p", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('phase:N', scale=alt.Scale(
            domain=['Burn-in', 'Post Burn-in'],
            range=['orange', 'blue']  # Orange for burn-in, blue for post-burn-in
        ))
    )
    
    # Line for post-burn-in phase
    post_burn_in_line = None
    if burn_in > 0 and i > burn_in and not post_burn_in_pts.empty:
        # Create a dataframe with the burn-in endpoint and all post-burn-in points
        connected_pts = pd.DataFrame({
            'iteration': [burn_in] + post_burn_in_pts['iteration'].tolist(),
            'state': [burn_in_end_state] + post_burn_in_pts['state'].tolist(),
            'phase': ['Post Burn-in'] * (len(post_burn_in_pts) + 1)
        })
        
        # Sort by iteration to ensure proper line drawing
        connected_pts = connected_pts.sort_values('iteration')
        
        # Create the post-burn-in line that connects to the burn-in endpoint
        post_burn_in_line = alt.Chart(connected_pts).mark_line().encode(
            x=alt.X("iteration:Q", scale=alt.Scale(domain=fixed_x_domain)),
            y=alt.Y("state:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.value('blue')  # Blue for post-burn-in
        )
    
    # Points for accepted states with color based on phase
    accepted_chart = alt.Chart(accepted_pts).mark_circle(size=60).encode(
        x=alt.X("iteration:Q", scale=alt.Scale(domain=fixed_x_domain)),
        y=alt.Y("state:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["iteration", "state"],
        color=alt.Color('phase:N', scale=alt.Scale(
            domain=['Burn-in', 'Post Burn-in'],
            range=['gray', 'blue']  # Gray for burn-in, blue for post-burn-in
        ))
    )

    accepted_line = alt.Chart(accepted_pts).mark_line().encode(
        x=alt.X("iteration:Q", scale=alt.Scale(domain=fixed_x_domain)),
        y=alt.Y("state:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('phase:N', scale=alt.Scale(
            domain=['Burn-in', 'Post Burn-in'],
            range=['orange', 'blue']  # Gray for burn-in, blue for post-burn-in
        )) 
    )

    accepted_chart = accepted_line + accepted_chart
    
    # Grey ticks for rejected proposals - use fixed colors instead of phase-based coloring
    # to prevent rejections inheriting colors from their associated line
    rejected_chart = alt.Chart(rejected_pts).mark_tick(thickness=2).encode(
        x=alt.X("iteration:Q", scale=alt.Scale(domain=fixed_x_domain)),
        y=alt.Y("candidate:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["iteration", "candidate"],
        color=alt.value('darkgray')  # Fixed color for all rejected points
    )
    
    # Highlight the current iteration in green
    current_iter_df = filtered_df[filtered_df['iteration'] == i]
    current_pt_chart = alt.Chart(current_iter_df).mark_circle(size=80, color='green').encode(
        x=alt.X("iteration:Q", scale=alt.Scale(domain=fixed_x_domain)),
        y=alt.Y("state:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["iteration", "state"]
    )
    
    # Add a red X at the burn-in endpoint if we're past burn-in
    burn_in_end_chart = None
    if burn_in > 0 and i > burn_in:
        # Create a red marker at the burn-in endpoint
        burn_in_end_df = pd.DataFrame({
            'iteration': [burn_in],
            'state': [burn_in_end_state]
        })
        
        # Create a red X marker at the burn-in endpoint
        burn_in_end_chart = alt.Chart(burn_in_end_df).mark_point(
            shape='circle',
            size=100,
            color='red',
            strokeWidth=3
        ).encode(
            x=alt.X("iteration:Q", scale=alt.Scale(domain=fixed_x_domain)),
            y=alt.Y("state:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["iteration", alt.Tooltip("state:Q", title="Burn-in End State")]
        )
    
    # Combine these trace layers
    trace_layers = burn_in_line
    if post_burn_in_line is not None:
        trace_layers += post_burn_in_line
    trace_layers += accepted_chart + rejected_chart + current_pt_chart
    if burn_in_end_chart is not None:
        trace_layers += burn_in_end_chart
    
    # Add graph for the proposal distribution which we pick samples from
    dist_layer = None
    if animate_dist and not current_iter_df.empty:
        current_p = current_iter_df['state'].iloc[0]
        
        # Generate a small range around the current p, clipped to [0, 1].
        lower = max(0, current_p - 3*std_dev)
        upper = min(1, current_p + 3*std_dev)
        if lower < upper:  # ensure we have a valid range
            theta_vals = np.linspace(lower, upper, 100)
            
            # Normal PDF
            pdf_vals = (1.0 / (std_dev * np.sqrt(2*np.pi))) * \
                       np.exp(-0.5 * ((theta_vals - current_p) / std_dev)**2)
            # Scale the PDF horizontally
            pdf_vals *= dist_scale
            
            # x = iteration i + scaled pdf, y = theta_vals
            dist_df = pd.DataFrame({
                'x': i + pdf_vals,
                'theta': theta_vals
            })
            
            dist_layer = alt.Chart(dist_df).mark_line(color='white', strokeWidth=0.1).encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=fixed_x_domain)),
                y=alt.Y('theta:Q', scale=alt.Scale(domain=[0, 1]))
            )
            
            # Optional text label near the peak
            peak_idx = np.argmax(pdf_vals)
            peak_x = dist_df['x'].iloc[peak_idx]
            peak_theta = dist_df['theta'].iloc[peak_idx]
            label_df = pd.DataFrame({
                'x': [peak_x],
                'theta': [peak_theta],
                'text': [f"N({current_p:.2f}, {std_dev:.2f})"]
            })
            text_layer = alt.Chart(label_df).mark_text(
                align='left',
                dx=5,
                color='orange'
            ).encode(
                x='x:Q',
                y='theta:Q',
                text='text:N'
            )
            
            dist_layer = dist_layer + text_layer

    # combine trace + dist if we built dist_layer
    if dist_layer is not None:
        trace_with_dist = (trace_layers + dist_layer).properties(
            width=400, height=300, 
            title=f"MH Trace (iterations 1-{i})"
        )
    else:
        trace_with_dist = trace_layers.properties(
            width=400, height=300, 
            title=f"MH Trace (iterations 1-{i})"
        )

    

    # Add a dashed line at the true p-value
    trace_with_rule = trace_with_dist + true_p_rule
    
    # Concatenate for final chart
    final_chart = alt.hconcat(hist_with_rule, trace_with_rule)
    return final_chart
