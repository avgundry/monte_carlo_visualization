# Library imports
import altair as alt
import numpy as np
import streamlit as st
import pandas as pd
import time
import streamlit.components.v1 as components

# Project-level imports
from streamlit_text_blocks import text_block_1, text_block_2, text_block_3, volatility_text_block, volatility_text_block2, strike_price_text_block, strike_price_text_block2
import full_plot_test
import num_paths
import num_iterations
from mcmc import simulate_coin_flips, mh_chain
from plotting_tools import build_mh_charts


st.set_page_config(
    page_title="Monte Carlo Option Pricing",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Initialize session state for simulation data
if "simulation_data" not in st.session_state:
    st.session_state.simulation_data = None
if "mh_chain_df" not in st.session_state:
    st.session_state.mh_chain_df = None
if "burn_mh_chain_df" not in st.session_state:
    st.session_state.burn_mh_chain_df = None

# Sidebar navigation
st.sidebar.title("Navigation")
selected_section = st.sidebar.radio("Go to section:", [
    "Introduction",
    "Options Basics",   
    "Monte Carlo Simulations",
    "Paths",
    "Iterations",
    "Burn In",
    "Volatility",
    "Drift",
    "Conclusion"
])


def intro_section():
    text_block_1()
    full_plot_test.display_monte_carlo_graph()
    
def basics_section():
    text_block_2()
    strike_price_section()

def coin_simulation_section():
    # Initialize session state for simulation section (keys not directly linked to widgets)
    if "simulation_data" not in st.session_state:
        st.session_state.simulation_data = None
    if "current_flip" not in st.session_state:
        st.session_state.current_flip = 1
    if "p_default" not in st.session_state:
        st.session_state.p_default = 0.5

    # --- Simulation Section ---
    st.header("Coin Flip Simulation Section")

    # Simulation parameter widgets with unique keys.
    n_flips = st.slider("Flips", min_value=10, max_value=5000, value=1000, step=10, key="n_flips_widget")
    p = st.slider("p", min_value=0.0, max_value=1.0, value=st.session_state.p_default, step=0.01, key="p_widget")

    # Animation speed slider: iterations per second
    animation_speed = st.slider("Animation Speed (iterations per second)",
                                min_value=1, max_value=60, value=20, step=1, key="animation_speed")

    # Control buttons arranged in columns.
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Randomize p", key="randomize_p"):
            st.session_state.p_default = np.random.rand()
            st.rerun()  # Refresh the app to update the p slider with new default
    with col2:
        if st.button("Run Simulation", key="run_simulation"):
            coin_flips_df, num_heads, num_tails = simulate_coin_flips(p, n_flips)
            st.session_state.simulation_data = coin_flips_df
            st.session_state.current_flip = 1
            st.write("True probability $p$: ", p)
            st.write("Current proportion of heads: ", num_heads / n_flips)
            st.rerun()  # Refresh to initialize simulation



    # Function to create the Altair chart based on the current flip value.
    def get_chart(flip):
        df_subset = st.session_state.simulation_data[
            st.session_state.simulation_data["Flip Number"] <= flip
        ]
        line_chart = alt.Chart(df_subset).mark_line().encode(
            x=alt.X("Flip Number:Q", title="Number of Flips", scale=alt.Scale(domain=[0, n_flips])),
            y=alt.Y("Running Average:Q", title="Proportion of Heads", scale=alt.Scale(domain=[0, 1]))
        ).properties(
            width=600,
            height=300,
            title=f"Running Average up to Flip {flip} (True p = {p:.2f})"
        )
        # Horizontal rule for the true probability
        rule = alt.Chart(pd.DataFrame({"True p": [p]})).mark_rule(
            color='red', strokeDash=[4, 2]
        ).encode(
            y=alt.Y("True p:Q", scale=alt.Scale(domain=[0, 1]))
        )
        return line_chart + rule

    # If simulation data exists, show animation controls.
    # If empty, display an empty container to prevent flickering.
    if st.session_state.simulation_data is not None:
        st.subheader("Simulation Animation")
        
        # Create a container for the simulation animation controls.
        with st.container():
            # If the "Animate" button is clicked, run the animation loop.
            if st.button("Animate", key="animate"):
                # Create placeholders for the "Current Flip" slider and the chart.
                flip_slider_placeholder = st.empty()
                chart_placeholder = st.empty()
                
                # Animate from the current flip to the total number of flips.
                for flip in range(st.session_state.current_flip, n_flips + 1):
                    st.session_state.current_flip = flip
                    # Update the slider placeholder without specifying a key to avoid duplicate keys.
                    flip_slider_placeholder.slider("Current Flip", min_value=1,
                                                    max_value=n_flips, value=flip)
                    # Update the chart placeholder.
                    chart_placeholder.altair_chart(get_chart(flip))
                    # Pause between iterations based on the chosen animation speed.
                    # time.sleep(1 / animation_speed)
                st.session_state.current_flip = n_flips
                st.rerun()  # Final refresh after animation completes
            else:
                # When not animating, display a static "Current Flip" slider and chart.
                current_flip = st.slider("Current Flip", min_value=1, max_value=n_flips,
                                        value=st.session_state.current_flip, key="current_flip_static")
                st.altair_chart(get_chart(current_flip))

    else:
        st.empty()

def Monte_Carlo_section():
    text_block_3()

    if "true_p" not in st.session_state:
        st.session_state.true_p = np.random.uniform(0, 1)
        st.session_state.simulation_result = simulate_coin_flips(st.session_state.true_p, 1000)

    # Button to rerun the simulation
    

    # Retrieve stored values
    true_p = st.session_state.true_p
    initial_coin_df, num_heads, num_tails = st.session_state.simulation_result

    # Display results
    st.write(f"### True probability of landing heads: {true_p:.4f}")
    st.write(f"Number of heads: {num_heads}, number of tails: {num_tails}")
    st.write(f"Empirical estimate of $p$: $\\frac{{{num_heads}}}{{{1000}}} = {num_heads / 1000:.4f}$")

    if st.button("Run Simulation"):
        st.session_state.true_p = np.random.uniform(0, 1)  # Generate new probability
        st.session_state.simulation_result = simulate_coin_flips(st.session_state.true_p, 1000)  # New flips
        st.rerun()  # Force UI to refresh with new results
        
    coin_simulation_section()
    rejection_sampling_section()

def rejection_sampling_section():
    st.header("Rejection Sampling Section")

    st.write("However, in this case we know the exact distribution of the coin - we can just flip it ourselves - so it's easy for us to compute the probability of heads. What if we didn't have the coin, or were dealing with a more complex set of variables? In this case, we can use something called the Metropolis-Hastings algorithm to estimate the probability of an event happening - in case of options, the probability of the stock price at expiration being above the strike price.")

    st.write("To do this, we have to use something called rejection sampling. What this does is draw samples from a normal distribution, and then accept or reject those samples based on whether they are above or below the *target* distribution - what we expect the upper and lower bounds of the true distribution to be. For example, if we were to randomly draw a number greater than 1 from the normal distribution when trying to estimate the probability of heads, we would reject it, because we know the probability of heads being greater than 1 is 0.")

    st.write("A visualization of this is shown below, where we use rejection sampling to estimate the true probability of heads using a Monte Carlo simulation:")

    if "mh_chain_df" not in st.session_state:
        st.session_state.mh_chain_df = None
    if "coin_data" not in st.session_state:
        st.session_state.coin_data = {'p_true': 0.6, 'N': 1000, 'heads_count': 0, 'tails_count': 0}
    if "mh_current_iteration" not in st.session_state:
        st.session_state.mh_current_iteration = 1
    if "mh_n_steps" not in st.session_state:
        st.session_state.mh_n_steps = 100  # default value for MH steps

    # MH Simulation Section UI
    st.header("Metropolis-Hastings MCMC Simulation")

    # Dat ageneration
    st.subheader("Data Generation Controls")
    col_data1, col_data2 = st.columns(2)
    with col_data1:
        mh_true_p = st.slider("True p", min_value=0.0, max_value=1.0,
                                value=st.session_state.coin_data['p_true'], step=0.01, key="mh_true_p")
        if st.button("Randomize p", key="mh_randomize_p"):
            st.session_state.coin_data['p_true'] = np.random.rand()
            st.rerun()
    with col_data2:
        mh_data_flips = st.slider("Data flips", min_value=10, max_value=5000,
                                value=st.session_state.coin_data['N'], step=10, key="mh_data_flips")

    if st.button("Generate Data", key="mh_generate_data"):
        initial_coin_df, num_heads, num_tails = simulate_coin_flips(st.session_state.coin_data['p_true'], mh_data_flips)
        st.session_state.coin_data['N'] = mh_data_flips
        st.session_state.coin_data['heads_count'] = num_heads
        st.session_state.coin_data['tails_count'] = num_tails
        st.success(f"Generated data with p_true={st.session_state.coin_data['p_true']:.2f}, N={mh_data_flips}. Heads={num_heads}, Tails={num_tails}")

    # --- MCMC Controls ---
    st.subheader("MCMC Controls")
    col_mcmc1, col_mcmc2 = st.columns(2)
    with col_mcmc1:
        mh_n_steps = st.slider("MH steps", min_value=10, max_value=5000,
                            value=st.session_state.mh_n_steps, step=10, key="mh_n_steps")
    with col_mcmc2:
        mh_proposal_width = st.slider("Proposal σ", min_value=0.01, max_value=0.2,
                                    value=0.05, step=0.01, key="mh_proposal_width")

    if st.button("Run MCMC", key="mh_run_mcmc"):
        if st.session_state.coin_data['heads_count'] == 0 and st.session_state.coin_data['tails_count'] == 0:
            st.error("Please generate coin data first.")
        else:
            st.session_state.mh_chain_df = mh_chain(
                mh_n_steps,
                st.session_state.coin_data['heads_count'],
                st.session_state.coin_data['tails_count'],
                proposal_width=mh_proposal_width,
                start_p=0.5
            )
            st.session_state.mh_current_iteration = 1
            st.success("MH chain generated. Use the slider or 'Animate' to view the chain.")
            st.rerun()


    st.info(
        "Instructions:\n"
        "1) Choose/Randomize True $p$, set number of data flips, and click 'Generate Data'.\n"
        "2) Adjust MH steps and proposal $\\sigma$, then click 'Run MCMC'.\n"
        "3) Use the 'Iteration' slider or click 'Animate' to view the MH chain."
    )

    # --- MH Chain Animation ---
    if st.session_state.mh_chain_df is not None:
        st.subheader("MH Chain Animation")
        # TODO: Verify working?
        # Animation speed: iterations per second
        mh_anim_speed = st.slider("Animation Speed (iterations per second)",
                                min_value=1, max_value=50, value=10, step=1, key="mh_anim_speed")
        
        # Placeholders for the iteration slider and chart
        slider_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Static iteration slider (when not animating)
        current_iter = st.slider("Iteration", min_value=1,
                                max_value=st.session_state.mh_chain_df['iteration'].max(),
                                value=st.session_state.mh_current_iteration, key="mh_iteration_static")
        
        # Display the chart for the current iteration with a dynamic x-axis
        chart = build_mh_charts(
            st.session_state.mh_chain_df, 
            current_iter, 
            st.session_state.coin_data['p_true'], 
            mh_n_steps,
            std_dev=mh_proposal_width
        )
        chart_placeholder.altair_chart(chart)
        
        # Animate button: animate chain from the current iteration to the final iteration.
        if st.button("Animate", key="mh_animate"):
            max_steps = mh_n_steps
            # Set update_interval: update chart only every few iterations to improve performance.
            update_interval = 1 if max_steps < 500 else int(max_steps / 100)
            for i in range(current_iter, max_steps + 1):
                st.session_state.mh_current_iteration = i
                # Update the iteration slider placeholder without a fixed key.
                slider_placeholder.slider("Iteration", min_value=1,
                                            max_value=max_steps,
                                            value=i)
                # Only update the chart if it's the final iteration or at the update interval.
                if i % update_interval == 0 or i == max_steps:
                    chart = build_mh_charts(
                        st.session_state.mh_chain_df, 
                        i, 
                        st.session_state.coin_data['p_true'], 
                        max_steps,
                        std_dev=mh_proposal_width
                    )
                    chart_placeholder.altair_chart(chart)
                time.sleep(1 / mh_anim_speed)
            st.session_state.mh_current_iteration = max_steps
            st.rerun()


def burn_in_section():
    st.write("# Burn-In Section")

    st.write("When running the simulation repeatedly, early samples can overly influence the estimate because they are averaged over a small number of flips. By discarding an initial 'burn-in' period, we let the chain settle before using its samples. The visualization below shows the MH chain with the burn-in phase excluded.")

    # Initialize session state for the burn-in section with unique keys.
    if "burn_mh_chain_df" not in st.session_state:
        st.session_state.burn_mh_chain_df = None
    if "burn_coin_data" not in st.session_state:
        st.session_state.burn_coin_data = {'p_true': 0.5, 'N': 1000, 'heads_count': 0, 'tails_count': 0}
    if "burn_mh_current_iteration" not in st.session_state:
        st.session_state.burn_mh_current_iteration = 1
    

    # Set defaults for sliders below.
    burn_mh_n_steps_default = 100
    burn_mh_burn_in_default = 10
    burn_mh_proposal_width_default = 0.05

    st.header("Metropolis-Hastings MCMC Simulation with Burn-in")

    # --- Data Generation Controls ---
    st.subheader("Data Generation Controls")
    col_data1, col_data2 = st.columns(2)
    with col_data1:
        if st.button("Randomize p", key="burn_mh_randomize_p"):
            st.session_state.burn_coin_data['p_true'] = np.random.rand()
            st.rerun()
    with col_data2:
        burn_data_flips = st.slider("Data flips", min_value=1, max_value=5000,
                                value=st.session_state.burn_coin_data['N'], step=1, key="burn_mh_data_flips")

    if st.button("Generate Data", key="burn_mh_generate_data"):
        _, num_heads, num_tails = simulate_coin_flips(st.session_state.burn_coin_data['p_true'], burn_data_flips)
        st.session_state.burn_coin_data['N'] = burn_data_flips
        st.session_state.burn_coin_data['heads_count'] = num_heads
        st.session_state.burn_coin_data['tails_count'] = num_tails
        st.success(f"Generated data: p_true={st.session_state.burn_coin_data['p_true']:.2f}, N={burn_data_flips}. Heads={num_heads}, Tails={num_tails}")

    # Controls for MCMC sim
    st.subheader("MCMC Controls")
    col_mcmc1, col_mcmc2, col_mcmc3 = st.columns(3)
    with col_mcmc1:
        burn_n_steps = st.slider("MH steps", min_value=1, max_value=5000,
                            value=burn_mh_n_steps_default, step=1, key="burn_mh_n_steps")
    with col_mcmc2:
        burn_proposal_width = st.slider("Proposal σ", min_value=0.01, max_value=0.2,
                                    value=burn_mh_proposal_width_default, step=0.01, key="burn_mh_proposal_width")
    with col_mcmc3:
        # Do not reassign st.session_state.burn_mh_burn_in here; gives an error.
        burn_burn_in = st.slider("Burn-in steps", min_value=0, max_value=burn_n_steps - 1,
                            value=burn_mh_burn_in_default, step=1, key="burn_mh_burn_in")

    if st.button("Run MCMC", key="burn_mh_run_mcmc"):
        if st.session_state.burn_coin_data['heads_count'] == 0 and st.session_state.burn_coin_data['tails_count'] == 0:
            st.error("Please generate coin data first.")
        else:
            st.session_state.burn_mh_chain_df = mh_chain(
                burn_n_steps,
                st.session_state.burn_coin_data['heads_count'],
                st.session_state.burn_coin_data['tails_count'],
                proposal_width=burn_proposal_width,
                start_p=0.5
            )
            st.session_state.burn_mh_current_iteration = 1
            st.success("MH chain generated. Use the slider or 'Animate' to view the chain.")
            st.rerun()

    # MH Chain Animation 
    if st.session_state.burn_mh_chain_df is not None:
        st.subheader("MH Chain Animation")
        burn_anim_speed = st.slider("Animation Speed (iterations per second)",
                                min_value=1, max_value=50, value=10, step=1, key="burn_mh_anim_speed")
        
        # Placeholders for the iteration slider and chart.
        slider_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Initialize the slider with current iteration value
        current_iter = slider_placeholder.slider("Iteration", 
                                min_value=1,
                                max_value=st.session_state.burn_mh_chain_df['iteration'].max(),
                                value=st.session_state.burn_mh_current_iteration, 
                                key="burn_mh_iteration")
        
        # Update session state with the slider value
        st.session_state.burn_mh_current_iteration = current_iter
        
        # Display the chart for the current iteration with burn-in applied
        chart = build_mh_charts(
            st.session_state.burn_mh_chain_df, 
            current_iter, 
            st.session_state.burn_coin_data['p_true'], 
            burn_n_steps,
            burn_in=burn_burn_in,
            std_dev=burn_proposal_width
        )
        chart_placeholder.altair_chart(chart)
        
        if st.button("Animate", key="burn_mh_animate"):
            max_steps = burn_n_steps
            update_interval = 1 if max_steps < 500 else int(max_steps / 100)
            for i in range(current_iter, max_steps + 1):
                st.session_state.burn_mh_current_iteration = i
                # Update the iteration slider in the placeholder
                slider_placeholder.slider("Iteration", 
                                        min_value=burn_burn_in + 1,
                                        max_value=max_steps,
                                        value=i,
                                        key=f"burn_mh_iteration_anim_{i}")  # Dynamic key during animation
                if i % update_interval == 0 or i == max_steps:
                    chart = build_mh_charts(
                        st.session_state.burn_mh_chain_df, 
                        i, 
                        st.session_state.burn_coin_data['p_true'], 
                        max_steps,
                        burn_in=burn_burn_in,
                        std_dev=burn_proposal_width,
                        animate_dist=True
                    )
                    chart_placeholder.altair_chart(chart)
                time.sleep(1 / burn_anim_speed)
            st.session_state.burn_mh_current_iteration = max_steps
            st.rerun()

    st.info(
        "Instructions:\n"
        "1) Set True $p$ and number of data flips, then click 'Generate Data'.\n"
        "2) Adjust MH steps, proposal $\\sigma$, and Burn-in, then click 'Run MCMC'.\n"
        "3) Use the 'Iteration' slider or click 'Animate' to view the chain (burn-in excluded)."
    )
    

def volatility_section():
    # TODO: Move volatility calculation to top of graph so it can be seen 
    # at the same time as the graph without scrolling.
    # TODO: Fix legend so it doesn't run off the screen.
    volatility_text_block()

    # JavaScript and HTML for interactive chart in D3
    html_code = """
    <div id="d3-container" style="width: 100%; height: 600px;"></div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
        <div id="volatility-display" style="color: white; font-size: 18px;">Volatility (σ): 0.000</div>
        <button id="reset-button" style="background-color: red; color: white; font-size: 16px; border: none; padding: 8px 12px; cursor: pointer; border-radius: 5px;">Reset</button>
    </div>

    <script src="https://d3js.org/d3.v6.min.js"></script>
    <script>
    const width = window.innerWidth * 0.95, height = 550;
    const margin = { top: 20, right: 120, bottom: 50, left: 70 };

    // Create SVG container
    const svg = d3.select("#d3-container")
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // Default stock price data 
    let originalData = Array.from({length: 30}, (_, i) => ({ day: i + 1, price: 50 }));
    let data = JSON.parse(JSON.stringify(originalData)); 

    // Volatility data 
    let volatilityData = Array.from({length: 30}, (_, i) => ({ day: i + 1, vol: 0 }));

    // Scales
    const x = d3.scaleLinear().domain([1, 30]).range([0, width - margin.left - margin.right]);
    const y = d3.scaleLinear().domain([0, 100]).range([height - margin.top - margin.bottom, 0]);

    // Axes 
    svg.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x))
      .attr("color", "white");

    svg.append("g")
      .call(d3.axisLeft(y))
      .attr("color", "white");

    // Graph label
    svg.append("text")
        .attr("x", (margin.left - margin.right) / 2)
        .attr("y", -8)
        .attr("fill", "white")
        .style("text-anchor", "left")
        .style("font-size", "16px")  
        .style("font-weight", "bold")
        .text("Volatility Calculation from Stock Price");

    // X-axis label
    svg.append("text")
        .attr("x", (width - margin.left - margin.right) / 2)
        .attr("y", height - 23)  
        .attr("fill", "white")
        .style("text-anchor", "middle")
        .style("font-size", "16px")
        .text("Days");



    // Y-axis label
    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -50)
      .attr("x", -height / 2 + margin.top)
      .attr("dy", "1em")
      .style("fill", "white")
      .style("text-anchor", "middle")
      .style("font-size", "16px")
      .text("Stock Price ($) & Volatility (σ)");

    const line = d3.line().x(d => x(d.day)).y(d => y(d.price));

    // Stock price line 
    const path = svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "#FF0000")  
      .attr("stroke-width", 2)
      .attr("d", line);

    // Volatility line 
    const volatilityLine = svg.append("path")
      .datum(volatilityData)
      .attr("fill", "none")
      .attr("stroke", "#00FFFF")  
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5")
      .attr("d", d3.line().x(d => x(d.day)).y(d => y(d.vol)));

    // Tooltip div 
    const tooltip = d3.select("body")
      .append("div")
      .style("position", "absolute")
      .style("background", "black")
      .style("color", "white")
      .style("padding", "8px 10px")
      .style("border-radius", "8px")
      .style("visibility", "hidden")
      .style("font-size", "14px")
      .style("font-family", "Arial, sans-serif")
      .style("box-shadow", "0px 4px 6px rgba(0,0,0,0.3)");

    // Define minimum and maximum allowed values for price points
    const MIN_PRICE = 1;  // Minimum allowed price
    const MAX_PRICE = 99; // Maximum allowed price

    //  Draggable circles 
    let circles = svg.selectAll("circle")
      .data(data)
      .enter()
      .append("circle")
      .attr("cx", d => x(d.day))
      .attr("cy", d => y(d.price))
      .attr("r", 6)
      .attr("fill", "#FF0000")  
      .call(d3.drag()
        .on("start", function() { tooltip.style("visibility", "visible"); })
        .on("drag", function(event, d) {
          // Limit the price to stay within MIN_PRICE and MAX_PRICE
          d.price = Math.max(MIN_PRICE, Math.min(MAX_PRICE, y.invert(event.y))); 
          d3.select(this).attr("cy", y(d.price));
          path.attr("d", line);
          updateVolatility();
          tooltip.text(`Day: ${d.day} | Price: $${d.price.toFixed(2)}`)
            .style("top", (event.pageY - 35) + "px")
            .style("left", (event.pageX + 10) + "px");
        })
        .on("end", function() {
          tooltip.style("visibility", "hidden");
        })
      );

    // Calculate and update volatility
    function updateVolatility() {
      let returns = [];
      for (let i = 1; i < data.length; i++) {
        let pctChange = (data[i].price - data[i - 1].price) / data[i - 1].price;
        returns.push(pctChange);
      }

      let mean = returns.reduce((sum, val) => sum + val, 0) / returns.length;
      let variance = returns.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / returns.length;
      let stdDev = Math.sqrt(variance);

      document.getElementById("volatility-display").innerText = `Volatility (σ): ${stdDev.toFixed(3)}`;

      // Remove the old volatility line and bands completely
      svg.selectAll(".volatility-band").remove();
      volatilityLine.attr("d", ""); // Clear the old volatility line

      // Create upper and lower volatility bands with clamping to keep within visible range
      let upperBand = data.map(d => ({
        day: d.day, 
        price: Math.min(100, d.price * (1 + stdDev)) // Clamp to max y value
      }));
      
      let lowerBand = data.map(d => ({
        day: d.day, 
        price: Math.max(0, d.price * (1 - stdDev)) // Clamp to min y value
      }));
      
      // Add upper volatility band
      svg.append("path")
        .datum(upperBand)
        .attr("class", "volatility-band")
        .attr("fill", "none")
        .attr("stroke", "#00FFFF")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "5,5")
        .attr("d", line);
        
      // Add lower volatility band
      svg.append("path")
        .datum(lowerBand)
        .attr("class", "volatility-band")
        .attr("fill", "none")
        .attr("stroke", "#00FFFF")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "5,5")
        .attr("d", line);
        
      // Shaded area between bands
      svg.append("path")
        .datum(upperBand.concat(lowerBand.slice().reverse()))
        .attr("class", "volatility-band")
        .attr("fill", "rgba(0, 255, 255, 0.1)")
        .attr("stroke", "none")
        .attr("d", d3.area()
          .x(d => x(d.day))
          .y0(d => y(d.price)) // For the lower band
          .y1(d => y(d.price)) // For the upper band
        );
    }

    // Reset the graph
    function resetGraph() {
      data = JSON.parse(JSON.stringify(originalData));
      path.datum(data).attr("d", line);
      circles.data(data).attr("cy", d => y(d.price));
      updateVolatility();
    }

    document.getElementById("reset-button").addEventListener("click", resetGraph);
    updateVolatility();

    // Update the legend to reflect the new visualization
    const legendXPos = width - 180;
    svg.append("rect").attr("x", legendXPos).attr("y", 10).attr("width", 12).attr("height", 12).style("fill", "red");
    svg.append("text").attr("x", legendXPos + 20).attr("y", 20).style("fill", "white").text("Stock Price").style("font-size", "14px");
    
    svg.append("rect").attr("x", legendXPos).attr("y", 30).attr("width", 12).attr("height", 12).style("fill", "rgba(0, 255, 255, 0.3)");
    svg.append("text").attr("x", legendXPos + 20).attr("y", 40).style("fill", "white").text("Volatility Band (±σ)").style("font-size", "14px");

    </script>
    """
    components.html(html_code, height=700, width=2000)
    volatility_text_block2()

def strike_price_section():
    # TODO: Make the line draggable instead of just using a slider.
    # strike_price_text_block()
    # text_block_2()
    num_days = 60
    initial_price = 100
    volatility = 0.15  
    drift = 0.01  
    dt = 1  
    np.random.seed(42)

    # Store stock prices in session state so they don't change when the slider is adjusted
    if "stock_price_data" not in st.session_state:
        random_shocks = np.random.normal(drift * dt, volatility * np.sqrt(dt), num_days)
        prices = initial_price * np.exp(np.cumsum(random_shocks))

        # Keep prices within 0 to 200
        prices = np.clip(prices, 0, 200)
        st.session_state.stock_price_data = prices

    # Single strike price slider at the very top
    strike_price = st.slider(
        "Strike Price",
        min_value=0,
        max_value=200,
        value=100,
        step=1,
        key="strike_price_slider",
        on_change=None  ,
        help="Drag this slider to change the strike price of the option.",
        label_visibility="visible"
    )

    # Get stored stock prices
    prices = st.session_state.stock_price_data

    df = pd.DataFrame({
        "Day": np.arange(1, num_days + 1),
        "Stock Price": prices
    })

    # Set baseline (strike price)
    df["Base"] = strike_price

    # Upsample data so there is no overlap between red and green sections 
    upsample_factor = 100  # Number of points per day
    new_days = np.linspace(1, num_days, num_days * upsample_factor)  
    new_prices = np.interp(new_days, df["Day"], df["Stock Price"])  

    df_upsampled = pd.DataFrame({
        "Day": new_days,
        "Stock Price": new_prices
    })

    # Reassign baseline after upsampling
    df_upsampled["Base"] = strike_price

    # Classifies section as ITM or OTM
    df_upsampled["ITM"] = df_upsampled["Stock Price"] >= df_upsampled["Base"]
    df_upsampled["ITM"] = df_upsampled["ITM"].replace({True: "ITM", False: "OTM"})  

    # Create main columns for layout
    chart_col, info_col = st.columns([0.7, 0.3])  # 70% chart, 30% info

    with chart_col:
        # Colors the chart and adds tooltip 
        area_chart = (
            alt.Chart(df_upsampled)
            .mark_area()
            .encode(
                x=alt.X("Day:Q", title="Day", scale=alt.Scale(domain=[1, num_days])),
                y=alt.Y("Stock Price:Q", title="Stock/Strike Price ($)"),
                y2="Base:Q",
                color=alt.Color(
                    "ITM:N",
                    scale=alt.Scale(domain=["ITM", "OTM", "Strike Price"], range=["green", "red", "#FFFFFF"]),
                    legend=alt.Legend(title="Option Status"),
                ),
                tooltip=[
                    alt.Tooltip("Day:Q", title="Day", format=".0f"),  
                    alt.Tooltip("Stock Price:Q", title="Stock Price", format=".0f"),  
                    alt.Tooltip("ITM:N", title="Status")
                ]
            )
        )

        # Strike price line
        strike_line = alt.Chart(pd.DataFrame({"Strike Price": [strike_price]})).mark_rule(
            color="#FFFFFF", strokeWidth=2
        ).encode(
            y="Strike Price:Q",
            tooltip=[alt.Tooltip("Strike Price:Q", title="Strike Price", format=".0f")]  
        )

        # Combine charts and display
        final_chart = (area_chart + strike_line).properties(
            width=800, height=400, title="ITM vs. OTM for Call Option"
        )
        st.altair_chart(final_chart, use_container_width=True)

    with info_col:
        # Hide the default delta arrows with CSS
        st.write(
            """
            <style>
            [data-testid="stMetricDelta"] svg {
                display: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # Calculate option values at expiration
        final_stock_price = prices[-1]
        call_value = max(0, final_stock_price - strike_price)
        put_value = max(0, strike_price - final_stock_price)
        
        st.write("#### Option Values at Expiration")
        
        # More compact call option info
        st.metric(
            label="Call Option",
            value=f"${call_value:.2f}",
            delta="↓ OTM" if call_value == 0 else f"↑ ${final_stock_price - strike_price:.2f} from strike",
            delta_color="inverse" if call_value == 0 else "normal"
        )
        st.markdown("""
        <span style='color: rgba(128, 128, 128, 0.6); font-size: 0.8em;'>
        Profit if price > strike<br>
        • Max loss = premium (cost to buy option)<br>
        • Max profit = unlimited
        </span>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 5px 0px'>", unsafe_allow_html=True)
        
        # More compact put option info
        st.metric(
            label="Put Option",
            value=f"${put_value:.2f}",
            delta="↓ OTM" if put_value == 0 else f"↑ ${strike_price - final_stock_price:.2f} from strike",
            delta_color="inverse" if put_value == 0 else "normal"
        )
        st.markdown("""
        <span style='color: rgba(128, 128, 128, 0.6); font-size: 0.8em;'>
        Profit if price < strike<br>
        • Max loss = premium (cost to buy option)<br>
        • Max profit = strike
        </span>
        """, unsafe_allow_html=True)

    # Text block at the bottom
    strike_price_text_block2()

def conclusion_section(): 
    st.title("Conclusion")
    st.write("Now that you have a basic understanding of the important variables that affect a Monte Carlo Simulation, try playing around with the full simulation from before.")
    st.write("Try adjusting different variables and see how they change the Monte Carlo estimate for the option price and the Black-Scholes estimate for the option price.")
    full_plot_test.display_monte_carlo_graph()

def drift_section():
    st.write("# Drift (μ):")
    st.write("Drift represents the expected long-term trend of the price of a stock over time. It ignores short-term randomness (volatility) and reflects the average rate of return a stock is expected to achieve.")
    st.write("- If drift is positive, a stock is expected to have an increase in value over time.")
    st.write("- If drift is neutral, a stock is expected to have no change in value over time.")
    st.write("- If drift is negative, a stock is expected to have a decrease in value over time.")
    st.write("Now that we've shown how Monte Carlo simulations can be used to estimate the value of an option, let's look at how drift affects its price.")
    st.write("Drag the slider below to change the drift of the stock price in the simulation.")

    drift = st.slider("Drift (μ)", -1.0, 1.0, 0.05, step=0.01)  

    # Fixed parameters
    initial_price = 100  # Starting asset price S0
    T = 3.0  # Extended time to expiration (years) for more drift effect
    num_steps = 100  # Time steps
    dt = T / num_steps
    volatility = 0.05  # Lower volatility so drift stands out more

    # Generate a single stock price path using geometric Brownian motion
    prices = np.zeros(num_steps + 1)
    prices[0] = initial_price  # Start at initial price

    for t in range(1, num_steps + 1):
        z = np.random.standard_normal()
        prices[t] = prices[t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)  # 2× Drift to amplify effect

    # Create DataFrame for Altair
    df = pd.DataFrame({
        "Day": np.linspace(0, T, num_steps+1),
        "Stock Price": prices
    })

    # Create a single line chart (no multiple paths)
    chart = alt.Chart(df).mark_line(color="red", size=2).encode(
        x=alt.X("Day:Q", title="Time (years)"),
        y=alt.Y("Stock Price:Q", title="Stock Price ($)")
    ).properties(
        width=700,
        height=400,
        title="Monte Carlo Simulation: Stronger Drift Effect"
    )

    st.altair_chart(chart, use_container_width=True)

    st.write("The main purpose of drift in Monte Carlo simulations is to help simulate the overall trend of a stock's value over time.")

def num_paths_section():
    st.title("Number of Path Simulations")
    st.write("The number of path simulations for a Monte Carlo Simulation is the number of possible different outcomes trees we are exploring. For example, if we have one path simulation, then we are only mapping one possible outcome. If we have two path simulations, then we are looking at two possible futures, etc etc. The more simulation paths that we observe, the more data we will have to analyze, and the more complete of a picture we will have prediting the future. A simple way we can see this is with the law of large numbers, which states that the more data we have, that dataset's mean will be closer to the true mean of whatever we are sampling. So, in our example, the more paths we have, the better an estimate of the stock's future value we will have.")
    num_paths.display_monte_carlo_graph()

def num_iterations_section():
    st.title("Number of Iterations")
    st.write("The number of iterations for a Monte Carlo Simulation is how many times the simulation takes a data point. We find this number through looking at the time period of the option and the number of steps for the simulation. Essentially, how often do we want to take a measuerment and over what period of time will we be measuring?")
    st.write("A higher number of time steps will give you a more accurate understanding of what happening, as you will have more data points, taken more often. More data points will give you better and more accurate analysis of the option, however computing more data points obviously takes more computing power, and with complicated models and large numbers of data points, simulating can take several hours. A longer time period is also a safer bet that an option will increase in value, but this will also lead to more data points being taken (and thus more computing power used). Additionally a longer time to expiration will generally lead to a more expensive option, given that the liklihood of increased value is higher. However, with a longer time to expiration and more iterations of the simulation, we are more likely to find a value that we like (one that makes us money).")
    num_iterations.display_monte_carlo_graph()

selection_renders = {
    "Introduction": intro_section,
    "Volatility": volatility_section,
    "Strike Price": strike_price_section,
    "Options Basics": basics_section,
    "Monte Carlo Simulations": Monte_Carlo_section,
    "Drift": drift_section,
    "Paths": num_paths_section,
    "Iterations": num_iterations_section,
    "Conclusion": conclusion_section,
    "Burn In": burn_in_section,
}

# Render the selected section
if selected_section in selection_renders:
    selection_renders[selected_section]()
else:
    st.error("Invalid section selected")
