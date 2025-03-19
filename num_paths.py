import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm

def display_monte_carlo_graph():
    # Hide fullscreen buttons
    st.markdown("""
        <style>
        [aria-label="Fullscreen"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Controls above the graph ---
    st.write("### Simulation Settings")
    st.info("""
            Instructions:
            \n Either use the +/- buttons or type in a number below to change the number of simulated paths.
            """)

    col1, col2, col3 = st.columns(3)

    with col1:
        num_paths = st.number_input("Number of Simulation Paths", 1, 1000, 50, step=1)
        volatility = 0.15
        drift = 0.01

    with col2:
        initial_price = 100  # Adjustable initial price
        interest_rate = 0.2
        T = 1

    with col3:
        num_steps = 250
        strike = 100

    st.write("---")  # Separator before the graph

    dt = T / num_steps

    # Initialize simulation array
    paths = np.zeros((num_steps + 1, int(num_paths)))
    paths[0] = initial_price  # 👈 Uses user-input S₀

    # Generate paths using geometric Brownian motion
    for t in range(1, num_steps + 1):
        z = np.random.standard_normal(int(num_paths))
        paths[t] = paths[t-1] * np.exp((interest_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)  # Changed drift to interest_rate

    # Monte Carlo option pricing
    call_payoffs = np.maximum(paths[-1] - strike, 0)
    mc_option_price = np.exp(-interest_rate * T) * np.mean(call_payoffs)

    # Add standard error calculation
    mc_std_error = np.std(call_payoffs) / np.sqrt(num_paths)
    mc_confidence_interval = (
        mc_option_price - 1.96 * mc_std_error,
        mc_option_price + 1.96 * mc_std_error
    )

    # Black-Scholes option pricing
    d1 = (np.log(initial_price / strike) + (interest_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
    d2 = d1 - volatility * np.sqrt(T)
    bs_option_price = initial_price * norm.cdf(d1) - strike * np.exp(-interest_rate * T) * norm.cdf(d2)

    # Display the computed option prices
    st.write("### Option Prices")
    st.write(f"**Monte Carlo Estimated Option Price:** {mc_option_price:.2f}")
    # st.write(f"**95% Confidence Interval:** ({mc_confidence_interval[0]:.2f}, {mc_confidence_interval[1]:.2f})")
    # st.write(f"**Standard Error:** {mc_std_error:.2f}")
    st.write(f"**Black-Scholes Option Price:** {bs_option_price:.2f}")

    # --- Visualization section ---
    df = pd.DataFrame(paths)
    df['time'] = np.linspace(0, T, num_steps+1)
    df_melted = df.melt(id_vars=['time'], var_name='Simulation', value_name='Price')

    # Compute mean price path
    mean_path = np.mean(paths, axis=1)
    df_mean = pd.DataFrame({'time': np.linspace(0, T, num_steps+1), 'Price': mean_path})

    # Paths chart
    paths_chart = alt.Chart(df_melted).mark_line(opacity=0.3).encode(
        x=alt.X("time:Q", title="Time (years)"),
        y=alt.Y("Price:Q", title="Underlying Price"),
        color=alt.Color("Simulation:N", legend=None),
        tooltip=alt.value(None) # disable this tooltip to allow hover chart to take over
    )

    # Invisible thick lines for better hover detection
    hover_chart = alt.Chart(df_melted).mark_line(
        strokeWidth=10,  # Much thicker line
        opacity=0        # Completely transparent
    ).encode(
        x="time:Q",
        y="Price:Q",
        color=alt.Color("Simulation:N", legend=None),
        tooltip=["time:Q", "Price:Q", "Simulation:N"]  # Added more detailed tooltip
    )

    # Overlay mean path in bold red
    mean_chart = alt.Chart(df_mean).mark_line(color="red", size=3).encode(
        x="time:Q",
        y="Price:Q",
    )

    # Combine charts
    chart = (paths_chart + hover_chart + mean_chart).properties(
        width=700,
        height=400,
        title="Monte Carlo Simulation: Colored Paths with Mean Price",
    ).configure_view(
        strokeOpacity=0
    )

    st.altair_chart(chart, use_container_width=True)
