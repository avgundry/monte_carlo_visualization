import streamlit as st

def text_block_1():
    # Monte-Carlo Options Pricing.
    st.write("# Monte-Carlo Options Pricing: A Visualization")
    st.write("This article aims to help you understand the basics of options, a type of financial asset, and how you can use a type of simulation called a Monte Carlo simulation to evaluate stock options. After learning about the basics of options, you will be able to simulate option pricing based on adjustable parameters like volatility, drift, interest rate, time to expiration, and number of simulation paths. As paths evolve visually, a convergence plot will dynamically show the option price estimate stabilizing. This will be plotted against the Black-Scholes model, a formulaic method for pricing options, highlighting the convergence of Monte Carlo simulations to the underlying price. Note, however, that in many cases we could not use the Black-Scholes model at all, which is why the Monte Carlo simulations are useful - in this case it is only shown for comparison.")
    st.write("Pricing options is a complex topic that requires estimating the expected future value of an asset, a task that becomes analytically intractable when the option has complex features or many sources of uncertainty. Monte Carlo simulations are a powerful technique to approximate option prices by simulating many potential future price paths, but their stochastic nature can make them difficult to grasp.")
    st.write("At the end of this article, we hope you will be able to understand all the components of the following visualization. Don't worry if it looks intimidating right now! We will go step by step through each variable to help you grasp how they affect the option price.")


def text_block_2():
    # Basics of Options Pricing.
    st.write("# Basics of Options Pricing")
    st.write("We'll begin with some explanations. Options are a financial contract that give the buyer the *right* (but not the obligation! hence the name \"option\") to buy or sell an asset at a specific price, the **strike price**, before the option's **expiration date**.")
    st.write("There are two main kinds of options: **call options**, which let you buy an asset at the strike price before the expiration date, and **put options**, which let you sell an asset at the strike price before the expiration date.")
    st.write("To put it all together, the strike price is the predetermined price at which an option holder can buy (call option) or sell (put option) the underlying asset before the expiration date. The strike price determines if an option is In The Money (ITM) or Out of The Money (OTM).")
    st.write("- ITM is when the option is profitable to exercise. For call options, this is when the stock price is above the strike price. For put options, this is when the stock value is below the strike price.")
    st.write("- OTM is when exercising the option would result in a loss. The option therefore has no value and is worthless.")
    st.write("Pricing options - aka determining their **premium**, the price they sell for - is important for both buyers and sellers of options. For buyers, it allows them to figure out if an option is a good deal and thus whether they can make a profit on it. For sellers, it allows them to figure out how much to charge for an option, i.e. the 'fair price'.")
    st.write("Below is a visualization for the strike price of a call option where the underlying stock's value fluctuates over 60 days. You can hover over the graph to see more exact dates, stock price, and whether the option is ITM vs. OTM")
    st.write("Try changing the strike price by dragging the slider to see what happens to the graph. You can see how much potential profit you could gain after 60 days based on what the strike price was set at.")
    # st.write(r"For example, let's say you want to buy a call option with a strike price of \$110 that expires in 1 month on Pepsi stock, which is currently worth \$100. How would you price this option? If you pay \$10 for the option, you are hoping that Pepsi stock will be worth more than \$110 in 1 month, so that you can buy it at \$110 and sell it for more than you paid for the option.")
    # st.write(r"Say we buy this option for \$10. If Pepsi stock goes up to \$135 in 1 month, you can use the option to buy it at \$110 and then sell it for \$135, netting you a profit of \$135 (what you sell it for) - \$110 (the strike price you buy at) - \$10 (the price of the option) = \$15. You can see that the option is now worth \$15, so it is a good deal!")
    # st.write(r"However, assume that Pepsi stock goes down to \$90 in 1 month. If you bought the option, you would not use it, because you could just buy Pepsi stock for \$90 - there's no reason to buy it at the option's strike price of \$110. In this case, the option is worthless, and you would have lost the \$10 you paid for it.")
    # st.write("Therefore, when buying an option, we seek to only buy it if we think that *on average* the option will return a profit above the price of buying it.")


def text_block_3():
    # Monte-Carlo Simulation.
    st.write("# Monte Carlo Simulation with Metropolis-Hastings")
    st.write(r"In our strike price simulation, it's easy to see exactly what the option is worth, and when we should buy or sell it. However, in the real world, we often can't determine the exact value of the option. Instead, we can use a Monte Carlo simulation to estimate the value of the option.")
    st.write("Let's start with a simple example to demonstrate how a Monte Carlo simulation works. To make things as simple as possible, we'll only concern ourselves with two outcomes: whether the option will be in the money or out of the money at the expiration date. This is easy to simulate as a coin toss.")
    st.write("Say we have a coin that has an unknown probability $p$ of landing heads (in this analogy, coming up as ITM). An intuitive way to estimate $p$ is to flip the coin many times and take the average of the number of heads.")
    st.write("At its core, this is what a Monte Carlo simulation does: it repeatedly draws random samples from a probability distribution and then takes the average of the function evaluated at those samples.")
    st.write("Let's demonstrate this now using code. Try running the simulation below to randomly generate a probability $p$ of landing heads, between 0 and 1 - in the real world, we wouldn't know what $p$ is! The visualization will show how repeatedly flipping the coin converges to the true probability $p$.")

def volatility_text_block():
    st.write(r"# Volatility($\sigma$):")
    st.write("Volatility measures how much the price of a stock fluctuates over time. It can be conceptualized as the amount we expect a stock's price to fluctuate around the mean at any given time. When analyzing options, there are two key types of volatility: implied volatility (IV) and historical volatility (HV).")
    st.write("- IV reflects the market's expectation of future price fluctuations. It measures how likely a stock is to experience major price swings (in either direction) before the option expires. ")
    st.write("- HV measures how much a stock's price has actually fluctuated in the past over a given period (e.g., 30 days, 60 days, or 1 year). It is calculated using the standard deviation of past returns.")
    st.write("Below is a visualization that calculates the HV of a stock based how it's price has fluctuated over the past 30 days. This shows the volatility as a bar around the mean price, indicating the average historical fluctuation around that price.")
    st.write("Try dragging each point on the graph or pressing the randomize button to see how the calculated volatility changes. Large sudden fluctuations in price will result in higher volatility.")

def volatility_text_block2():
    st.write("IV is important because it is a key factor in determining an option's price. Higher IV indicates a greater chance the stock will reach or exceed the strike price, increasing the potential profit for buyers. As a result, options on stocks with high IV tend to be more expensive than those on stable stocks. HV is important because it serves as a benchmark for IV, helping traders assess if an options IV over or under priced. In Monte Calro simulations, IV determines the likelihood of the option expiring In The Money (ITM) or not. Because options do not require you to exercise your right to buy or sell at any time, if IV is high, the stock has a greater chance of hitting the strike price before the option expires. In other words, more large fluctuations means you have a higher chance of the stock reaching a value where you can return a profit before the option expires. ")


def strike_price_text_block():
    st.write("# Strike price (K): ")
    st.write("The strike price is the predetermined price at which an option holder can buy(call option) or sell (put option) the underlying asset before the expiration date. The strike price determines if an option is In The Money (ITM) or Out of The Money (OTM)")
    st.write("- ITM is when the option is profitable to exercise. For call options, this is when the stock price is above the strike price. For put options, this is when the stock value is below the strike price.")
    st.write("- OTM is when exercising the option would result in a loss. The option therefore has no value and is worthless.")
    st.write("Below is a visualization for the strike price of a call option where the underlying stock's value fluctuates over 60 days. You can hover over the graph to see more exact dates, stock price, and whether the option is ITM vs. OTM")
    st.write("Try changing the strike price to see what happens to the graph")

def strike_price_text_block2():
    st.write("Strike price is important because it determines when you can make money by exercising an option. If you are ITM for a call option, you can buy the stock for a value less than what it is currently trading at and instantly sell it for a profit. For put options, you can buy the stock for the current trading price and instantly sell them at the higher option price for a profit.")
