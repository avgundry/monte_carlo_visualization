import numpy as np
import pandas as pd


def simulate_coin_flips(true_p, N):
    coin_flips = np.random.binomial(1, true_p, size=N)
    num_heads = np.sum(coin_flips)
    num_tails = N - num_heads
    df = pd.DataFrame({
        "Flip Number": np.arange(N),
        "Running Average": np.cumsum(coin_flips) / np.arange(1, N + 1)
                       })
    print(f"df: {df}")
    return df, num_heads, num_tails

# def generate_coin_data(p_true: float, N: int):
#     """
#     Generate coin flip data from a Bernoulli(p_true).
#     Returns heads_count, tails_count.
#     """
#     flips = np.random.binomial(1, p_true, size=N)
#     heads = np.sum(flips)
#     tails = N - heads
#     return heads, tails

def log_likelihood(p, heads, tails):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return heads * np.log(p) + tails * np.log(1 - p)

def propose_new_state(current_p, proposal_width=0.05):
    candidate = current_p + np.random.normal(0, proposal_width)
    return np.clip(candidate, 0, 1)

def mh_chain(n_steps, heads_count, tails_count, proposal_width=0.05, start_p=0.5):
    chain_records = []
    current_p = start_p
    current_ll = log_likelihood(current_p, heads_count, tails_count)
    
    for i in range(1, n_steps + 1):
        candidate_p = propose_new_state(current_p, proposal_width)
        cand_ll = log_likelihood(candidate_p, heads_count, tails_count)
        log_ratio = cand_ll - current_ll
        if np.log(np.random.rand()) < log_ratio:
            accepted = True
            current_p = candidate_p
            current_ll = cand_ll
        else:
            accepted = False
        
        chain_records.append({
            'iteration': i,
            'candidate': candidate_p,
            'accepted': accepted,
            'state': current_p
        })
    return pd.DataFrame(chain_records)

def acceptance_ratio(p_current, p_proposal, heads_count, tails_count):
    # We'll use the difference of log-likelihoods (log-ratio) to keep things stable.
    ll_current = log_likelihood(p_current, heads_count, tails_count)
    ll_proposal = log_likelihood(p_proposal, heads_count, tails_count)
    
    # ratio = exp(ll_proposal - ll_current)
    log_ratio = ll_proposal - ll_current
    return np.exp(log_ratio)


def metropolis_hastings_step(p_current, heads_count, tails_count, proposal_width=0.05):
    # Propose new state
    p_proposal = propose_new_state(p_current, proposal_width)
    
    # Compute acceptance ratio
    ratio = acceptance_ratio(p_current, p_proposal, heads_count, tails_count)
    
    # Draw a random uniform(0,1) to decide acceptance
    if np.random.rand() < ratio:
        # Accept
        return p_proposal
    else:
        # Reject, stay at current
        return p_current

# Define target distribution:
# Posterior alpha p^(num_heads) * (1 - p)^(num_tails), uniform prior
def target(p, num_heads, num_tails):
    if p <= 0 or p >= 1:
        return 0
    return (p ** num_heads) * ((1 - p) ** num_tails)

# Function to run Metropolis-Hastings algorithm
def metropolis_hastings(num_iterations, proposal_std, num_heads, num_tails, burn_in):
    chain = []
    current_p = 0.5  # Starting guess
    # Burn-in period.
    burn_in_iterations = np.int32(num_iterations * burn_in)
    print(f"Burn-in iterations: {burn_in_iterations}")
    for _ in range(burn_in_iterations):
        proposed_p = current_p + np.random.normal(0, proposal_std)
        proposed_p = np.clip(proposed_p, 0, 1)  # Ensure it's in [0,1]

        current_target = target(current_p, num_heads, num_tails)
        proposed_target = target(proposed_p, num_heads, num_tails)

        if current_target == 0:
            acceptance_prob = 1 if proposed_target > 0 else 0
        else:
            acceptance_prob = min(1, proposed_target / current_target)

        if np.random.rand() < acceptance_prob:
            current_p = proposed_p

        chain.append(current_p)

    # Sampling period.
    for _ in range(num_iterations - burn_in_iterations):
        proposed_p = current_p + np.random.normal(0, proposal_std)
        proposed_p = np.clip(proposed_p, 0, 1)  # Ensure it's in [0,1]

        current_target = target(current_p, num_heads, num_tails)
        proposed_target = target(proposed_p, num_heads, num_tails)

        if current_target == 0:
            acceptance_prob = 1 if proposed_target > 0 else 0
        else:
            acceptance_prob = min(1, proposed_target / current_target)
            

    return np.array(chain)

