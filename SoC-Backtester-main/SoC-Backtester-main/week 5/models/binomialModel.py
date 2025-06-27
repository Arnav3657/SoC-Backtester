import numpy as np

def binomial_american_option(S, K, T, r, sigma, N, option_type="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Asset prices at maturity
    asset_prices = np.array([S * u**(N-i) * d**i for i in range(N+1)])

    # Option values at maturity
    if option_type == "call":
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    # Backward induction
    for j in range(N-1, -1, -1):
        asset_prices = asset_prices[:-1] / u  # Move to previous step
        option_values = disc * (p * option_values[:-1] + (1-p) * option_values[1:])
        if option_type == "call":
            option_values = np.maximum(option_values, asset_prices - K)
        else:
            option_values = np.maximum(option_values, K - asset_prices)

    return option_values[0]


S = 100      # stock price
K = 100      # strike price
T = 1        # time to expiry (years)
r = 0.05     # risk-free rate (annual)
sigma = 0.2  # volatility (annual)
N = 100      # number of steps
option_type = "call"  # call

price = binomial_american_option(S, K, T, r, sigma, N, option_type)
print(f"American {option_type} option price: {price:.4f}")