import pandas as pd
import numpy as np
DATA_PATH = "C:/Users/arnav/OneDrive/Desktop/backtesting soc/week2/SoC-Backtester-main/SoC-Backtester-main/week 5/data"



def your_strat_1(stock, max_weight=0.3, momentum_days=50, reversal_days=20):
    """
    Simpler strategy combining short-term reversal with medium-term momentum
    """
    dfs = []
    n = None

    for ticker in stock:
        print(f"Trying to read: C:/Users/arnav/OneDrive/Desktop/backtesting soc/week2/SoC-Backtester-main/SoC-Backtester-main/week 5/data/{ticker}.csv")
        df = pd.read_csv(f"{DATA_PATH}/{ticker}.csv")
        df.reset_index(drop=True, inplace=True)
        df['Close'] = pd.to_numeric(df['Close'])
        dfs.append(df)
        if n is None:
            n = len(df)

    portfolio = np.zeros((len(stock), n))

    for i, df in enumerate(dfs):
        returns = df['Close'].pct_change()
        
        # Medium-term momentum
        momentum = returns.rolling(window=momentum_days).mean()
        
        # Short-term reversal
        short_term_return = returns.rolling(window=reversal_days).mean()
        reversal = -short_term_return  # Opposite of recent performance
        
        # Combine: follow medium-term trend but fade short-term moves, now its more short term focused
        signal = 0.7 * momentum + 0.3 * reversal # Try changing parameters. Might get some better results 
        
        # Volatility adjustment
        vol = returns.rolling(window=20).std()
        vol_adjusted_signal = signal / (vol + 1e-8) # Avoid division by zero. Important.
        
        portfolio[i, :] = vol_adjusted_signal.shift(1).fillna(0).to_numpy()

    abs_sum = np.sum(np.abs(portfolio), axis=0, keepdims=True)
    abs_sum[abs_sum == 0] = 1.0 # Prevents division by 0
    normalized_portfolio = portfolio / abs_sum # Normalising the portfolio to have a sum of 1
    clipped_portfolio = np.clip(normalized_portfolio, -max_weight, max_weight) # Clip to prevent over-weighting
    market_neutral_portfolio = clipped_portfolio - clipped_portfolio.mean(axis=0, keepdims=True) # Subtract the mean to make it market neutral
    
    return market_neutral_portfolio
    
    return -1