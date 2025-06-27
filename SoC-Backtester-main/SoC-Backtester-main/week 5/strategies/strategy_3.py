import pandas as pd
import numpy as np

DATA_PATH = "C:/Users/arnav/OneDrive/Desktop/backtesting soc/week2/SoC-Backtester-main/SoC-Backtester-main/week 5/data"


def your_strat_3(stock, max_weight=0.3):
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
        volatility = returns.rolling(window=20).std()
        signal = -volatility  # mean reversion on volatility spike
        portfolio[i, :] = signal.shift(1).fillna(0).to_numpy()

    abs_sum = np.sum(np.abs(portfolio), axis=0, keepdims=True)
    abs_sum[abs_sum == 0] = 1.0
    normalized_portfolio = portfolio / abs_sum
    clipped_portfolio = np.clip(normalized_portfolio, -max_weight, max_weight)
    market_neutral_portfolio = clipped_portfolio - clipped_portfolio.mean(axis=0, keepdims=True)

    return market_neutral_portfolio