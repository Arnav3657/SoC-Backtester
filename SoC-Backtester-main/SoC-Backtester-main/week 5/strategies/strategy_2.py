import pandas as pd
import numpy as np

DATA_PATH = "C:/Users/arnav/OneDrive/Desktop/backtesting soc/week2/SoC-Backtester-main/SoC-Backtester-main/week 5/data"

def your_strat_2(stocks, max_weight=0.3, momentum_days=120, reversal_days=20):
    dfs = []
    n = None
    for ticker in stocks:
        print(f"Trying to read: C:/Users/arnav/OneDrive/Desktop/backtesting soc/week2/SoC-Backtester-main/SoC-Backtester-main/week 5/data/{ticker}.csv")
        df = pd.read_csv(f"{DATA_PATH}/{ticker}.csv")
        df.reset_index(drop=True, inplace=True)
        df['Close'] = pd.to_numeric(df['Close'])
        dfs.append(df)
        if n is None:
            n = len(df)
    portfolio = np.zeros((len(stocks), n))
    
    for i, df in enumerate(dfs):
        df['Close'] = df['Close'].fillna(method='ffill')
        returns = df['Close'].pct_change()

    
    # Calculate SMA over a rolling window (momentum_days + reversal_days)
    window = momentum_days + reversal_days
    sma = returns.rolling(window=window).mean()

    momentum = 0
    reversal = 0

    # Iterate over valid indices where SMA is available
    for idx in range(window - 1, len(returns)):
        current_return = returns.iloc[idx]
        current_sma = sma.iloc[idx]

        if pd.isna(current_sma) or pd.isna(current_return):
            continue

        if current_return > current_sma:
            # Medium-term momentum
            momentum += current_return
        else:
            # Short-term reversal
            reversal += -current_return

    # Combine signals: follow medium-term trend but fade short-term moves
    signal = 0.3 * momentum + 0.7 * reversal # Try changing parameters. Might get some better results
    # Volatility adjustment
    vol = returns.rolling(window=10).std()
    vol_adjusted_signal = signal / (vol + 1e-8)
    portfolio[i, :] = vol_adjusted_signal.shift(1).fillna(0).to_numpy()
    abs_sum = np.sum(np.abs(portfolio), axis=0, keepdims=True)
    abs_sum[abs_sum == 0] = 1.0  # Prevents division by 0
    normalized_portfolio = portfolio / abs_sum  # Normalising the portfolio to have a sum of 1
    clipped_portfolio = np.clip(normalized_portfolio, -max_weight, max_weight)  # Clip to prevent over-weighting
    market_neutral_portfolio = clipped_portfolio - clipped_portfolio.mean(axis=0, keepdims=True)  # Subtract the mean to make it market neutral
    return market_neutral_portfolio 
    return -1