import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as py
import os

def backtest_portfolio(portfolio, tickers, data_path= "C:/Users/arnav/OneDrive/Desktop/backtesting soc/week2/SoC-Backtester-main/SoC-Backtester-main/week 5/data", risk_free_rate=0.0451):
    print("✅ Running updated backtest_portfolio")

    n_stocks, n_days = portfolio.shape
    price_matrix = np.zeros((n_stocks, n_days))

    # Load price data
    for i, ticker in enumerate(tickers):
        file_path = os.path.join(data_path, f"{ticker}.csv")
        print(f"Trying to read: {file_path}")
        df = pd.read_csv(file_path)

        df = df.reset_index(drop=True)
        df['Close'] = pd.to_numeric(df['Close'])
        price_matrix[i, :] = df['Close'].to_numpy()

    # # Stock daily returns
    stock_returns = pd.DataFrame(price_matrix).pct_change(axis=1).fillna(0).values

    # print(stock_returns)

    # Portfolio returns
    daily_returns = np.nansum(portfolio * stock_returns, axis=0)

    # Cumulative Returns
    cumulative_returns = np.cumprod(1 + daily_returns) - 1
    
    # Sharpe Ratio
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = np.mean(excess_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

    # Max Drawdown
    cumulative_curve = np.cumprod(1 + daily_returns)
    rolling_max = np.maximum.accumulate(cumulative_curve)
    drawdown = 1 - cumulative_curve / rolling_max
    max_drawdown = np.max(drawdown)

    # Turnover
    turnover = np.sum(np.abs(np.diff(portfolio, axis=1))) / n_days

    # Mean Correlation with individual stocks
    correlations = []
    for i in range(n_stocks):
        x = daily_returns
        y = stock_returns[i, :]
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) > 1:
            corr = np.corrcoef(x[mask], y[mask])[0, 1]
            correlations.append(corr)
        else:
            correlations.append(np.nan)
    mean_correlation = np.nanmean(correlations)

    results = {
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Turnover": turnover,
        "Total Return": cumulative_returns[-1] * 100,
        "Mean Correlation (Portfolio vs Stocks)": mean_correlation,
        "Daily Returns": daily_returns,
        "Cumulative Returns": cumulative_returns
    }

    # PLotting 
    days = np.arange(n_days)

    # Create subplots layout: 3 rows for returns, drawdown, top stocks
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("Cumulative Returns", "Daily Returns", "Drawdown Curve", "Portfolio vs Top 5 Stocks"),
        vertical_spacing=0.1,
        shared_xaxes=True
    )


    # Daily Returns
    fig.add_trace(go.Scatter(x=days, y=daily_returns, name="Daily Return"), row=2, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(x=days, y=drawdown, name="Drawdown"), row=3, col=1)

    top_5_idx = np.argsort(correlations)[-5:]
    # Portfolio vs Top 5 stocks plot
    fig.add_trace(go.Scatter(x=days, y=np.cumprod(1 + daily_returns) - 1, name="Portfolio"), row=4, col=1)
    for i in top_5_idx:
        fig.add_trace(go.Scatter(x=days, y=np.cumprod(1 + stock_returns[i]) - 1, name=f"{tickers[i]}"), row=4, col=1)


    # for i in top_5_idx:
    #     print(stock_returns[i])
    # print(top_5_idx)

    for i in top_5_idx:
        fig.add_trace(go.Scatter(x=days, y=stock_returns[i], name=f"{tickers[i]}"), row=4, col=1)

    # Layout
    fig.update_layout(
        height=1200,
        title_text="Portfolio Performance Summary",
        showlegend=True
    )

    # Save and open in browser
    py.plot(fig, filename="portfolio_summary.html")

    return results, correlations
