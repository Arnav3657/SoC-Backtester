import pandas as pd
import numpy as np

from strategies.strategy_1 import your_strat_1
from strategies.strategy_2 import your_strat_2
from strategies.strategy_3 import your_strat_3  # If made
from backtest.backtest import backtest_portfolio
import json
import os

DATA_PATH = "C:/Users/arnav/OneDrive/Desktop/backtesting soc/week2/SoC-Backtester-main/SoC-Backtester-main/week 5/data"

print("Current working dir:", os.getcwd())
print("Files in data_path:", os.listdir("C:/Users/arnav/OneDrive/Desktop/backtesting soc/week2/SoC-Backtester-main/SoC-Backtester-main/week 5/data"))
# Choose your tickers (make sure CSVs exist in data/)
tickers = ['AAPL', 'MSFT', 'GOOG']

# Run strategy 1
portfolio1 = your_strat_1(tickers)
results1, correlations1 = backtest_portfolio(portfolio1, tickers, DATA_PATH)
with open("results_strategy1.json", "w") as f:
    json.dump(results1, f, indent=4)
print("Strategy 1 done ✔️")

# Run strategy 2
portfolio2 = your_strat_2(tickers)
results2, correlations2 = backtest_portfolio(portfolio2, tickers, DATA_PATH)
with open("results_strategy2.json", "w") as f:
    json.dump(results2, f, indent=4)
print("Strategy 2 done ✔️")

# Run strategy 3 (optional)
try:
    from strategies.strategy_3 import your_strat_3
    portfolio3 = your_strat_3(tickers)
    results3, correlations3 = backtest_portfolio(portfolio3, tickers, DATA_PATH)
    with open("results_strategy3.json", "w") as f:
        json.dump(results3, f, indent=4)
    print("Strategy 3 done ✔️")
except ImportError:
    print("Strategy 3 not implemented or failed to import.")
