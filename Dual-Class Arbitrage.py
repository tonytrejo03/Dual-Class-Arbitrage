# Anthony Trejo
# 11/08/2025
#
# Dual-Class Arbitrage
#
# A trading strategy that involves taking advantage of the price discrepancy
# that exists between stocks that have dual listings. (Ex: GOOG vs GOOGL)

import warnings
warnings.filterwarnings('ignore')


import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['figure.figsize'] = (12,6)

# 1) PARAMETERS
ticker_x = "GOOG"   # stock 1 (price series we'll call X)
ticker_y = "GOOGL"  # stock 2 (price series we'll call Y)
start = "2018-01-01"
end = datetime.today().strftime("%Y-%m-%d")

entry_z = 2.0       # entry z-score
exit_z = 0.5        # exit z-score
lookback = 60       # for z-score rolling mean/std
transaction_cost = 0.001  # 10 bps per trade (per side)
slippage = 0.0005          # 5 bps slippage estimate
capital = 1000000     # starting capital

# 2) DOWNLOAD ADJUSTED CLOSE
data = yf.download([ticker_x, ticker_y], start=start, end=end, progress=False)['Close'].dropna()
data = data.dropna()  # ensure alignment
data.head()


# 3) CHECK VISUAL
data.plot(title=f"{ticker_x} vs {ticker_y} Adjusted Close")

# 4) COINTEGRATION TEST (Engle-Granger)
score, pvalue, _ = coint(data[ticker_x], data[ticker_y])
print(f"Cointegration test p-value: {pvalue:.4f} (lower is evidence of coint)")

# 5) ESTIMATE HEDGE RATIO (OLS)
# Regress X on Y: X = beta * Y + intercept + eps
Y = sm.add_constant(data[ticker_y])
model = sm.OLS(data[ticker_x], Y).fit()
beta = model.params[1]
intercept = model.params[0]
print(f"Hedge ratio (beta): {beta:.6f}, intercept: {intercept:.6f}")
# Spread
spread = data[ticker_x] - (beta * data[ticker_y] + intercept)

# 6) BUILD Z-SCORE
rolling_mean = spread.rolling(window=lookback).mean()
rolling_std  = spread.rolling(window=lookback).std()
zscore = (spread - rolling_mean) / rolling_std

# plot spread and zscore
fig, ax = plt.subplots(2,1, figsize=(12,8), sharex=True)
ax[0].plot(spread.index, spread, label='spread')
ax[0].plot(rolling_mean.index, rolling_mean, label='rolling_mean')
ax[0].legend()
ax[0].set_title('Spread')
ax[1].plot(zscore.index, zscore, label='zscore')
ax[1].axhline(entry_z, color='r', linestyle='--')
ax[1].axhline(-entry_z, color='r', linestyle='--')
ax[1].axhline(exit_z, color='g', linestyle='--')
ax[1].axhline(-exit_z, color='g', linestyle='--')
ax[1].set_title('Z-score')
plt.show()

# 7) SIGNALS -> POSITION SIZING
# We'll do a simple dollar-neutral scheme: long one leg and short the other so dollar exposures balance.
# Define position: +1 means long X short Y with size determined by hedge ratio so position is market-neutral.

# Signals:
# If zscore > entry_z -> SHORT spread -> (Short X, Long beta*Y)  -> position = -1
# If zscore < -entry_z -> LONG spread -> (Long X, Short beta*Y) -> position = +1
# Exit when abs(zscore) < exit_z -> position -> 0

signals = pd.Series(index=spread.index, dtype=float)
position = 0
for t in range(len(zscore)):
    if np.isnan(zscore.iloc[t]): 
        signals.iloc[t] = 0.0
        continue
    z = zscore.iloc[t]
    if position == 0:
        if z > entry_z:
            position = -1
        elif z < -entry_z:
            position = 1
    elif position == 1:
        if z >= -exit_z:
            position = 0
    elif position == -1:
        if z <= exit_z:
            position = 0
    signals.iloc[t] = position

# positions in shares: choose dollar exposure equal for legs
# For each day, dollar_exposure = capital * portfolio_leverage (we'll do 1:1 -> not leveraged)
dollar_exposure = capital  # you could scale by fraction; for simpler P&L compute returns per $1 capital
# Convert to shares: we allocate half the exposure to each leg to be dollar neutral
px_x = data[ticker_x]
px_y = data[ticker_y]

# Number of shares long/short at time t (per $1 of dollar exposure)
# We'll compute notional per side = 0.5 * dollar_exposure
notional_per_side = 0.5 * dollar_exposure
# shares_x = notional_per_side / px_x  (but hedge ratio adjusts Y side)
# To be dollar neutral while honoring hedge ratio, it's simpler to set:
# long X notional = short Y notional = notional_per_side
# but since hedge ratio affects units, we need to convert notional to shares
shares_x = notional_per_side / px_x
shares_y = notional_per_side / px_y

# Build actual positions series for each ticker
pos_x = signals * shares_x  # sign indicates long/short X
pos_y = -signals * shares_y # opposite sign for Y (dollar neutral)
# Note: this ignores hedge ratio in units; alternative is to use beta to scale shares_y = beta * shares_x
# Here we use strict dollar neutrality. Both are acceptable approaches; you'll want to test both.

# 8) P&L computation (mark-to-market daily)
# daily returns for tickers
ret_x = px_x.pct_change().fillna(0)
ret_y = px_y.pct_change().fillna(0)

# daily pnl = position(t-1) * return(t)
pnl_x = pos_x.shift(1) * (px_x.pct_change().fillna(0)) * px_x  # pos * change in price
pnl_y = pos_y.shift(1) * (px_y.pct_change().fillna(0)) * px_y
total_pnl = pnl_x + pnl_y

# incorporate transaction costs when a trade occurs (change in position)
trades_x = (pos_x.diff().abs()).fillna(0)
trades_y = (pos_y.diff().abs()).fillna(0)
# cost per trade = transaction_cost * traded_notional
trade_costs = (trades_x * px_x + trades_y * px_y) * transaction_cost
# simple slippage as extra cost
slippage_costs = (trades_x * px_x + trades_y * px_y) * slippage

net_pnl = total_pnl - trade_costs - slippage_costs

# portfolio equity series
equity = (1 + net_pnl.cumsum() / capital) * capital
returns = net_pnl / capital  # daily portfolio returns
cum_returns = equity / capital - 1

# 9) METRICS
def annualized_return(rets, periods_per_year=252):
    return (1 + rets.mean())**periods_per_year - 1

def annualized_vol(rets, periods_per_year=252):
    return rets.std() * np.sqrt(periods_per_year)

def sharpe_ratio(rets, periods_per_year=252, rf=0.0):
    ar = annualized_return(rets, periods_per_year)
    av = annualized_vol(rets, periods_per_year)
    if av == 0:
        return np.nan
    return (ar - rf) / av

def max_drawdown(equity_series):
    h = equity_series.cummax()
    dd = (equity_series - h) / h
    return dd.min()

ann_ret = annualized_return(returns)
ann_vol = annualized_vol(returns)
sr = sharpe_ratio(returns)
mdd = max_drawdown(equity)

print(f"Annualized Return: {ann_ret:.2%}, Annual Vol: {ann_vol:.2%}, Sharpe: {sr:.2f}, Max Drawdown: {mdd:.2%}")

# 10) PLOTS: equity and positions
fig, ax = plt.subplots(2,1, figsize=(12,8), sharex=True)
ax[0].plot(equity.index, equity, label='Equity')
ax[0].set_title('Equity Curve')
ax[0].legend()
ax[1].plot(signals.index, signals, label='Signal (1 long spread, -1 short spread)')
ax[1].set_title('Signals / Positions')
ax[1].legend()
plt.show()


