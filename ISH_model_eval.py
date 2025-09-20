import os
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, 'data', 'processed')
assets = ["S&P500", "FTSE100", "Nikkei225", "EEM", "Gold", "UST10Y"]
test_df = pd.read_csv(os.path.join(input_dir, "test_norm.csv"), index_col=0)
actual_ret = test_df[[f"{a}_ret" for a in assets]].values[60:]
dates = test_df.index[60:]
train_mean = np.load(os.path.join(input_dir, 'train_ret_mean.npy'))
train_std = np.load(os.path.join(input_dir, 'train_ret_std.npy'))

def safe_cov(submat):
    try:
        cov = np.cov(submat.T)
        cov += np.eye(submat.shape[1]) * 1e-3
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            return np.eye(submat.shape[1]) * 1e-6
        return cov
    except Exception:
        return np.eye(submat.shape[1]) * 1e-6

def safe_nav_calc(nav_series):
    nav = np.array(nav_series)
    nav = nav[nav > 0]
    if len(nav) < 2: return None
    rets = np.diff(np.log(nav))
    if not np.all(np.isfinite(rets)): return None
    ar = np.exp(np.mean(rets) * 252) - 1
    av = np.std(rets) * np.sqrt(252)
    sharpe = ar / av if av != 0 else np.nan
    mdd = np.max(np.maximum.accumulate(nav) - nav) / np.max(nav)
    calmar = ar / mdd if mdd != 0 else np.nan
    return {
        'nav': nav,
        'ar': ar,
        'av': av,
        'sharpe': sharpe,
        'mdd': mdd,
        'calmar': calmar
    }

results = {}
for model_name, pred_file in [
    ('LSTM', 'pred_returns_lstm.csv'),
    ('Transformer', 'pred_returns_transformer.csv')
]:
    pred_df = pd.read_csv(os.path.join(base_dir, pred_file), index_col=0)
    pred_returns = pred_df.values
    for i in range(len(pred_returns)):
        pred_returns[i] = pred_returns[i] * train_std + train_mean
    print(f"\n==== {model_name} predicted returns stats (de-normalized) ====")
    print("min:", np.min(pred_returns), "max:", np.max(pred_returns),
          "mean:", np.mean(pred_returns), "std:", np.std(pred_returns))
    print("Sample predictions (first 2 rows):\n", pred_returns[:2])
    initial_value = 100.0
    window = 60
    nav = [initial_value]
    current_weights = np.ones(len(assets)) / len(assets)
    fallback_counts = 0
    for i in range(len(pred_returns)):
        if i < window: continue
        cov = safe_cov(actual_ret[i-window+1:i+1])
        # CLIP for robust allocation
        expected = np.clip(pred_returns[i], -0.01, 0.01)
        try:
            ef = EfficientFrontier(expected, cov)
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: w <= 0.4)
            ef.add_constraint(lambda w: sum(w) == 1)
            weights = np.array(list(ef.max_sharpe().values()))
            # VOLATILITY TARGETING (avoid NAV death spiral)
            portfolio_vol = np.sqrt(weights @ cov @ weights.T) * np.sqrt(252)
            target_vol = 0.10
            if portfolio_vol > 0:
                scale = min(1.0, target_vol / portfolio_vol)
                weights *= scale
            if (np.any(weights < 0) or np.any(weights > 0.4)
                or not np.all(np.isfinite(weights))):
                print(f"Invalid weights on day {i}, fallback to equal weights. Orig: {weights}")
                weights = np.ones(len(assets)) / len(assets)
                fallback_counts += 1
        except Exception as e:
            print(f"Optimizer error on day {i}: {e}")
            weights = current_weights
            fallback_counts += 1
        realized_ret = np.dot(actual_ret[i], weights)
        if not np.isfinite(realized_ret):
            nav.append(nav[-1])
            continue
        turnover = np.abs(weights - current_weights).sum()
        tc = 0.001 * turnover
        nav_value = nav[-1] * (1 + realized_ret - tc)
        if nav_value <= 0 or not np.isfinite(nav_value):
            print(f"NAV became non-positive ({nav_value}) on day {i}; stopping {model_name} backtest.")
            break
        nav.append(nav_value)
        current_weights = weights
    print(f"Fallback to equal weights occurred {fallback_counts} times ({100*fallback_counts/max(1, len(pred_returns)-window):.1f}%)")
    metrics = safe_nav_calc(nav)
    if metrics is not None:
        results[model_name] = metrics
        print(f"\n==== {model_name} Results ====")
        print(f"Annualized Return: {metrics['ar']:.2%}")
        print(f"Annualized Volatility: {metrics['av']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
        print(f"Maximum Drawdown: {metrics['mdd']:.2%}")
        print(f"Calmar Ratio: {metrics['calmar']:.2f}")
    else:
        results[model_name] = {'nav': nav}
        print("\n==== {model_name} Results ====\nNot enough valid NAV points.")

plt.figure(figsize=(10, 5))
for model_name in results:
    nav = results[model_name]['nav']
    if nav is not None and len(nav) > 1:
        plt.plot(np.arange(len(nav)), nav, label=model_name)
plt.title("Portfolio NAV Comparison: LSTM vs Transformer (De-normalized)")
plt.xlabel("Time (days)")
plt.ylabel("Net Asset Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
