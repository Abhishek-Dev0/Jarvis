"""
market_analysis.py — honest backtesting on historical market data. This is
the market/trading analysis skeleton from the backlog, built the way it was
actually agreed on 2026-08-22: Abi's original ask was a system targeting
90-95% prediction accuracy wired to a live trading account; that was flagged
as unrealistic for any market (anything hitting that in backtest is almost
certainly overfit) and agreed instead to build honest analysis/backtesting
tooling, with risk stated explicitly, and nothing ever wired to a live
account without that being revisited deliberately later.

What's here: fetch real historical OHLCV data (yfinance — works for stocks
and crypto, e.g. "AAPL" or "BTC-USD", no API key), run a few fully
transparent strategies against it (buy-and-hold, SMA crossover, and
ml_signal — a plain logistic regression on named technical features, not a
black box), and report standard honest metrics (return, CAGR, Sharpe, max
drawdown, win rate, trade count) against a buy-and-hold benchmark, with
trading costs modeled so the numbers aren't flattered by pretending trades
are free. ml_signal is fit ONLY on a training prefix of the history and
evaluated ONLY on the untouched remainder — trading it on data it trained
on would be look-ahead bias, exactly the kind of self-deception this module
exists to refuse to hide.

What's NOT here, on purpose: no live trading account connection, no order
execution, no "recommendation" or "signal" language, no accuracy claims.
Every report carries the same disclaimer block. A strategy that backtests
well is not evidence it will perform well going forward — the single most
common way backtesting tools mislead people is overfitting a strategy to
history and presenting that as skill. This module does not protect you from
doing that to yourself if you keep tuning parameters until a number looks
good; it only refuses to hide the fact that that risk exists.

2026-08-25: asked to add "AI prediction," and — in the same conversation —
to build a backtester that "practices until accurate," live exchange API
credential handling, and an auto-execution engine. The first is the exact
overfitting failure mode described above with extra steps; declined, along
with the live/auto-execution pieces, which reverse the 2026-08-22 decision
above without that being revisited deliberately. What got built instead:
ml_signal, an honestly out-of-sample-evaluated strategy signal, held to the
exact same standard as sma_crossover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .base import SkillModule
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule

DISCLAIMER = (
    "This is a backtest on historical data using a simple, published strategy "
    "— not investment advice, not a prediction, and not connected to any live "
    "trading account. Backtested results are prone to overfitting: a strategy "
    "that performed well on this stretch of history is not evidence it will "
    "perform well going forward."
)

_TRADING_DAYS_PER_YEAR = 252


def fetch_history(symbol: str, period: str = "2y", interval: str = "1d"):
    """Real historical OHLCV via yfinance. Works for stocks ('AAPL') and
    crypto ('BTC-USD') through the same call, no API key. Raises if the
    symbol is unknown or nothing came back."""
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"no data returned for '{symbol}' (check the symbol/period)")
    # yfinance returns a MultiIndex (field, ticker) column header even for a
    # single symbol as of the version installed here — flatten it back to
    # plain OHLCV columns.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    return df


# ------------------------------------------------------------------ strategies
# Each strategy takes a price DataFrame and returns a same-length position
# series: 1.0 = fully invested, 0.0 = in cash. Nothing here is proprietary or
# tuned — both are textbook baselines, chosen so there's nothing hidden about
# what's actually being tested.

def buy_and_hold(df) -> np.ndarray:
    return np.ones(len(df))


def sma_crossover(df, fast: int = 20, slow: int = 50) -> np.ndarray:
    """Long while the fast SMA is above the slow SMA, flat otherwise. The
    textbook trend-following baseline — included as a second data point, not
    because it's known to work; see the module disclaimer."""
    close = df["Close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    position = (fast_ma > slow_ma).astype(float)
    position[: max(fast, slow) - 1] = 0.0  # no signal until both MAs exist
    return position.to_numpy()


def _ml_features(df) -> pd.DataFrame:
    """Named, inspectable technical features — not a black box. Every
    column here is a standard, well-known indicator; there's nothing
    hidden about what the model actually sees."""
    close = df["Close"]
    feats = pd.DataFrame(index=df.index)
    feats["return_1d"] = close.pct_change(1)
    feats["return_5d"] = close.pct_change(5)
    feats["sma_ratio"] = close.rolling(10).mean() / close.rolling(50).mean()
    feats["volatility_10d"] = close.pct_change().rolling(10).std()
    if "Volume" in df.columns:
        feats["volume_change_5d"] = df["Volume"].pct_change(5)
    return feats


def ml_signal(df, train_frac: float = 0.6, min_train_rows: int = 30) -> np.ndarray:
    """A plain logistic regression on named technical features, predicting
    next-day up/down. Honestly out-of-sample, not a look-ahead trick: fit
    ONLY on the first train_frac of the history, predict ONLY on the
    untouched remainder. The training portion is reported flat (0.0 —
    no trades) rather than backtested on data the model was fit on, since
    doing that would be the exact overfitting/look-ahead bias this module
    exists to refuse to hide (see the module docstring). If there isn't
    enough valid training data, stays flat everywhere rather than fit on
    too little."""
    from sklearn.linear_model import LogisticRegression

    close = df["Close"]
    features = _ml_features(df)
    next_close = close.shift(-1)
    # NaN > x compares as False, not NaN, in pandas -- target.notna() would
    # never actually catch the last row (no real next-day close to compare
    # against), letting it leak into the test set with a fabricated label.
    # Real bug, caught by testing: track validity from next_close directly.
    has_target = next_close.notna()
    target = (next_close > close).astype(int)

    n = len(df)
    position = np.zeros(n)
    split = int(n * train_frac)

    valid = features.notna().all(axis=1) & has_target
    train_mask = valid & (np.arange(n) < split)
    test_mask = valid & (np.arange(n) >= split)

    if int(train_mask.sum()) < min_train_rows or int(test_mask.sum()) == 0:
        return position  # not enough data to fit/evaluate honestly -- stay flat

    model = LogisticRegression(max_iter=1000)
    model.fit(features[train_mask], target[train_mask])
    position[test_mask.to_numpy()] = model.predict(features[test_mask])
    return position


_STRATEGIES = {"buy_and_hold": buy_and_hold, "sma_crossover": sma_crossover, "ml_signal": ml_signal}

_STRATEGY_NOTES = {
    "ml_signal": (
        "ml_signal only actively trades in its out-of-sample test window (the most "
        "recent portion of the period, by default the last ~40%) -- the earlier portion "
        "is flat/untraded because that's the data the model was fit on. Trading on data "
        "it trained on would be look-ahead bias, not a real backtest."
    ),
}


# ------------------------------------------------------------------- backtest

def backtest(df, position: np.ndarray, cost_bps: float = 5.0) -> dict:
    """Vectorized backtest of a position series against daily returns.
    cost_bps: round-trip trading cost in basis points charged on every
    position *change* — modeling zero cost is the classic way a backtest
    flatters a strategy that trades often; 5bps is a conservative retail
    estimate for liquid symbols, override for your actual venue."""
    close = df["Close"].to_numpy()
    daily_returns = np.diff(close) / close[:-1]
    pos = np.asarray(position, dtype=float)[:-1]  # position held *going into* each return

    trade_changes = np.abs(np.diff(pos, prepend=0.0))
    costs = trade_changes * (cost_bps / 10000.0)

    strategy_returns = pos * daily_returns - costs
    bench_returns = daily_returns  # buy-and-hold, no cost (single entry trade, negligible)

    def _metrics(returns: np.ndarray) -> dict:
        equity = np.cumprod(1.0 + returns)
        total_return = float(equity[-1] - 1.0) if len(equity) else 0.0
        years = len(returns) / _TRADING_DAYS_PER_YEAR
        cagr = float(equity[-1] ** (1 / years) - 1.0) if years > 0 and equity[-1] > 0 else float("nan")
        vol = float(np.std(returns) * np.sqrt(_TRADING_DAYS_PER_YEAR)) if len(returns) else float("nan")
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(_TRADING_DAYS_PER_YEAR)) \
            if len(returns) and np.std(returns) > 0 else float("nan")
        running_max = np.maximum.accumulate(equity) if len(equity) else np.array([1.0])
        drawdown = equity / running_max - 1.0 if len(equity) else np.array([0.0])
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        wins = int(np.sum(returns > 0))
        total_days = int(np.sum(returns != 0))
        win_rate = wins / total_days if total_days else float("nan")
        return {"total_return": total_return, "cagr": cagr, "annualized_vol": vol,
                "sharpe": sharpe, "max_drawdown": max_drawdown, "win_rate": win_rate}

    result = {"strategy": _metrics(strategy_returns), "buy_and_hold": _metrics(bench_returns),
              "num_trades": int(np.sum(trade_changes > 0)), "num_days": len(daily_returns)}
    return result


def run_backtest(symbol: str, strategy: str = "sma_crossover", period: str = "2y",
                  cost_bps: float = 5.0, **strategy_kwargs) -> dict:
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}' (choices: {list(_STRATEGIES)})")
    df = fetch_history(symbol, period=period)
    position = _STRATEGIES[strategy](df, **strategy_kwargs)
    result = backtest(df, position, cost_bps=cost_bps)
    result["symbol"] = symbol
    result["strategy_name"] = strategy
    result["period"] = period
    return result


def format_report(result: dict) -> str:
    s, b = result["strategy"], result["buy_and_hold"]

    def pct(x):
        return f"{x * 100:.1f}%" if x == x else "n/a"  # x == x is False for NaN

    lines = [
        f"Backtest: {result['symbol']} — {result['strategy_name']} vs buy-and-hold "
        f"({result['period']}, {result['num_days']} trading days, {result['num_trades']} trades)",
        "",
        f"{'':14s}{'strategy':>12s}{'buy & hold':>14s}",
        f"{'total return':14s}{pct(s['total_return']):>12s}{pct(b['total_return']):>14s}",
        f"{'CAGR':14s}{pct(s['cagr']):>12s}{pct(b['cagr']):>14s}",
        f"{'sharpe':14s}{s['sharpe']:>12.2f}{b['sharpe']:>14.2f}",
        f"{'max drawdown':14s}{pct(s['max_drawdown']):>12s}{pct(b['max_drawdown']):>14s}",
        f"{'win rate':14s}{pct(s['win_rate']):>12s}{pct(b['win_rate']):>14s}",
    ]
    note = _STRATEGY_NOTES.get(result["strategy_name"])
    if note:
        lines.append("")
        lines.append(note)
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# --------------------------------------------------------------------- skill

_TRIGGERS = ("backtest", "analyze stock", "analyze crypto", "analyze ticker")


class MarketAnalysisSkill(SkillModule):
    """Read-only — no money moves, no gating needed. Runs a default SMA(20/50)
    crossover backtest against buy-and-hold on 2 years of daily data."""

    name = "market_analysis"
    description = "backtests a simple, published strategy against buy-and-hold on real historical data"
    priority = 8  # same tier as web_search — informational, not a physical/OS action

    @property
    def available(self) -> bool:
        try:
            import yfinance  # noqa: F401
            return True
        except ImportError:
            return False

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        return any(t.startswith(p) for p in _TRIGGERS)

    def handle(self, text: str) -> str:
        t = text.strip()
        low = t.lower()
        for p in _TRIGGERS:
            if low.startswith(p):
                rest = t[len(p):].strip().strip("?")
                break
        else:
            rest = ""
        if not rest:
            return ("Backtest which symbol? (e.g. \"backtest AAPL\", \"backtest BTC-USD\", or "
                     f"\"backtest AAPL ml_signal\" to pick a strategy — choices: "
                     f"{', '.join(_STRATEGIES)})")

        parts = rest.split()
        strategy = "sma_crossover"
        if len(parts) > 1 and parts[-1].lower() in _STRATEGIES:
            strategy = parts[-1].lower()
            symbol = " ".join(parts[:-1]).upper()
        else:
            symbol = rest.upper()

        try:
            result = run_backtest(symbol, strategy=strategy)
        except Exception as e:
            return f"Couldn't backtest '{symbol}' ({e})."
        return format_report(result)
