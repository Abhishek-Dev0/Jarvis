import numpy as np
import pandas as pd
import pytest

from jarvis.modules.market_analysis import (
    DISCLAIMER, MarketAnalysisSkill, _STRATEGY_NOTES, backtest,
    bollinger_bands, buy_and_hold, format_report, macd, ml_signal, rsi,
    run_backtest, sma_crossover, volatility,
)


def _price_df(prices):
    return pd.DataFrame({"Close": prices})


def test_buy_and_hold_is_always_fully_invested():
    df = _price_df([100, 101, 99, 105])
    position = buy_and_hold(df)
    assert list(position) == [1.0, 1.0, 1.0, 1.0]


def test_buy_and_hold_backtest_matches_simple_return():
    prices = [100.0, 110.0, 121.0]  # +10% each day
    df = _price_df(prices)
    result = backtest(df, buy_and_hold(df), cost_bps=0.0)
    # total return over the whole window: 121/100 - 1 = 0.21
    assert result["buy_and_hold"]["total_return"] == pytest.approx(0.21, abs=1e-9)
    assert result["strategy"]["total_return"] == pytest.approx(0.21, abs=1e-9)


def test_backtest_costs_reduce_strategy_return_relative_to_buy_and_hold():
    prices = [100.0, 110.0, 100.0, 110.0, 100.0, 110.0]
    df = _price_df(prices)
    # sma_crossover on this short/noisy series will trade; a nonzero cost
    # must make the strategy's return worse than the zero-cost case.
    position = sma_crossover(df, fast=1, slow=2)
    free = backtest(df, position, cost_bps=0.0)
    costly = backtest(df, position, cost_bps=50.0)
    assert costly["strategy"]["total_return"] <= free["strategy"]["total_return"]


def test_sma_crossover_flat_before_both_averages_exist():
    df = _price_df(list(range(100, 160)))  # monotonic, 60 points
    position = sma_crossover(df, fast=5, slow=20)
    assert all(p == 0.0 for p in position[:19])  # slow-1 warmup


def test_backtest_num_trades_counts_position_changes():
    df = _price_df([100.0, 101.0, 102.0, 103.0])
    # backtest() uses position[:-1] (the position held *going into* each of
    # the 3 daily returns) -- the position array's last entry has no return
    # to apply to, so only the first 3 values [0.0, 1.0, 0.0] count here:
    # 0->1 and 1->0 are two real transitions.
    position = [0.0, 1.0, 0.0, 1.0]
    result = backtest(df, position, cost_bps=0.0)
    assert result["num_trades"] == 2


def _synthetic_ohlcv(n=200, seed=7):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume})


def test_ml_signal_stays_flat_during_its_own_training_window():
    df = _synthetic_ohlcv(n=200)
    position = ml_signal(df, train_frac=0.6)
    split = int(len(df) * 0.6)
    # Trading on data the model was fit on would be look-ahead bias -- the
    # training prefix must be exactly flat regardless of what the model
    # would have predicted there.
    assert all(p == 0.0 for p in position[:split])


def test_ml_signal_only_trades_in_valid_test_rows():
    df = _synthetic_ohlcv(n=200)
    position = ml_signal(df, train_frac=0.6)
    assert len(position) == len(df)
    assert set(np.unique(position)) <= {0.0, 1.0}
    # the last row has no next-day target -- must never be traded
    assert position[-1] == 0.0


def test_ml_signal_stays_flat_with_insufficient_training_data():
    df = _synthetic_ohlcv(n=20)  # far below min_train_rows
    position = ml_signal(df, train_frac=0.6)
    assert all(p == 0.0 for p in position)


def test_ml_signal_is_registered_and_runs_through_run_backtest(monkeypatch):
    from jarvis.modules import market_analysis
    df = _synthetic_ohlcv(n=200)
    monkeypatch.setattr(market_analysis, "fetch_history", lambda symbol, period="2y": df)

    result = run_backtest("TEST", strategy="ml_signal")
    assert result["strategy_name"] == "ml_signal"
    report = format_report(result)
    assert _STRATEGY_NOTES["ml_signal"] in report
    assert DISCLAIMER in report


def test_skill_handle_accepts_a_trailing_strategy_name(monkeypatch):
    from jarvis.modules import market_analysis
    df = _synthetic_ohlcv(n=200)
    monkeypatch.setattr(market_analysis, "fetch_history", lambda symbol, period="2y": df)

    sk = MarketAnalysisSkill()
    reply = sk.handle("backtest AAPL ml_signal")
    assert "ml_signal" in reply
    assert "AAPL" in reply


def test_skill_handle_defaults_to_sma_crossover_without_a_strategy_name(monkeypatch):
    from jarvis.modules import market_analysis
    df = _synthetic_ohlcv(n=200)
    monkeypatch.setattr(market_analysis, "fetch_history", lambda symbol, period="2y": df)

    sk = MarketAnalysisSkill()
    reply = sk.handle("backtest AAPL")
    assert "sma_crossover" in reply


# --------------------------------------------------------------- indicators

def test_rsi_approaches_100_on_a_pure_uptrend():
    close = pd.Series(range(100, 140))  # strictly increasing -- no losses at all
    value = rsi(close, period=14)
    assert value.iloc[-1] > 95


def test_rsi_approaches_0_on_a_pure_downtrend():
    close = pd.Series(range(140, 100, -1))  # strictly decreasing -- no gains at all
    value = rsi(close, period=14)
    assert value.iloc[-1] < 5


def test_macd_is_flat_zero_on_a_constant_price():
    close = pd.Series([100.0] * 60)
    macd_line, signal_line, histogram = macd(close)
    assert macd_line.iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert histogram.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_bollinger_bands_collapse_to_the_price_when_flat():
    close = pd.Series([100.0] * 30)
    upper, middle, lower = bollinger_bands(close, window=20)
    assert upper.iloc[-1] == pytest.approx(100.0)
    assert middle.iloc[-1] == pytest.approx(100.0)
    assert lower.iloc[-1] == pytest.approx(100.0)


def test_bollinger_bands_widen_with_more_volatility():
    calm = pd.Series([100.0, 100.5, 99.5, 100.2, 99.8] * 6)
    wild = pd.Series([100.0, 110.0, 90.0, 108.0, 92.0] * 6)
    calm_upper, _, calm_lower = bollinger_bands(calm, window=20)
    wild_upper, _, wild_lower = bollinger_bands(wild, window=20)
    assert (wild_upper.iloc[-1] - wild_lower.iloc[-1]) > (calm_upper.iloc[-1] - calm_lower.iloc[-1])


def test_volatility_is_zero_for_a_constant_price():
    close = pd.Series([100.0] * 20)
    value = volatility(close, window=10)
    assert value.iloc[-1] == pytest.approx(0.0)


# ----------------------------------------------------------------- watchlist

def test_watchlist_empty_when_file_absent(tmp_path):
    from jarvis.modules.market_analysis import load_watchlist
    assert load_watchlist(str(tmp_path / "missing.json")) == []


def test_add_to_watchlist_persists_and_dedupes(tmp_path):
    from jarvis.modules.market_analysis import add_to_watchlist, load_watchlist
    path = str(tmp_path / "watchlist.json")
    add_to_watchlist("aapl", path)
    add_to_watchlist("BTC-USD", path)
    add_to_watchlist("AAPL", path)  # same symbol, different case -- must not duplicate
    assert load_watchlist(path) == ["AAPL", "BTC-USD"]


def test_remove_from_watchlist(tmp_path):
    from jarvis.modules.market_analysis import (
        add_to_watchlist, load_watchlist, remove_from_watchlist,
    )
    path = str(tmp_path / "watchlist.json")
    add_to_watchlist("AAPL", path)
    add_to_watchlist("MSFT", path)
    remove_from_watchlist("aapl", path)
    assert load_watchlist(path) == ["MSFT"]


def test_format_report_always_includes_disclaimer():
    df = _price_df([100.0, 101.0, 102.0])
    result = backtest(df, buy_and_hold(df))
    result["symbol"] = "TEST"
    result["strategy_name"] = "buy_and_hold"
    result["period"] = "test"
    report = format_report(result)
    assert DISCLAIMER in report
    assert "not investment advice" in report
    assert "TEST" in report
