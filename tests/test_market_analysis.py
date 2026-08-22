import pandas as pd
import pytest

from jarvis.modules.market_analysis import (
    DISCLAIMER, backtest, buy_and_hold, format_report, sma_crossover,
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
