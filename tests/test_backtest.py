"""回测引擎的账户核算与信号时序测试。

必须通过的测试：
1. 恒定价格+零费用：任何买卖都不亏不赚，净值恒为1
2. 恒定价格+存在费用：每笔交易产生费用，期末净值必须低于1
3. 任何一天 现金+持仓市值 = 总资产（资金守恒）
4. 时序/未来函数：修改未来行情，不影响此前的净值与信号执行
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest import run_backtest  # noqa: E402

SYMS = ["AAA.SH", "BBB.SZ"]
T0 = pd.Timestamp("2016-01-04")


def make_days(n=300, start=T0):
    bdays = pd.bdate_range(start, periods=n)
    return list(bdays)


def make_prices(days, price=1.0, later_change=None):
    """构造行情：恒定价格；later_change={date: multiplier} 在指定日改价格。"""
    rows = []
    for d in days:
        for s, suff in zip(SYMS, [".SH", ".SZ"]):
            p = price
            if later_change and d in later_change:
                p = price * later_change[d]
            rows.append({"date": d, "symbol": s, "open_adj": p,
                         "close_adj": p, "open_raw": p, "close_raw": p,
                         "volume": 1000.0})
    return pd.DataFrame(rows)


def base_cfg(days, fee=0.0, slip=0.0):
    return {"symbols": SYMS, "initial_cash": 100000.0,
            "fee_rate": fee, "slippage": slip,
            "start_date": str(days[0].date()),
            "end_date": str(days[-1].date()),
            "benchmark_symbol": "AAA.SH"}


def month_ends(days):
    by_month = {}
    for d in days:
        by_month.setdefault((d.year, d.month), []).append(d)
    return {v[-1] for v in by_month.values()}


def alternate_targets(days):
    """月末交替目标：奇数月买AAA，偶数月买BBB，制造持续换仓。"""
    tgts = {}
    for d in sorted(month_ends(days)):
        tgts[d] = "AAA.SH" if d.month % 2 == 1 else "BBB.SZ"
    return tgts


def test_constant_price_zero_cost_nav_unchanged():
    days = make_days(300)
    prices = make_prices(days, price=1.0)
    res = run_backtest(prices, days, base_cfg(days, fee=0.0, slip=0.0),
                       mode="strategy", targets=alternate_targets(days))
    nav = res["nav_daily"]["nav"]
    assert (np.abs(nav - 1.0) < 1e-9).all(), "恒定价格零费用下净值应恒为1"
    assert len(res["trades"]) > 0, "应存在换仓成交"


def test_constant_price_with_fee_nav_drops():
    days = make_days(300)
    prices = make_prices(days, price=1.0)
    res = run_backtest(prices, days, base_cfg(days, fee=0.0003, slip=0.0002),
                       mode="strategy", targets=alternate_targets(days))
    nav = res["nav_daily"]["nav"]
    assert nav.iloc[-1] < 1.0, "恒定价格下费用必须使净值降低"
    # 每笔BUY后现金≈0，每笔SELL后现金增加
    tr = res["trades"]
    buys = tr[tr["side"] == "BUY"]
    assert (buys["cash_after"] >= -1e-6).all(), "买入后现金不允许为负"


def test_cash_plus_holdings_equals_total():
    days = make_days(300)
    prices = make_prices(days, price=1.0)
    res = run_backtest(prices, days, base_cfg(days, fee=0.0003, slip=0.0002),
                       mode="strategy", targets=alternate_targets(days))
    h = res["holdings_daily"]
    parts = h["cash"] + sum(h[f"value_{s}"] for s in SYMS)
    assert (np.abs(parts - h["total_assets"]) < 1e-6).all(), "资金守恒被破坏"
    n = res["nav_daily"]
    assert (np.abs(n["cash"] + n["holdings_value"] - n["total_assets"]) < 1e-6).all()


def test_no_lookahead_future_change_not_affect_past():
    """截断重演：把2021年后的价格改掉，2020年前的净值必须完全一致。"""
    days = make_days(800)
    cut = days[500]
    p1 = make_prices(days, price=1.0)
    p2 = make_prices(days, price=1.0, later_change={d: 3.7 for d in days[500:]})
    tgts = alternate_targets(days)
    r1 = run_backtest(p1, days, base_cfg(days, fee=0.0003, slip=0.0002),
                      mode="strategy", targets=tgts)
    r2 = run_backtest(p2, days, base_cfg(days, fee=0.0003, slip=0.0002),
                      mode="strategy", targets=tgts)
    past = r1["nav_daily"]["date"] < cut
    np.testing.assert_allclose(r1["nav_daily"]["nav"][past],
                               r2["nav_daily"]["nav"][past], atol=1e-9)


def test_equal_weight_total_invested():
    """等权基准：再平衡后单只市值约等于总资产25%。"""
    days = make_days(300)
    prices = make_prices(days, price=1.0)
    res = run_backtest(prices, days, base_cfg(days, fee=0.0003, slip=0.0002),
                       mode="equal_weight")
    h = res["holdings_daily"].iloc[30]
    vals = np.array([h[f"value_{s}"] for s in SYMS])
    total = h["total_assets"]
    assert (np.abs(vals - total / len(SYMS)) / total < 0.02).all(), \
        "恒定价格下等权再平衡后各标的应约25%"


def test_split_start_uses_prior_month_end_without_replaying_older_orders():
    days = list(pd.to_datetime(["2023-11-30", "2023-12-01", "2023-12-29",
                                "2024-01-02", "2024-01-03", "2024-01-31"]))
    config = base_cfg(days, fee=.0003, slip=.0002)
    config["start_date"] = "2024-01-01"
    prices = make_prices(days)
    targets = {days[0]: "AAA.SH", days[2]: "BBB.SZ", days[-1]: "AAA.SH"}
    result = run_backtest(prices, days, config, targets=targets)
    assert result["trades"]["symbol"].tolist() == ["BBB.SZ"]
    assert result["trades"]["signal_date"].tolist() == [days[2]]
    assert result["trades"]["trade_date"].tolist() == [days[3]]
    assert result["nav_daily"].iloc[0]["nav"] < 1
    for mode in ("equal_weight", "buy_hold"):
        benchmark = run_backtest(prices, days, config, mode=mode)
        assert benchmark["trades"]["trade_date"].drop_duplicates().tolist() == [days[3]]
