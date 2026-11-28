"""ETF 轮动回测内核。

执行与估值规则
1. 账户按复权价格记账，策略目标在信号后的下一交易日开盘执行。
2. 买入执行价 = 调整开盘 × (1+滑点)；卖出执行价 = 调整开盘 × (1-滑点)。
3. 买入单位 = 可用现金 / [买入执行价 × (1+手续费)]，即先预留费用，现金不允许为负。
4. 卖出后现金增加 = 卖出单位 × 卖出执行价 × (1-手续费)。
5. 换仓顺序：先卖旧仓、扣费，再买新仓、扣费；目标相同不交易；转现金只卖不买。
6. 开盘缺价时取消对应订单。旧仓无法卖出时保留持仓，订单记录取消原因。
7. 收盘估值：总资产 = 现金 + 持仓单位 × 调整收盘（缺收盘价日沿用最近可得收盘）。
8. 日净值记录收盘值，指标以初始资金对应的净值 1 为起点。期末按收盘市值估值。

支持三种模式：
- mode="strategy"    ：读取 targets 字典（signal_date -> "CASH" 或 ETF代码），单资产轮动（E1/E2/E3）
- mode="equal_weight"：每月执行日开盘再平衡至资产池等权（E0基准）
- mode="buy_hold"    ：首日全仓买入 benchmark 后不再交易（买入持有基准）

输出：orders / trades / holdings_daily / nav_daily 四张表（DataFrame）。
"""
from __future__ import annotations

import pandas as pd

DEFAULT_FEE = 0.0003
DEFAULT_SLIP = 0.0002
DEFAULT_CASH = 100_000.0


class _Account:
    """单一策略账户状态：现金、持仓单位、最近可得收盘价。"""

    def __init__(self, symbols, initial_cash):
        self.cash = float(initial_cash)
        self.units = {s: 0.0 for s in symbols}
        self.last_close = {}  # symbol -> 最近可得调整收盘

    def holdings_value(self, prices=None):
        """持仓市值。prices 为None时用最近可得收盘（估值用途）。"""
        total = 0.0
        for s, u in self.units.items():
            if u > 0:
                px = prices.get(s) if prices else None
                if px is None:
                    px = self.last_close.get(s, 0.0)
                total += u * px
        return total

    def total_assets(self, prices=None):
        return self.cash + self.holdings_value(prices)


def run_backtest(prices: pd.DataFrame,
                 trading_days: list,
                 config: dict,
                 mode: str = "strategy",
                 targets: dict | None = None) -> dict:
    """运行一次回测。

    参数
    ----
    prices        : 清洗后的 ETF 日线，含 date、symbol、open_adj 和 close_adj
    trading_days  : 市场交易日列表（升序，来自日历 is_open=1 的日子，datetime）
    config        : dict，含 initial_cash / fee_rate / slippage / symbols /
                    start_date / end_date / benchmark_symbol
    mode          : "strategy" | "equal_weight" | "buy_hold"
    targets       : mode="strategy" 时必须提供 {signal_date(Timestamp): "CASH"或代码}

    返回
    ----
    dict(orders, trades, holdings_daily, nav_daily) 四个 DataFrame
    """
    assert mode in ("strategy", "equal_weight", "buy_hold"), f"未知mode: {mode}"
    if mode == "strategy":
        assert targets is not None, "strategy模式必须提供targets"

    symbols = list(config["symbols"])
    fee = float(config.get("fee_rate", DEFAULT_FEE))
    slip = float(config.get("slippage", DEFAULT_SLIP))
    cash0 = float(config.get("initial_cash", DEFAULT_CASH))
    bench = config.get("benchmark_symbol", "510300.SH")

    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])

    # ---- 价格查询表：{symbol: {date: price}} ----
    open_px: dict[str, dict] = {s: {} for s in symbols}
    close_px: dict[str, dict] = {s: {} for s in symbols}
    for d, s, o, c in zip(prices["date"], prices["symbol"],
                          prices["open_adj"], prices["close_adj"]):
        if s in open_px:
            open_px[s][d] = float(o)
            close_px[s][d] = float(c)

    # ---- 交易日序列（限定回测区间）----
    calendar = sorted({pd.Timestamp(d) for d in trading_days})
    days = [d for d in calendar if start <= d <= end]
    if not days:
        raise ValueError("回测区间内没有交易日，请检查start_date/end_date")
    # 在完整日历上定位次日，然后限制成交日期。样本前一交易日的信号
    # 可能在样本首日成交，更早且已经过了执行日的信号不能补单。
    next_open = dict(zip(calendar, calendar[1:]))
    previous_open = dict(zip(calendar[1:], calendar))

    # ---- 生成执行计划 pending: (exec_date, action, payload) ----
    pendings = []  # list of dict(exec_date, kind, target, signal_date)
    if mode == "strategy":
        for sig, tgt in targets.items():
            sig = pd.Timestamp(sig)
            ex = next_open.get(sig)
            if ex is None or not start <= ex <= end:
                continue
            if tgt not in ("CASH",) and tgt not in symbols:
                raise ValueError(f"非法目标 {tgt}")
            pendings.append({"exec_date": ex, "kind": "switch",
                             "target": tgt, "signal_date": sig})
    elif mode == "equal_weight":
        for i, day in enumerate(days):
            prev = previous_open.get(day)
            # 首日按样本设定建仓，之后仅在月首再平衡。
            if i == 0 or (prev.year, prev.month) != (day.year, day.month):
                pendings.append({"exec_date": day, "kind": "rebalance",
                                 "target": None, "signal_date": prev})
    else:  # 买入持有基准在区间首个交易日建仓
        pendings.append({"exec_date": days[0], "kind": "buy_hold",
                         "target": bench, "signal_date": None})
    pendings.sort(key=lambda p: p["exec_date"])
    pending_by_day: dict = {}
    for p in pendings:
        pending_by_day.setdefault(p["exec_date"], []).append(p)

    # ---- 账户与记录 ----
    acct = _Account(symbols, cash0)
    orders, trades, holdings_rows, nav_rows = [], [], [], []

    def record_trade(t, sig, symbol, side, units, ref_px, exec_px, fee_amt, reason):
        trades.append({
            "trade_date": t, "signal_date": sig, "strategy": mode,
            "symbol": symbol, "side": side, "units": round(units, 6),
            "reference_price": round(ref_px, 6),
            "execution_price": round(exec_px, 6),
            "fee": round(fee_amt, 6),
            "cash_after": round(acct.cash, 6), "reason": reason,
        })

    # 每日依次处理开盘成交与收盘估值。
    for t in days:
        # ---------- 1. 开盘：执行此前已知的目标 ----------
        for pend in pending_by_day.get(t, []):
            sig = pend["signal_date"]
            if pend["kind"] == "switch":
                tgt = pend["target"]
                held = [s for s, u in acct.units.items() if u > 0]
                held_sym = held[0] if held else None
                if tgt == held_sym:
                    orders.append({"exec_date": t, "strategy": mode,
                                   "signal_date": sig, "target": tgt,
                                   "status": "NO_CHANGE", "reason": "目标与持仓相同，不交易"})
                    continue
                # 先卖旧仓
                if held_sym is not None:
                    o_sell = open_px[held_sym].get(t)
                    if o_sell is None:
                        orders.append({"exec_date": t, "strategy": mode,
                                       "signal_date": sig, "target": tgt,
                                       "status": "FAILED",
                                       "reason": f"旧仓{held_sym}开盘缺价，保留旧仓至下次月度计划"})
                        continue  # 卖不出旧仓：既不卖也不买
                    exec_px = o_sell * (1 - slip)
                    u = acct.units[held_sym]
                    fee_amt = u * exec_px * fee
                    acct.cash += u * exec_px * (1 - fee)
                    acct.units[held_sym] = 0.0
                    record_trade(t, sig, held_sym, "SELL", u, o_sell,
                                 exec_px, fee_amt, f"卖出旧仓->准备转向{tgt}")
                # 再买新仓（转现金时tgt为CASH，跳过）
                if tgt != "CASH":
                    o_buy = open_px[tgt].get(t)
                    if o_buy is None:
                        orders.append({"exec_date": t, "strategy": mode,
                                       "signal_date": sig, "target": tgt,
                                       "status": "FAILED",
                                       "reason": "目标ETF开盘缺价，买单取消，持有现金"})
                        continue
                    exec_px = o_buy * (1 + slip)
                    u = acct.cash / (exec_px * (1 + fee))  # 先预留费用
                    fee_amt = u * exec_px * fee
                    acct.cash -= u * exec_px * (1 + fee)
                    acct.units[tgt] += u
                    assert acct.cash >= -1e-6, "现金不允许为负"
                    record_trade(t, sig, tgt, "BUY", u, o_buy,
                                 exec_px, fee_amt, f"买入新仓(来自{sig.date() if sig else '初始'}信号)")
                else:
                    orders.append({"exec_date": t, "strategy": mode,
                                   "signal_date": sig, "target": "CASH",
                                   "status": "EXECUTED", "reason": "转为空仓持有现金"})
            elif pend["kind"] == "buy_hold":
                o_buy = open_px[pend["target"]].get(t)
                if o_buy is None:
                    orders.append({"exec_date": t, "strategy": mode,
                                   "signal_date": None, "target": pend["target"],
                                   "status": "FAILED", "reason": "基准建仓日开盘缺价"})
                    continue
                exec_px = o_buy * (1 + slip)
                u = acct.cash / (exec_px * (1 + fee))
                fee_amt = u * exec_px * fee
                acct.cash -= u * exec_px * (1 + fee)
                acct.units[pend["target"]] += u
                record_trade(t, None, pend["target"], "BUY", u, o_buy,
                             exec_px, fee_amt, "买入持有基准首次建仓")
            elif pend["kind"] == "rebalance":
                ok = _rebalance_equal_weight(acct, open_px, symbols, t,
                                             fee, slip, mode, sig, orders,
                                             record_trade)
                _ = ok

        # ---------- 2. 收盘估值 ----------
        # 更新最近可得收盘（缺收盘价日沿用上一可得值）
        for s in symbols:
            c = close_px[s].get(t)
            if c is not None:
                acct.last_close[s] = c
        total = acct.total_assets()  # 用最近可得收盘估值
        nav = total / cash0
        nav_rows.append({"date": t, "strategy": mode, "cash": round(acct.cash, 6),
                         "holdings_value": round(acct.holdings_value(), 6),
                         "total_assets": round(total, 6), "nav": round(nav, 10)})

        # ---------- 3. 记录持仓 ----------
        row = {"date": t, "strategy": mode, "cash": round(acct.cash, 6)}
        for s in symbols:
            row[f"units_{s}"] = round(acct.units[s], 6)
            row[f"value_{s}"] = round(acct.units[s] * acct.last_close.get(s, 0.0), 6)
        row["total_assets"] = round(total, 6)
        holdings_rows.append(row)

    orders_df = pd.DataFrame(orders)
    trades_df = pd.DataFrame(trades)
    holdings_df = pd.DataFrame(holdings_rows)
    nav_df = pd.DataFrame(nav_rows)
    return {"orders": orders_df, "trades": trades_df,
            "holdings_daily": holdings_df, "nav_daily": nav_df}


def _rebalance_equal_weight(acct, open_px, symbols, t, fee, slip,
                            mode, sig, orders, record_trade):
    """E0等权基准：每月执行日开盘按账户总资产再平衡至25%×4。
    先卖超配、再买低配、预留费用；买单资金不足按统一比例缩减。"""
    opens = {}
    for s in symbols:
        o = open_px[s].get(t)
        if o is None:
            orders.append({"exec_date": t, "strategy": mode, "signal_date": sig,
                           "target": "EQUAL_WEIGHT", "status": "FAILED",
                           "reason": f"{s}开盘缺价，本月再平衡取消"})
            return False
        opens[s] = o

    total = acct.cash + sum(acct.units[s] * opens[s] for s in symbols)
    target_v = total / len(symbols)

    # 先卖超配
    for s in symbols:
        cur = acct.units[s] * opens[s]
        if cur > target_v + 1e-9 and acct.units[s] > 0:
            exec_px = opens[s] * (1 - slip)
            u_sell = (cur - target_v) / exec_px / (1 - fee)
            u_sell = min(u_sell, acct.units[s])
            fee_amt = u_sell * exec_px * fee
            acct.cash += u_sell * exec_px * (1 - fee)
            acct.units[s] -= u_sell
            record_trade(t, sig, s, "SELL", u_sell, opens[s], exec_px,
                         fee_amt, "等权再平衡卖出超配")

    # 再买低配（资金不足按统一比例缩减）
    buys = []
    for s in symbols:
        cur = acct.units[s] * opens[s]
        need = target_v - cur
        if need > 1e-9:
            exec_px = opens[s] * (1 + slip)
            u_buy = need / exec_px            # 未含费
            cost = u_buy * exec_px * (1 + fee)
            buys.append([s, u_buy, exec_px, cost])
    total_cost = sum(b[3] for b in buys)
    if buys and total_cost > acct.cash:
        scale = acct.cash / total_cost
        for b in buys:
            b[1] *= scale
            b[3] *= scale
    for s, u_buy, exec_px, cost in buys:
        if cost > acct.cash + 1e-6:
            continue
        fee_amt = u_buy * exec_px * fee
        acct.cash -= cost
        acct.units[s] += u_buy
        record_trade(t, sig, s, "BUY", u_buy, exec_px / (1 + slip), exec_px,
                     fee_amt, "等权再平衡买入低配")
    orders.append({"exec_date": t, "strategy": mode, "signal_date": sig,
                   "target": "EQUAL_WEIGHT", "status": "EXECUTED",
                   "reason": "月度等权再平衡"})
    return True
