"""按配置运行 ETF 回测。

读取 ETF 日线、交易日历和月末目标，运行轮动策略或基准。
订单、成交、逐日持仓、净值和绩效指标保存在 results/<run_id>/。

用法：
    python main.py                          # 全区间 2016-2025 基础成本
    python main.py --split dev              # 只跑开发期
    python main.py --cost zero_cost         # 指定成本实验
    python main.py --list-strategies E3,E0  # 只跑指定策略
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from backtest import run_backtest

ROOT = Path(__file__).resolve().parent


def load_config(path="config.json"):
    with open(ROOT / path, encoding="utf-8") as f:
        if Path(path).suffix.lower() == ".json":
            return json.load(f)
        import yaml
        return yaml.safe_load(f)


def load_data(cfg):
    prices = pd.read_csv(ROOT / cfg["data"]["prices_csv"], encoding="utf-8-sig",
                         parse_dates=["date"])
    cal = pd.concat([
        pd.read_csv(ROOT / cfg["data"]["calendar_sse_csv"], encoding="utf-8-sig"),
        pd.read_csv(ROOT / cfg["data"]["calendar_szse_csv"], encoding="utf-8-sig"),
    ], ignore_index=True)
    open_days = sorted(set(
        pd.to_datetime(cal.loc[cal["is_open"].astype(str).eq("1"), "cal_date"]
                       .astype(str), format="%Y%m%d")))
    targets_df = pd.read_csv(ROOT / cfg["data"]["targets_csv"], encoding="utf-8-sig",
                             parse_dates=["signal_date"])
    return prices, open_days, targets_df


def build_targets(targets_df: pd.DataFrame, mode: str) -> dict:
    """提取指定策略的月末目标，按信号日期索引。"""
    col = {"E1": "E1_target", "E2": "E2_target", "E3": "E3_target"}[mode]
    t = {}
    for sig, tgt in zip(targets_df["signal_date"], targets_df[col]):
        t[sig] = str(tgt)
    return t


def quick_metrics(nav: pd.Series) -> dict:
    """计算累计收益、年化收益、最大回撤与夏普比率。"""
    series = pd.concat([pd.Series([1.0]), nav], ignore_index=True)
    rets = series.pct_change().dropna()
    cum = series.iloc[-1] - 1
    annual = series.iloc[-1] ** (252 / len(rets)) - 1
    dd = (series / series.cummax() - 1).min()
    std = rets.std(ddof=1)
    sharpe = rets.mean() / std * (252 ** 0.5) if std > 0 else float("nan")
    return {"累计收益率": round(cum, 4), "年化收益率": round(annual, 4),
            "最大回撤": round(abs(dd), 4), "夏普比率": round(sharpe, 3)}


def run_all(cfg, split=None, cost_name=None, strategies=None):
    prices, open_days, targets_df = load_data(cfg)

    bt_cfg = dict(cfg["backtest"])
    if split:
        sp = cfg["sample_splits"][split]
        bt_cfg["start_date"], bt_cfg["end_date"] = sp["start"], sp["end"]

    cost = {c["name"]: c for c in cfg["cost_experiments"]}
    if cost_name:
        c = cost[cost_name]
        bt_cfg["fee_rate"], bt_cfg["slippage"] = c["fee_rate"], c["slippage"]

    base_cfg = {"initial_cash": cfg["account"]["initial_cash"],
                "symbols": cfg["asset_pool"],
                "benchmark_symbol": cfg["backtest"]["benchmark_symbol"],
                "fee_rate": cfg["account"]["fee_rate"],
                "slippage": cfg["account"]["slippage"],
                "start_date": bt_cfg["start_date"],
                "end_date": bt_cfg["end_date"]}
    if cost_name:  # 成本实验覆盖基础费率（保持其余规则不变）
        c = cost[cost_name]
        base_cfg["fee_rate"], base_cfg["slippage"] = c["fee_rate"], c["slippage"]

    jobs = []
    for m in ("E1", "E2", "E3"):
        if strategies and m not in strategies:
            continue
        jobs.append((m, "strategy", build_targets(targets_df, m)))
    if not strategies or "E0" in strategies:
        jobs.append(("E0", "equal_weight", None))
    if not strategies or "BUYHOLD" in strategies:
        jobs.append(("BUYHOLD", "buy_hold", None))

    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = ROOT / cfg["data"]["results_dir"] / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    summary = []
    for name, mode, tgt in jobs:
        res = run_backtest(prices, open_days, base_cfg, mode=mode, targets=tgt)
        for key, df in res.items():
            df.to_csv(out_dir / f"{name}_{key}.csv", index=False, encoding="utf-8-sig")
        m = quick_metrics(res["nav_daily"]["nav"])
        summary.append({"strategy": name, "mode": mode,
                        "fee": base_cfg["fee_rate"], "slip": base_cfg["slippage"],
                        "start": base_cfg["start_date"], "end": base_cfg["end_date"],
                        "trades": len(res["trades"]), **m})
        print(f"[done] {name:8s} 期末净值 {res['nav_daily']['nav'].iloc[-1]:.4f}  "
              f"成交{len(res['trades'])}笔")

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    meta = {"run_id": run_id, "config": base_cfg, "split": split,
            "cost_experiment": cost_name, "data_rows": len(prices),
            "trading_days": len(open_days)}
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n回测指标")
    print(sum_df.to_string(index=False))
    print(f"\n输出目录: {out_dir}")
    return sum_df, out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ETF 轮动回测")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--split", choices=["dev", "valid", "test", None], default=None,
                    help="样本切分：dev/valid/test，缺省为全区间")
    ap.add_argument("--cost", choices=["zero_cost", "base_cost", "double_cost"],
                    default=None, help="成本敏感性实验")
    ap.add_argument("--list-strategies", default=None,
                    help="逗号分隔，如 E3,E0；缺省全部")
    args = ap.parse_args()
    cfg = load_config(args.config)
    strat = set(args.list_strategies.split(",")) if args.list_strategies else None
    run_all(cfg, split=args.split, cost_name=args.cost, strategies=strat)
