"""根据 60 日动量和 120 日均线生成月末信号。"""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SIGNAL_COLUMNS = ['date', 'symbol', 'close_adj', 'mom60', 'ma120', 'eligible', 'year_month']


def generate_signals(prices_csv, output_dir):
    """输出每月指标明细和 E1、E2、E3 的月末目标。"""
    prices = pd.read_csv(prices_csv, parse_dates=['date']).sort_values(['symbol', 'date'])
    close = prices.groupby('symbol').close_adj
    prices['mom60'] = close.pct_change(60, fill_method=None)
    prices['ma120'] = close.transform(lambda series: series.rolling(120).mean())
    prices['eligible'] = prices.mom60.gt(0) & prices.close_adj.gt(prices.ma120)
    prices['year_month'] = prices.date.dt.to_period('M')
    monthly = prices.groupby(['symbol', 'year_month']).tail(1)
    monthly = monthly.sort_values(['year_month', 'mom60'], ascending=[True, False])

    targets = []
    for _, month in monthly.groupby('year_month'):
        positive = month.loc[month.mom60.gt(0), 'symbol']
        eligible = month.loc[month.eligible, 'symbol']
        targets.append({
            'signal_date': month.date.max(),
            'E1_target': month.symbol.iloc[0],
            'E2_target': 'CASH' if positive.empty else positive.iloc[0],
            'E3_target': 'CASH' if eligible.empty else eligible.iloc[0],
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly[SIGNAL_COLUMNS].to_csv(output_dir / 'signals_monthly.csv', index=False)
    targets_path = output_dir / 'targets.csv'
    pd.DataFrame(targets).to_csv(targets_path, index=False)
    return targets_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=ROOT / 'data/processed/prices_clean.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/signals')
    args = parser.parse_args()
    print(generate_signals(args.prices, args.output_dir))
