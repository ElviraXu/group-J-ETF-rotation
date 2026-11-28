"""从原始行情快照生成回测所需的 ETF 日线和交易日历。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXCHANGE = {
    '159915.SZ': 'SZSE',
    '510300.SH': 'SSE',
    '510500.SH': 'SSE',
    '518880.SH': 'SSE',
}
SUSPENSIONS = {'159915.SZ': {'20210208'}, '510500.SH': {'20150413', '20150414'}}
KEYS = ['ts_code', 'trade_date']


def prepare_data(raw_dir, output_dir):
    """计算复权价格、校验行情覆盖并返回输出文件路径。"""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    raw = pd.read_csv(raw_dir / 'prices_raw.csv', dtype={'trade_date': str})
    supplement = pd.read_csv(raw_dir / 'prices_supplement_2025Q4.csv', dtype={'trade_date': str})
    factors = pd.read_csv(raw_dir / 'adj_factor.csv', dtype={'trade_date': str})
    quotes = pd.concat([raw, supplement], ignore_index=True).sort_values(KEYS)
    values = quotes[['open', 'close', 'vol']]
    if set(quotes.ts_code) != set(EXCHANGE) or quotes.duplicated(KEYS).any():
        raise ValueError('行情标的或主键异常，请检查原始快照')
    if not np.isfinite(values).all().all() or values.le(0).any().any():
        raise ValueError('行情包含缺失或非正的开盘价、收盘价或成交量')

    quotes = quotes.merge(factors[KEYS + ['adj_factor']], on=KEYS, how='left', validate='one_to_one')
    for symbol, group in quotes.groupby('ts_code'):
        known = group.loc[group.adj_factor.notna(), 'trade_date']
        if known.empty or group.loc[group.trade_date.between(known.min(), known.max()), 'adj_factor'].isna().any():
            raise ValueError(f'{symbol} 复权因子缺失，请检查原始快照')
    quotes['adj_factor'] = quotes.groupby('ts_code').adj_factor.transform(lambda s: s.bfill().ffill())
    if not np.isfinite(quotes.adj_factor).all() or quotes.adj_factor.le(0).any():
        raise ValueError('复权因子必须为有限正数')

    # 510300 于 2015-01-20 每份除息 0.035 元，用前一日收盘价近似补算因子。
    index_etf = quotes.ts_code.eq('510300.SH')
    before = quotes.loc[index_etf & quotes.trade_date.eq('20150119'), 'close'].item()
    after_factor = quotes.loc[index_etf & quotes.trade_date.eq('20150120'), 'adj_factor'].item()
    quotes.loc[index_etf & quotes.trade_date.lt('20150120'), 'adj_factor'] = after_factor / (1 + .035 / before)
    base = quotes.groupby('ts_code').adj_factor.transform('first')
    quotes['open_adj'] = (quotes.open * quotes.adj_factor / base).round(6)
    quotes['close_adj'] = (quotes.close * quotes.adj_factor / base).round(6)
    quotes['date'] = pd.to_datetime(quotes.trade_date, format='%Y%m%d').dt.strftime('%Y-%m-%d')
    quotes = quotes.rename(columns={'ts_code': 'symbol', 'open': 'open_raw', 'close': 'close_raw', 'vol': 'volume'})
    columns = ['date', 'symbol', 'open_raw', 'close_raw', 'open_adj', 'close_adj', 'volume']

    supplement_days = set(supplement.trade_date)
    calendars = {}
    for exchange in ('SSE', 'SZSE'):
        calendar = pd.read_csv(raw_dir / f'calendar_{exchange}.csv', dtype={'cal_date': str, 'pretrade_date': str})
        calendar = calendar.sort_values('cal_date').assign(source='tushare')
        tail = pd.date_range(calendar.cal_date.iloc[-1], '2025-12-31', inclusive='right')
        # 两所 2025 年休市通知确认这段期间除周末外照常开市。
        if set(tail[tail.weekday < 5].strftime('%Y%m%d')) != supplement_days:
            raise ValueError('补充行情的日期与 2025 年末的沪深交易日不一致')
        previous = calendar.loc[calendar.is_open.eq(1), 'cal_date'].iloc[-1]
        rows = []
        for date in tail.strftime('%Y%m%d'):
            is_open = int(date in supplement_days)
            rows.append((exchange, date, is_open, previous, 'sina_inferred'))
            if is_open:
                previous = date
        calendars[exchange] = pd.concat([calendar, pd.DataFrame(rows, columns=calendar.columns)], ignore_index=True)

    for symbol, group in quotes.groupby('symbol'):
        calendar = calendars[EXCHANGE[symbol]]
        expected = set(calendar.loc[calendar.is_open.eq(1), 'cal_date']) - SUSPENSIONS.get(symbol, set())
        dates = set(group.date.str.replace('-', '', regex=False))
        if dates != expected:
            raise ValueError(f'{symbol} 行情日期异常，缺 {sorted(expected - dates)}，多 {sorted(dates - expected)}')

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {'prices_csv': output_dir / 'prices_clean.csv'}
    quotes[columns].to_csv(files['prices_csv'], index=False, encoding='utf-8-sig')
    for exchange, calendar in calendars.items():
        file = output_dir / f'calendar_{exchange}.csv'
        calendar.to_csv(file, index=False, encoding='utf-8-sig')
        files[f'calendar_{exchange.lower()}_csv'] = file
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed')
    args = parser.parse_args()
    for path in prepare_data(args.raw_dir, args.output_dir).values():
        print(path)
