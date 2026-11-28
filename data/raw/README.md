# 回测输入快照

将五份原始行情快照放在 `data/raw/`。下表列出生成仓库结果时所用的数据。

| 文件 | 内容 |
|---|---|
| `prices_raw.csv` | Tushare `fund_daily`，2015-01-05 至 2025-11-19 |
| `adj_factor.csv` | Tushare `fund_adj`，2015-03-13 至 2025-11-19 |
| `calendar_SSE.csv`、`calendar_SZSE.csv` | Tushare `trade_cal`，2015-01-01 至 2025-11-19 |
| `prices_supplement_2025Q4.csv` | 新浪财经日线，2025-11-20 至 2025-12-31 |

接收快照后可用以下 SHA-256 核对输入版本。

```text
90e46a0fac2f97bf5810d11db3a6a4f19f327513eace7fe36f0d54de2a060fff  adj_factor.csv
5677abf1116f467c3ba7e7ab8363d78c7b7a5333bef5722096a38def8b39519e  calendar_SSE.csv
6500caebc3199ae830e5a01075550a8c4ed9cfc58bb2a55d24261ea8508fff29  calendar_SZSE.csv
d37de6e28324044ef3040bd2a23b3988479cb18cdb299fe0b0faf71bc6f6b727  prices_raw.csv
d732e1d371c33fbf7da6184e9b76440d9dd20fd9d2b1078dd474fa0ec5081a04  prices_supplement_2025Q4.csv
```
