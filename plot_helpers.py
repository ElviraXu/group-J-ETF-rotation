from pathlib import Path
from collections import deque
from matplotlib import font_manager
import matplotlib.pyplot as plt

def configure_font():
    # 显式注册中文字体，避免换电脑后出现方框。
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"),
                  Path("C:/Windows/Fonts/simhei.ttf"),
                  Path("/System/Library/Fonts/PingFang.ttc")]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            break
    else:
        preferred = ["Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei"]
        available = {f.name for f in font_manager.fontManager.ttflist}
        name = next((n for n in preferred if n in available), None)
        if not name:
            raise RuntimeError("未找到中文字体，请先安装 Noto Sans CJK SC 或思源黑体")
    plt.rcParams.update({"font.family": name, "axes.unicode_minus": False,
                         "font.size": 11, "axes.titlesize": 19,
                         "axes.labelsize": 12, "savefig.dpi": 300})

def paired_pnl(trades):
    """FIFO 成本配对；支持多次买卖与部分平仓，手续费按数量分摊。"""
    result, remaining = {}, {}
    for symbol, frame in trades.groupby("symbol", sort=True):
        lots, realized = deque(), 0.0
        for row in frame.itertuples(index=False):
            qty = float(row.units)
            if row.side == "BUY":
                lots.append([qty, float(row.execution_price) + float(row.fee) / qty])
            elif row.side == "SELL":
                left, allocated_cost = qty, 0.0
                if left > sum(lot[0] for lot in lots) + 1e-5:
                    raise ValueError(f"{symbol} 卖出超过已有持仓，可能缺少期初买入记录")
                while left > 1e-7:
                    if not lots:
                        raise ValueError(f"{symbol} 交易数量无法配对")
                    lot = lots[0]
                    matched = min(left, lot[0])
                    allocated_cost += matched * lot[1]
                    lot[0] -= matched
                    left -= matched
                    if lot[0] < 1e-7:
                        lots.popleft()
                realized += qty * row.execution_price - row.fee - allocated_cost
            else:
                raise ValueError(f"未知交易方向：{row.side}")
        result[symbol] = realized
        remaining[symbol] = {"units": sum(x[0] for x in lots),
                             "cost": sum(x[0] * x[1] for x in lots)}
    return result, remaining
