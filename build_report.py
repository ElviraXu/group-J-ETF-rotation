from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

from plot_helpers import configure_font, paired_pnl
from main import load_config, run_all
from signals import generate_signals
from prepare_data import prepare_data

P=Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='运行 ETF 轮动回测并生成图表')
parser.add_argument('--output-dir', type=Path, default=P/'report')
parser.add_argument('--raw-dir', type=Path, default=P/'data/raw')
args = parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
configure_font()

inputs = OUT / '回测输入'
data_files = prepare_data(args.raw_dir, inputs / 'processed')
data_files['targets_csv'] = generate_signals(data_files['prices_csv'], inputs / 'signals')

def replay(folder, split=None, cost=None, strategies=None):
 config = load_config('config.json')
 config['data'].update({name: str(path) for name, path in data_files.items()})
 config['data']['results_dir'] = str(OUT/'回测明细'/folder)
 _, run_dir = run_all(config, split=split, cost_name=cost, strategies=strategies)
 return run_dir

FULL = replay('全样本_基础成本')
ZERO = replay('全样本_零成本', cost='zero_cost', strategies={'E3'})
DOUBLE = replay('全样本_双倍成本', cost='double_cost', strategies={'E3'})
SPLITS = {'开发期': replay('开发期', split='dev'),
          '验证期': replay('验证期', split='valid'),
          '测试期': replay('测试期', split='test')}
cfg=json.loads((FULL/'run_meta.json').read_text())['config']
names={'E0':'E0 等权基准','E1':'E1 纯动量','E2':'E2 正动量','E3':'E3 主策略','BUYHOLD':'沪深300 ETF 买入持有'}
colors={'E3':'#2466AC','E0':'#D7952E','BUYHOLD':'#648A65'}
def read(p,col='date'):
 d=pd.read_csv(p,encoding='utf-8-sig',parse_dates=[col]);return d.sort_values(col,kind='stable').reset_index(drop=True)
navs={k:read(FULL/f'{k}_nav_daily.csv') for k in names}
for k,d in navs.items():
 assert not d.date.duplicated().any() and np.isfinite(d.nav).all() and d.nav.gt(0).all()
 assert d.date.equals(navs['E3'].date)
 assert np.allclose(d.nav,d.total_assets/cfg['initial_cash'],atol=1e-9,rtol=0)
prices=read(data_files['prices_csv'])
trades=read(FULL/'E3_trades.csv','trade_date')
hold=read(FULL/'E3_holdings_daily.csv')
assert not prices.duplicated(['date','symbol']).any()
assert hold.date.equals(navs['E3'].date)
assert np.allclose(hold.filter(regex='^value_').sum(axis=1)+hold.cash,hold.total_assets,atol=.01)
assert np.allclose(hold.total_assets,navs['E3'].total_assets,atol=.01)
audit=[]

def fig(title,subtitle,shape=(12,6.4)):
 f,a=plt.subplots(figsize=shape);f.subplots_adjust(left=.10,right=.96,bottom=.17,top=.80)
 f.text(.10,.92,title,size=19,weight='bold',color='#172B42');f.text(.10,.85,subtitle,size=10,color='#586878')
 style(a);return f,a
def style(a):
 a.spines[['top','right']].set_visible(False);a.grid(axis='y',color='#DFE5EA',lw=.7);a.set_axisbelow(True)
def dateaxis(a,start='2016-01-01',end='2025-12-31'):
 a.set_xlim(pd.Timestamp(start),pd.Timestamp(end));a.xaxis.set_major_locator(mdates.YearLocator(1));a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));a.set_xlabel('日期')
def save(f,name):
 f.savefig(OUT/name,dpi=300,facecolor='white');plt.close(f)
period='正式回测 2016—2025；基础成本：单边手续费 0.03%，滑点 0.02%'

f,a=fig('全周期净值对比图',period)
for k in ['E3','E0','BUYHOLD']:
 d=navs[k]
 times=[d.date.iloc[0]+pd.Timedelta(hours=9,minutes=30),*(d.date+pd.Timedelta(hours=15))]
 a.plot(times,np.r_[1.,d.nav],label=names[k],color=colors[k],lw=1.7)
a.axvspan(pd.Timestamp('2015-01-01'),pd.Timestamp('2016-01-01'),color='#EDF0F3')
a.text(pd.Timestamp('2015-07-01'),.96,'2015\n指标预热',transform=a.get_xaxis_transform(),ha='center',va='top',size=9,color='#6B7785')
a.set_ylabel('累计净值（初始资金 = 1）');dateaxis(a,'2015-01-01','2025-12-31 23:59');a.legend(frameon=False,loc='upper left',bbox_to_anchor=(.12,1))
key_switches = [
    ('2024-10-08', '卖黄金 ETF\n买创业板 ETF', (-12, -22)),
    ('2025-12-01', '卖创业板 ETF\n买黄金 ETF', (-16, 36)),
]
for date, label, offset in key_switches:
    day = pd.Timestamp(date)
    assert set(trades.loc[trades.trade_date.eq(day), 'side']) == {'BUY', 'SELL'}
    y = navs['E3'].loc[navs['E3'].date.eq(day), 'nav'].item()
    point = (day + pd.Timedelta(hours=15), y)
    a.scatter(*point, color=colors['E3'], edgecolor='white', marker='D', s=35, zorder=5)
    a.annotate(f'{date}\n{label}', point, xytext=offset, textcoords='offset points',
               ha='right', va='top', fontsize=8, color=colors['E3'], zorder=6,
               bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 2},
               arrowprops={'arrowstyle': '-', 'color': colors['E3'], 'lw': .8})
f.text(.10,.04,'2015 年用于指标预热。菱形标记 E3 换仓日的收盘净值。买入持有采用 510300.SH ETF。',size=9,color='#586878')
save(f,'1_全周期净值对比.png')

f,axs=plt.subplots(2,1,figsize=(12,8),sharex=True);f.subplots_adjust(left=.10,right=.96,bottom=.12,top=.84,hspace=.30)
f.suptitle('全周期回撤对比图',x=.10,y=.96,ha='left',size=19,weight='bold',color='#172B42')
dds=[]
for a,k in zip(axs,['E3','E0']):
 d=navs[k];dd=d.nav/d.nav.cummax().clip(lower=1)-1;idx=dd.idxmin();style(a)
 a.fill_between(d.date,dd,0,color=colors[k],alpha=.30);a.plot(d.date,dd,color=colors[k],lw=.7)
 a.scatter(d.date[idx],dd[idx],color=colors[k],s=24,zorder=5)
 a.text(.015,.06,f'最大回撤 {-dd[idx]:.2%}，{d.date[idx]:%Y-%m-%d}',transform=a.transAxes,color=colors[k],fontsize=10)
 a.set_title(names[k],loc='left',size=12);a.yaxis.set_major_formatter(PercentFormatter(1));a.set_ylabel('回撤率');dds.append(float(dd.min()))
for a in axs:a.set_ylim(min(dds)*1.14,.015)
dateaxis(axs[-1]);f.text(.10,.025,'回撤 = 当日净值 / 截至当日历史最高净值 - 1。两个子图使用相同纵轴范围。',size=9,color='#586878')
save(f,'2_全周期回撤对比.png')

annual={}
for k in ['E3','E0','BUYHOLD']:
 ends=navs[k].groupby(navs[k].date.dt.year).nav.last();previous=ends.shift(1);previous.iloc[0]=1.
 annual[k]=ends/previous-1
ann=pd.DataFrame(annual);ann.index.name='year';ann.to_csv(OUT/'年度收益明细.csv',encoding='utf-8-sig')
f,a=fig('年度收益对比图','首年以初始本金为基数；以后各年以此前一年的期末净值为基数')
x=np.arange(len(ann));w=.24
for j,k in enumerate(annual):a.bar(x+(j-1)*w,ann[k],width=w,color=colors[k],label=names[k])
a.set_xticks(x,ann.index);a.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));a.set_ylabel('年度收益率');a.set_xlabel('年份');a.axhline(0,color='#71808E',lw=.7);a.legend(frameon=False,ncol=3,fontsize=10,loc='upper left');a.margins(y=.20)
save(f,'3_年度收益柱状图.png')

f,axs=plt.subplots(2,2,figsize=(14,9));f.subplots_adjust(left=.075,right=.96,bottom=.12,top=.84,wspace=.20,hspace=.30)
f.suptitle('四只 ETF 价格与 E3 交易点',x=.075,y=.96,ha='left',size=20,weight='bold',color='#172B42')
f.text(.075,.90,'曲线为复权收盘价，标记为计入滑点的复权成交价',size=10,color='#586878')
for a,sym in zip(axs.flat,['159915.SZ','510300.SH','510500.SH','518880.SH']):
 p=prices.loc[prices.symbol.eq(sym)&prices.date.ge('2016-01-01')];style(a);a.plot(p.date,p.close_adj,color='#5B7490',lw=1)
 a.set_title(sym,loc='left',size=13);a.set_ylabel('调整价格');dateaxis(a)
 for side,color,marker,label in [('BUY','#D13F3F','^','买入'),('SELL','#24865B','v','卖出')]:
  t=trades.loc[trades.symbol.eq(sym)&trades.side.eq(side)]
  check=t.merge(p[['date']],left_on='trade_date',right_on='date',how='left',validate='many_to_one');assert check.date.notna().all()
  a.scatter(t.trade_date,t.execution_price,color=color,marker=marker,s=28,label=label,zorder=5,edgecolors='white',linewidths=.25)
 a.legend(frameon=False,fontsize=9,ncol=2)
 if not trades.symbol.eq(sym).any():a.text(.03,.88,'本策略在该 ETF 上无成交',transform=a.transAxes,fontsize=9,color='#667788')
save(f,'4_交易点图.png')

f,a=fig('E3 持仓时间线','按每日实际持仓绘制；空仓区间留白；连续持仓包含期间休市日')
symbols=['159915.SZ','510300.SH','510500.SH','518880.SH'];palette=['#467FB3','#71A898','#BA9764','#BDA33D']
hd=mdates.date2num(hold.date);boundary=np.r_[hd,hd[-1]+1]
for j,s in enumerate(symbols):
 mask=hold[f'units_{s}'].gt(1e-7).to_numpy();starts=np.where(mask&~np.r_[False,mask[:-1]])[0];ends=np.where(mask&~np.r_[mask[1:],False])[0]
 a.broken_barh([(boundary[i],boundary[e+1]-boundary[i]) for i,e in zip(starts,ends)],(j-.32,.64),facecolors=palette[j])
a.set_yticks(range(4),symbols);a.set_ylim(-.7,3.7);a.set_ylabel('ETF 代码');dateaxis(a)
cash_days=int(hold.filter(regex='^units_').abs().sum(axis=1).lt(1e-7).sum())
f.text(.10,.04,f'全样本共 {len(hold)} 个交易日，其中 {cash_days} 个交易日收盘时持有现金。',size=10,color='#586878')
save(f,'5_持仓时间线.png')

realized,remaining=paired_pnl(trades)
rows=[]
for s in symbols:
 rem=remaining.get(s,{'units':0.,'cost':0.});last=hold.iloc[-1]
 assert np.isclose(rem['units'],last[f'units_{s}'],atol=1e-5,rtol=1e-9)
 cashflow=(trades.loc[(trades.symbol==s)&(trades.side=='SELL')].eval('execution_price * units - fee').sum()
          -trades.loc[(trades.symbol==s)&(trades.side=='BUY')].eval('execution_price * units + fee').sum())
 value=float(last[f'value_{s}']);total=cashflow+value
 rows.append({'symbol':s,'realized_pnl':realized.get(s,0.),'unrealized_pnl':value-rem['cost'],'net_trading_cashflow':cashflow,'ending_market_value':value,'total_pnl':total})
contrib=pd.DataFrame(rows);contrib.to_csv(OUT/'持仓贡献明细.csv',index=False,encoding='utf-8-sig')
gap=float(contrib.total_pnl.sum()-(navs['E3'].total_assets.iloc[-1]-cfg['initial_cash']))
assert abs(gap)<1.,gap
f,a=fig('各 ETF 累计持仓贡献','总盈亏 = 卖出净收入 - 买入含费成本 + 期末持仓市值；未平仓持仓按期末市值计入')
v=contrib.total_pnl;bars=a.bar(contrib.symbol,v,color=['#CC4141' if x>=0 else '#25885B' for x in v],width=.55)
a.axhline(0,color='#71808E',lw=.7);a.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'));a.set_ylabel('累计总盈亏（元）');a.set_xlabel('ETF 代码');a.margins(y=.24)
for b,val in zip(bars,v):a.annotate(f'{val:+,.2f}',(b.get_x()+b.get_width()/2,val),xytext=(0,7 if val>=0 else -9),textcoords='offset points',ha='center',va='bottom' if val>=0 else 'top',size=10)
f.text(.10,.04,'红色表示盈利，绿色表示亏损；已实现与未实现盈亏分别保存在持仓贡献明细.csv。',size=10,color='#586878')
save(f,'6_各ETF持仓贡献.png')

# 指标以初始本金为 1，补入首日收益（含建仓成本）；不把首日收盘净值作为本金。
def metrics(d):
 values=d.total_assets.to_numpy()/cfg['initial_cash'];series=pd.Series(np.r_[1.,values]);ret=series.pct_change().dropna();sd=ret.std(ddof=1)
 return {'期末净值':float(values[-1]),'累计收益率':float(values[-1]-1),'年化收益率':float(values[-1]**(252/len(values))-1),'最大回撤':float(-(series/series.cummax()-1).min()),'夏普比率':float(ret.mean()/sd*np.sqrt(252)) if sd>0 else np.nan}
targets=read(data_files['targets_csv'],'signal_date')
table=[]
start_audit=[]
for split, run in {'全样本':FULL, **SPLITS}.items():
 meta=json.loads((run/'run_meta.json').read_text(encoding='utf-8'))['config']
 daily={k:read(run/f'{k}_nav_daily.csv') for k in names}
 first=daily['E3'].date.iloc[0]
 prior=targets.loc[targets.signal_date.lt(first)].iloc[-1]
 for k in names:
  d=daily[k]
  assert d.date.equals(daily['E3'].date), (split,k,'日历')
  assert np.allclose(d.holdings_value+d.cash,d.total_assets,atol=.000002,rtol=0)
  assert d.cash.ge(-1e-6).all()
  fills=read(run/f'{k}_trades.csv','trade_date')
  opening=fills.loc[fills.trade_date.eq(first)]
  if k in ('E1','E2','E3'):
   target=prior[f'{k}_target']
   if target=='CASH':
    assert opening.empty
   else:
    assert len(opening)==1 and opening.symbol.iloc[0]==target and opening.side.iloc[0]=='BUY'
    assert pd.Timestamp(opening.signal_date.iloc[0])==prior.signal_date
  else:
   assert len(opening)==(len(cfg['symbols']) if k=='E0' else 1)
   target='四只ETF等权' if k=='E0' else cfg['benchmark_symbol']
  opening_symbols='、'.join(opening.symbol.unique()) if not opening.empty else '现金'
  start_audit.append({'样本区间':split,'策略':k,'首个交易日':first.date(),'月末信号日期':prior.signal_date.date() if k in ('E1','E2','E3') else '', '首日目标':target,'首日买入':opening_symbols,'首日手续费':opening.fee.sum(),'首日收盘净值':d.nav.iloc[0]})
  table.append({'样本区间':split,'策略':k,'起始日期':meta['start_date'],'结束日期':meta['end_date'],'来源':str(run.relative_to(OUT)),**metrics(d)})
pd.DataFrame(start_audit).to_csv(OUT/'样本起点核对.csv',index=False,encoding='utf-8-sig')
t1=pd.DataFrame(table);t1.to_csv(OUT/'表1_核心指标汇总.csv',index=False,encoding='utf-8-sig')
cost=[]
for label,r,fee,slip in [('零成本',ZERO,0.,0.),('基础成本',FULL,.0003,.0002),('双倍成本',DOUBLE,.0006,.0004)]:
 m=json.loads((r/'run_meta.json').read_text());assert m['config']['fee_rate']==fee and m['config']['slippage']==slip
 d=read(r/'E3_nav_daily.csv');assert d.date.equals(navs['E3'].date)
 tt=read(r/'E3_trades.csv','trade_date');assert np.allclose(tt.fee,tt.units*tt.execution_price*fee,atol=.02,rtol=0)
 assert tt[['trade_date','side','symbol']].equals(trades[['trade_date','side','symbol']]),label
 cost.append({'策略':'E3','成本情景':label,'单边手续费率':fee,'单边滑点':slip,'来源':str(r.relative_to(OUT)),**metrics(d)})
t2=pd.DataFrame(cost);t2.to_csv(OUT/'表2_成本敏感性.csv',index=False,encoding='utf-8-sig')

audit.extend([
 'Tushare 提供截至 2025-11-19 的日线和交易日历。2025-11-20 至 12-31 的 30 个交易日使用新浪财经日线补齐，每只 ETF 30 条。交易日历和复权因子的处理方法见 README。',
 '回测输入目录保存本次使用的清洗行情、交易日历和月末信号。',
 '回测使用完整交易日历推导信号的次一交易日，以成交日判断是否纳入样本。E0 与沪深300 ETF 买入持有在样本首日建仓，E0 此后在月首再平衡。策略由上一月末信号决定首日买入资产或继续持有现金。',
 '全区间、开发期、验证期、测试期分别从10万元和净值1启动，不延续上一段持仓。零成本、基础成本、双倍成本均按相同样本起点和 E3 信号重新运行。',
 '全区间图实际曲线为2016—2025，2015仅灰色标记预热。年度收益第一年以初始本金为基数，以后采用上年期末净值。',
 '指标以初始净值 1 起算，日收益包含首日盈亏。年化按每年 252 个交易日换算。夏普比率使用样本标准差，无风险利率取 0。',
 '样本起点核对表记录每段的月末信号、首日成交和费用。完整成交可查阅逐笔成交表。',
 '图 6 的总盈亏计入期末持仓市值。已实现盈亏使用先进先出法配对。',
 f'持仓贡献对账户总盈利核对差额 {gap:.6f} 元，来自逐笔数量与价格小数截断。',
 '分段结果按各区间单独起算。严格样本外评估还需记录参数确定的时间。',
 '图表分辨率为 300 DPI。结果目录的 CSV 使用小数存储收益率和回撤率。'
])
(OUT/'数据来源与核算说明.txt').write_text('\n\n'.join(audit),encoding='utf-8-sig')
print(t1.to_string(index=False));print(t2.to_string(index=False));print(f'持仓贡献对账差额 {gap:.6f} 元')
