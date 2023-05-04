from pyecharts import Kline, Bar, Overlap
# 画 K 线图
def plot_kline(df,name):
    #画K线图数据
    date = df.index.strftime('%Y%m%d').tolist()
    k_value = df[['open','close', 'low','high']].values
    #引入pyecharts画图使用的是0.5.11版本，新版命令需要重写
    kline = Kline(name+'行情走势')
    kline.add('日K线图', date, k_value,
              is_datazoom_show=True,is_splitline_show=False)
    #成交量
    bar = Bar()
    bar.add('成交量', date, df['volume'],tooltip_tragger='axis',
                is_legend_show=False, is_yaxis_show=False,
                yaxis_max=5*max(df['volume']))
    overlap = Overlap()
    overlap.add(kline)
    overlap.add(bar,yaxis_index=1, is_add_yaxis=True)
    return overlap
#===================================================================
from pyecharts import Line
#
def plot_observe_index(data,v,title,plot_type='line',zoom=False):
    att=data.index
    try:
        attr=att.strftime('%Y%m%d')
    except:
        attr=att
    if plot_type=='line':
        p=Line(title)
        p.add('',attr,list(data[v].round(2)),
         is_symbol_show=False,line_width=2,
        is_datazoom_show=zoom,is_splitline_show=True)
    else:
        p=Bar(title)
        p.add('',attr,[int(i*1000)/10 for i in list(data[v])],
              is_label_show=True,
        is_datazoom_show=zoom,is_splitline_show=True)
    return p
