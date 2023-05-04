import backtrader as bt
def analysis(cerebro):
#    cerebro.addanalyzer(bt.analyzers.TotalValue, _name='totalvalue')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annualreturn')
    cerebro.addanalyzer(bt.analyzers.Calmar, _name='calmar')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name='timedrawdown')
    cerebro.addanalyzer(bt.analyzers.GrossLeverage, _name='grossleverage')
    cerebro.addanalyzer(bt.analyzers.PositionsValue, _name='positionsvalue')
    cerebro.addanalyzer(bt.analyzers.LogReturnsRolling, _name='logreturnsrolling')
    cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='periodstats')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharperatio')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharperatio_A')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DW')
    cerebro.addanalyzer(bt.analyzers.PyFolio,_name='pyfolio')

    #cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')  # 返回收益率时序数据
    #cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')  # 年化收益率
    #cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')  # 夏普比率
    #cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')  # 回撤
    return cerebro
#=====================================================================================
def index_calculation(results):
    # Calmar比率
    calmar = list(results[0].analyzers.calmar.get_analysis().values())[-1]
    # 平均回撤与最大回撤
    drawdown_info = results[0].analyzers.drawdown.get_analysis()
    average_drawdown_len = drawdown_info['len']
    average_drawdown_rate = drawdown_info['drawdown']
    average_drawdown_money = drawdown_info['moneydown']
    max_drawdown_len = drawdown_info['max']['len']
    max_drawdown_rate = drawdown_info['max']['drawdown']
    max_drawdown_money = drawdown_info['max']['moneydown']
    # 期间统计指标
    PeriodStats_info = results[0].analyzers.periodstats.get_analysis()
    average_rate = PeriodStats_info['average']
    stddev_rate = PeriodStats_info['stddev']
    positive_year = PeriodStats_info['positive']
    negative_year = PeriodStats_info['negative']
    nochange_year = PeriodStats_info['nochange']
    best_year = PeriodStats_info['best']
    worst_year = PeriodStats_info['worst']
    # SQN指标
    SQN_info = results[0].analyzers.sqn.get_analysis()
    sqn_ratio = SQN_info['sqn']
    # VWR指标：波动率加权收益率
    VWR_info = results[0].analyzers.vwr.get_analysis()
    vwr_ratio = VWR_info['vwr']
    # 夏普比率
    sharpe_info = results[0].analyzers.sharperatio.get_analysis()
    # sharpe_info=results[0].analyzers._SharpeRatio_A.get_analysis()
    sharpe_ratio = sharpe_info['sharperatio']
    print('sharpe_info',sharpe_info)
    print('sharpe_ratio', sharpe_ratio)

    # 将上述业绩评价指标放在同一个字典里
    from collections import OrderedDict
    performance = OrderedDict()
    # 字典赋值
    calmar_ratio = calmar
    performance['calmar%'] = round(calmar_ratio * 100, 4)
    performance['average_drawdown_len'] = round(average_drawdown_len)
    performance['average_drawdown_rate'] = round(average_drawdown_rate, 2)
    performance['average_drawdown_money'] = round(average_drawdown_money, 2)
    performance['max_drawdown_len'] = round(max_drawdown_len)
    performance['max_drawdown_rate'] = round(max_drawdown_rate, 2)
    performance['max_drawdown_money'] = round(max_drawdown_money, 2)
    performance['average_rate%'] = round(average_rate * 100, 2)
    performance['stddev_rate%'] = round(stddev_rate * 100, 2)
    performance['positive_year'] = round(positive_year)
    performance['negative_year'] = round(negative_year)
    performance['nochange_year'] = round(nochange_year)
    performance['best_year%'] = round(best_year * 100, 2)
    performance['worst_year%'] = round(worst_year * 100, 2)
    performance['sqn_ratio'] = round(sqn_ratio, 2)
    performance['vwr_ratio'] = round(vwr_ratio, 2)
    print('sharpe_ratio',sharpe_ratio)
    #performance['sharpe_info'] = round(sharpe_ratio, 2)
    performance['sharpe_info'] = sharpe_ratio # 回测时间不超过一年的话 sharpe_ratio 显示为 None
    performance['omega'] = 0

    # 普通交易指标与多空交易指标
    trade_1 = OrderedDict()
    trade_2 = OrderedDict()
    trade_info = results[0].analyzers.tradeanalyzer.get_analysis()
    # 普通交易指标的字典赋值
    trade_1['total_trade_num'] = trade_info['total']['total']
    trade_1['total_trade_opened'] = trade_info['total']['open']
    #trade_1['total_trade_closed'] = trade_info['total']['closed'] # multifactor 策略中 不会返回这个指标
    #trade_1['total_trade_len'] = trade_info['len']['total']       # multifactor 策略中 不会返回这个指标
    #trade_1['long_trade_len'] = trade_info['len']['long']['total']
    #trade_1['short_trade_len'] = trade_info['len']['short']['total']
    #trade_1['longest_win_num'] = trade_info['streak']['won']['longest']
    #trade_1['longest_lost_num'] = trade_info['streak']['lost']['longest']
    #trade_1['net_total_pnl'] = round(trade_info['pnl']['net']['total'], 2)
    #trade_1['net_average_pnl'] = round(trade_info['pnl']['net']['average'], 2)
    #trade_1['win_num'] = trade_info['won']['total']
    #trade_1['win_total_pnl'] = round(trade_info['won']['pnl']['total'], 2)
    #trade_1['win_average_pnl'] = round(trade_info['won']['pnl']['average'], 2)
    #trade_1['win_max_pnl'] = round(trade_info['won']['pnl']['max'], 2)
    #trade_1['lost_num'] = trade_info['lost']['total']
    #trade_1['lost_total_pnl'] = round(trade_info['lost']['pnl']['total'], 2)
    #trade_1['lost_average_pnl'] = round(trade_info['lost']['pnl']['average'], 2)
    #trade_1['lost_max_pnl'] = round(trade_info['lost']['pnl']['max'], 2)

    # 多空交易指标字典赋值
    '''
    trade_2['long_num'] = trade_info['long']['total']
    trade_2['long_win_num'] = trade_info['long']['won']
    trade_2['long_lost_num'] = trade_info['long']['lost']
    trade_2['long_total_pnl'] = round(trade_info['long']['pnl']['total'], 2)
    trade_2['long_average_pnl'] = round(trade_info['long']['pnl']['average'], 2)
    trade_2['long_win_total_pnl'] = round(trade_info['long']['pnl']['won']['total'], 2)
    trade_2['long_win_max_pnl'] = round(trade_info['long']['pnl']['won']['max'], 2)
    trade_2['long_lost_total_pnl'] = round(trade_info['long']['pnl']['lost']['total'], 2)
    trade_2['long_lost_max_pnl'] = round(trade_info['long']['pnl']['lost']['max'], 2)
    trade_2['short_num'] = trade_info['short']['total']
    trade_2['short_win_num'] = trade_info['short']['won']
    trade_2['short_lost_num'] = trade_info['short']['lost']
    trade_2['short_total_pnl'] = trade_info['short']['pnl']['total']
    trade_2['short_average_pnl'] = trade_info['short']['pnl']['average']
    trade_2['short_win_total_pnl'] = trade_info['short']['pnl']['won']['total']
    trade_2['short_win_max_pnl'] = trade_info['short']['pnl']['won']['max']
    trade_2['short_lost_total_pnl'] = trade_info['short']['pnl']['lost']['total']
    trade_2['short_lost_max_pnl'] = trade_info['short']['pnl']['lost']['max']
    '''
    import pandas as pd
    df = pd.DataFrame()
    df['绩效指标名称'] = performance.keys()
    df['绩效指标值'] = performance.values()
    #df['交易指标名称'] = trade_1.keys()
    #df['交易指标值'] = trade_1.values()
    #df['多空指标名称'] = trade_2.keys()
    #df['多空指标值'] = trade_2.values()


    # 账户收益率
#   df0 = df1 = pd.DataFrame([results[0].analyzers.totalvalue.get_analysis()]).T
#   df0.columns = ['total_value']

    # 总的杠杆
    df1 = pd.DataFrame([results[0].analyzers.grossleverage.get_analysis()]).T
    df1.columns = ['GrossLeverage']

    # 滚动的对数收益率
    df2 = pd.DataFrame([results[0].analyzers.logreturnsrolling.get_analysis()]).T
    df2.columns = ['log_return']

    # year_rate
    df3 = pd.DataFrame([results[0].analyzers.annualreturn.get_analysis()]).T
    df3.columns = ['year_rate']

    # 总的持仓价值
    df4 = pd.DataFrame(results[0].analyzers.positionsvalue.get_analysis()).T
    df4['total_position_value'] = df4.sum(axis=1)
    return [df,df1,df2,df3,df4]

#========================================================================================
# 用来制作格式化输出表格的表格
# import dash_html_components as html
def create_table(df, max_rows=18):
    """基于dataframe，设置表格格式"""

    table = html.Table(
        # Header
        [
            html.Tr(
                [
                    html.Th(col) for col in df.columns
                ]
            )
        ] +
        # Body
        [
            html.Tr(
                [
                    html.Td(
                        df.iloc[i][col]
                    ) for col in df.columns
                ]
            ) for i in range(min(len(df), max_rows))
        ]
    )
    return table
#====================================================================================
def appout(app,df,df0,df1,df2,df3,df4,strategy_name,colors):
    import plotly.graph_objs as go
    import dash
    import dash_core_components as dcc  # 交互式组件
    import dash_html_components as html  # 代码转html
    from dash.dependencies import Input, Output  # 回调
    app.layout = html.Div(
        style=dict(backgroundColor=colors['background']),
        children=[
            html.H1(
                children='{}的策略评估结果'.format(strategy_name),
                style=dict(textAlign='center', color=colors['text'])),
            dcc.Graph(
                id='账户价值',
                figure=dict(
                    data=[{'x': list(df0.index), 'y': list(df0.total_value),
                           # 'text':[int(i*1000)/10 for i in list(df3.year_rate)],
                           'type': 'scatter', 'name': '账户价值',
                           'textposition': "outside"}],
                    layout=dict(
                        title='账户价值',
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font=dict(color=colors['text'],
                                  )
                    )
                )
            ),

            dcc.Graph(
                id='持仓市值',
                figure=dict(
                    data=[{'x': list(df4.index), 'y': list(df4.total_position_value),
                           # 'text':[int(i*1000)/10 for i in list(df3.year_rate)],
                           'type': 'scatter', 'name': '持仓市值',
                           'textposition': "outside"}],
                    layout=dict(
                        title='持仓市值',
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font=dict(color=colors['text']),
                    )
                )
            ),
            dcc.Graph(
                id='年化收益',
                figure=dict(
                    data=[{'x': list(df3.index), 'y': list(df3.year_rate),
                           'text': [int(i * 1000) / 10 for i in list(df3.year_rate)],
                           'type': 'bar', 'name': '年收益率',
                           'textposition': "outside"}],
                    layout=dict(
                        title='年化收益率',
                        plot_bgcolor=colors['background'],
                        paper_bgcolor=colors['background'],
                        font=dict(color=colors['text']),
                    )
                )
            ),
            create_table(df)

        ]
    )
    return