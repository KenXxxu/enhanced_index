# ==================================================
import backtrader as bt
import pandas as pd
import datetime
import time
from multiprocessing import Pool


class MultifactorStrategy(bt.Strategy):
    # 策略参数
    params = dict(trade_info=pd.DataFrame(),)# 传一个 trade_info 表
    def log(self, txt, dt=None):
        ''' Logging function for this strategy '''
        dt = dt or self.datas[0].datetime.date(0)
        print('{}, {}'.format(dt.isoformat(), txt))

    '''选股策略'''

    def __init__(self):
        self.buy_stock = self.p.trade_info  # 保留调仓列表
        # 读取调仓日期，即每月的最后一个交易日，回测时，会在这一天下单，然后在下一个交易日，以开盘价买入
        self.trade_dates = pd.to_datetime(self.buy_stock['trade_date'].unique()).tolist()
        self.order_list = []  # 记录以往订单，方便调仓日对未完成订单做处理
        self.buy_stocks_pre = []  # 记录上一期持仓

    def next(self):
        dt = self.datas[0].datetime.date(0)  # 获取当前的回测时间点
        print('dt', dt)
        # 如果是调仓日，则进行调仓操作
        if dt in self.trade_dates:
            print("-----------{} 为调仓日----------".format(dt))
            # 在调仓之前，取消之前所下的没成交也未到期的订单
            if len(self.order_list) > 0:
                for od in self.order_list:
                    self.cancel(od)  # 如果订单未完成，则撤销订单，不会影响已经完成的订单
                self.order_list = []  # 重置订单列表
            # 提取当前调仓日的持仓离列表
            buy_stocks_data = self.buy_stock.query(f"trade_date=='{dt}'")
            long_list = buy_stocks_data['sec_code'].tolist()
            print('long_list', long_list)  # 打印持仓列表
            # 对现有持仓中，调仓后不再继续持有的股票进行卖出平仓
            sell_stock = [i for i in self.buy_stocks_pre if i not in long_list]
            print('sell_stock', sell_stock)  # 打印平仓列表
            if len(sell_stock) > 0:
                print('--------对不再持有的股票进行平仓-----------')
                for stock in sell_stock:
                    data = self.getdatabyname(stock)
                    if self.getposition(data).size > 0:
                        od = self.close(data=data)  # 进行平仓
                        self.order_list.append(od)  # 记录卖出订单
            # 买入此次调仓的股票，多退少补原则
            print('-----------买入此次调仓期的股票----------------')
            # total_value = self.broker.getvalue() # 获得当前持仓价值
            for stock in long_list:
                w = buy_stocks_data.query(f"sec_code=='{stock}'")['weight'].iloc[0]  # 提取持仓权重
                data = self.getdatabyname(stock)
                # buypercentage = w
                # targetvalue = buypercentage * total_value # 得到目标市值
                print('stock', stock)
                # print('data.open[1]:',data.open[1])
                # size = int(abs(targetvalue / data.open[1] // 100 * 100)) # 按次日开盘价计算下单量，下单量是100的整数倍
                order = self.order_target_percent(data=data, target=w * 0.95)
                # order = self.order_target_size(data=data,target=size)
                # 为了减少可用资金不足的情况，留5%的现金做备用
                self.order_list.append(order)
            self.buy_stocks_pre = long_list  # 保存此次调仓的股票列表

    # 打印订单信息：
    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, ref:%.0f, Price:%.2f, Cost:%.2f,Comm%.2f,Size:%.2f,Stock:%s' %
                    (order.ref,  # 订单编号
                     order.executed.price,  # 成交价
                     order.executed.value,  # 成交额
                     order.executed.comm,  # 佣金
                     order.executed.size,  # 成交量
                     order.data._name))  # 股票名称
            else:  # sell
                self.log('SELL EXECUTED, ref:%.0f,Price:%.2f,Cost:%.2f,Comm%.2f,Size:%.2f,Stock:%s' %
                         (order.ref,
                          order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.size,
                          order.data._name))
# ===========================================================
def MomTT(strategy,para):
    strategy = strategy  # 将类赋予类，strategy 是类，而不是类对象
    # 初始化 cerebro 回测系统设置
    cerebro = bt.Cerebro()
    # datafeed: 按股票代码，依次循环传入数据
    daily_price = para['daily_price']

    for stock in daily_price['sec_code'].unique():
        # 日期对齐
        data = pd.DataFrame(index=daily_price['datetime'].unique())  # 获取回测区间内所有交易日
        df = daily_price.query(f"sec_code=='{stock}'")[['datetime', 'open', 'high', 'low', 'close',
                                                        'volume', 'openinterest']]
        df['aaa'] = df['datetime']
        df = df.set_index('aaa')
        data_ = pd.merge(data, df, left_index=True, right_index=True, how='left')
        # 缺失值处理: 日期对齐时，会使得有些交易日的数据为空，所以需要对缺失数据进行填充
        data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
        data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].fillna(
            method='pad')
        data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].fillna(0)

        # data_ = data_[(data_.index<=datetime.datetime(2019,12,27)) & (data_.index>=datetime.datetime(2018,1,4))]

        # 将开盘价等于 0 的对应的那行数据进行前填充，因为backtrader 是按第二天的开盘价
        # 导入数据
        datafeed = bt.feeds.PandasData(dataname = data_,
                                       fromdate = para['start_date'],
                                       todate = para['end_date'])  # 通过这里可以控制回测的时间区间
        cerebro.adddata(datafeed, name=stock)  # 通过 name 实现数据集与股票的一一对应
        print(f"{stock} Done !")
    # ==========================================
    # ==========================================
    # 初始资金 100, 000, 000
    cerebro.broker.setcash(para['startcash'])
    # 佣金，双边各 0.0003
    cerebro.broker.setcommission(commission=para['commission'])
    # 滑点：双边各 0.0001
    cerebro.broker.set_slippage_perc(perc=para['slippage'])
    # ============================================
    # 将编写的策略添加给大脑：
    cerebro.addstrategy(strategy=strategy, trade_info=para['trade_info'])
    # ==================================
    # 加入 analyzer 分析模块
    
    # 修改by Ken，用来调取该路径下的py文件 Analyzer_Module_multifactor
    import sys 
    sys.path.append(r'lib\back_test') 
    
    import Analyzer_Module_multifactor as ANL

    cerebro = ANL.analysis(cerebro=cerebro)
    # =================================
    # 启动回测
    result = cerebro.run()
    # 获取回测结束后的总资金
    portvalue = cerebro.broker.getvalue()
    pnl = portvalue - para['startcash']
    # 打印结果
    print(f'总资金: {round(portvalue, 2)}')
    print(f'净收益: {round(pnl, 2)}')
    return result

#======================================================================================
def run_strategy(para):
    # 修改by Ken，用来调取该路径下的py文件 Analyzer_Module_multifactor
    import sys 
    sys.path.append(r'lib\back_test') 
    
    import Analyzer_Module_multifactor as ANL
    
    result = MomTT(strategy=MultifactorStrategy, para=para)
    df_list = ANL.index_calculation(result)  # 计算策略表现指标的值
    df00 = df_list[0]
    # df0 = df_list[1]
    df1 = df_list[1]
    df2 = df_list[2]
    df3 = df_list[3]
    df4 = df_list[4]
    df = {}
    df['df00'] = df00
    # df['df0'] = df0
    df['df1'] = df1
    df['df2'] = df2
    df['df3'] = df3
    df['df4'] = df4
    # ====================================
    # 从返回的 result 中提取回测结果
    strat = result[0]
    # 返回日度收益率序列
    daily_return = pd.Series(strat.analyzers.timereturn.get_analysis())
    pyfoliozer = result[0].analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    pyfoliozer_data = {}
    pyfoliozer_data['returns'] = returns
    pyfoliozer_data['positions'] = positions
    pyfoliozer_data['transactions'] = transactions
    pyfoliozer_data['gross_lev'] = gross_lev
    # 打印评价指标
    print('--------------AnnualReturn------------------')
    print(strat.analyzers.annualreturn.get_analysis())
    print('--------------SharpeRatio-------------------')
    print(strat.analyzers.SharpeRatio.get_analysis())
    print('--------------DrawDown----------------------')
    print(strat.analyzers.DW.get_analysis())
    # 可视化回测结果
    # cerebro.plot()
    # =======================================================================
    # 将分析结果保存到本地文件中。
    from datatrans import save_pr as save
    from datatrans import mkdir
    data_path = para['save_path']
    mkdir(data_path)
    data_exp = para['save_filename']
    save(data_path, data_exp + '.pr', (df, daily_return,pyfoliozer_data,para),
         ('analyze_result','daily_return','pyfoliozer','para'))

# 标准化输入函数
def standardiaze_input(daily_price,trade_info,start_date,end_date,startcash,commission,slippage,save_path,save_filename):
    """
    daliy_price:日线价格输入文件路径
    trade_info:仓位信息文件路径
    start_date:开始日期
    end_date:结束日期
    startcash:初始资金
    commission:手续费
    slippage:回测滑点比例
    save_path:保存输出路径
    save_filename:输出文件名
    """

    para = {}
    # 对齐数据格式
    daily_price = pd.read_csv(daily_price, parse_dates=['datetime'])
    trade_info = pd.read_csv(trade_info, parse_dates=['trade_date'])
    start_date =  datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    #
    
    para['daily_price'] = daily_price[['datetime', 'sec_code', 'open', 'high', 'low', 'close', 'volume', 'openinterest']]
    para['trade_info'] = trade_info[['trade_date', 'sec_code', 'weight']]
    para['start_date'] = start_date
    para['end_date'] = end_date
    para['startcash'] = startcash
    para['commission'] = commission
    para['slippage'] = slippage
    para['save_path'] = save_path
    para['save_filename'] = save_filename
    
    return para


def run_single_task(i):
    daily_price = pd.read_csv('/home/ubuntu/group_porject/raw_data/index_data/000905_成分股日行情.csv', parse_dates=['datetime'])
    daily_price = daily_price[['datetime', 'sec_code', 'open', 'high', 'low', 'close', 'volume', 'openinterest']]
    trade_info = pd.read_csv(
        "/home/ubuntu/group_porject/raw_data/strategy/%s.csv" % (i),
        parse_dates=['trade_date']
    )
    trade_info = trade_info[['trade_date', 'sec_code', 'weight']]
    para = {}
    para['daily_price'] = daily_price
    para['trade_info'] = trade_info
    para['start_date'] = datetime.datetime(2013, 1, 4) # 回测开始日期
    para['end_date'] = datetime.datetime(2021, 12, 31) # 回测结束日期
    para['startcash'] = 100000000.0
    para['commission'] = 0.0003
    para['slippage'] = 0.0001
    para['save_path'] = '/home/ubuntu/group_porject/raw_data/result'
    para['save_filename'] = '%s' % (i)
    run_strategy(para=para)


if __name__ == '__main__':
    # daily_price = pd.read_csv(r'./raw_data/index_data/000905_成分股日行情.csv', parse_dates=['datetime'])
    # daily_price = daily_price[['datetime', 'sec_code', 'open', 'high', 'low', 'close', 'volume', 'openinterest']]
    # trade_info = pd.read_csv(r"./raw_data/strategy/minimize_tracking_error_pre100_weight_random.csv", parse_dates=['trade_date'])
    # trade_info = trade_info[['trade_date', 'sec_code', 'weight']]
    # para = {}
    # para['daily_price'] = daily_price
    # para['trade_info'] = trade_info
    # para['start_date'] = datetime.datetime(2013, 1, 4) # 回测开始日期
    # para['end_date'] = datetime.datetime(2021, 12, 31) # 回测结束日期
    # para['startcash'] = 100000000.0
    # para['commission'] = 0.0003
    # para['slippage'] = 0.0001
    # para['save_path'] = '.\\raw_data\\result\\'
    # para['save_filename'] = 'Analyze_result_multifactor_pre100_weight_random'
    # run_strategy(para=para)
    t0 = time.time()
    pool_ = Pool(2)
    pool_.map(run_single_task, [
                "mv_TEconstraint_randomforest_group_one",
                "mv_TEconstraint_elasticnet_group_one",
                "mv_TEconstraint_adaboost_group_one",
                "mv_TEconstraint_GBDT_group_one",
                "mv_TEconstraint_MLP_group_one",
                "mv_TEconstraint_new_minmax_lasso_group_one"
            ]
        )
    pool_.close()
    pool_.join()
    t1 = time.time()
    print("总耗时为: {0}".format(t1 - t0))
    print("完成时间为: %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
