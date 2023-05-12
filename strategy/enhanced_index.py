import time
import pandas as pd
import numpy as np 
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import datetime
import statsmodels.formula.api as smf
import scipy.optimize as optimize
from multiprocessing import Pool


class enhanced_index:
    
    def __init__(self, start_date, end_date, n, freq="monthly", group_one="sfm_randomforest_group_one"):
        '''README
        start_date: 开始日期
        end_date: 结束日期
        freq: 策略频率, daily, monthly
        n: 选择的股票数量
        group_one: 筛选股票情况
        '''
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.n = n
    
        self.month_list = self.get_month_list(type='first')
        self.bond = self.bond_data()
        self.month_index_weight = self.stocks_beta('/home/ubuntu/group_porject/beta.parquet.gzip', self.index_weight('/home/ubuntu/group_porject/raw_data/index_data/000905.XSHG_成分股权重.csv'))
        self.month_stocks_data = self.stocks_month_data('/home/ubuntu/group_porject/raw_data/index_data/000905_成分股日行情.csv')
        self.month_index_price = self.index_data('/home/ubuntu/group_porject/raw_data/index_data/000905.XSHG_指数日行情.csv')
        self.indexBeta = self.index_beta()
        
        self.month_index_ret = (self.month_index_price['close'].pct_change()).rename('index_ret')
        # 引入lasso预测情况
        self.group_one = pd.read_csv(
                "/home/ubuntu/group_porject/raw_data/strategy/%s.csv" % group_one, index_col=0
            )
        self.group_five = pd.read_csv(
                "/home/ubuntu/group_porject/raw_data/strategy/%s_group_five.csv" % group_one[0:16], index_col=0
            )

        
    # 获取每月第一个交易日
    def get_month_list(self, type='last'):
        '''
        start:开始时间
        end:  终止时间
        type: 输入'last'返回每月最后一个交易日，输入'first' 返回每月第一个交易日
        '''
        df = pd.read_csv("/home/ubuntu/group_porject/raw_data/index_data/000001.XSHG指数日行情.csv", index_col=0)
        df['year-month'] = [str(i)[0:7] for i in df.index]
        days = df.drop_duplicates('year-month',type).index
        # trade_days = [datetime.datetime.strftime(i,'%Y-%m-%d') for i in days]
        trade_days = [i for i in days]
        
        return trade_days

    # 股票数据
    def stocks_month_data(self, file_path):

        stocks_daily_price = pd.read_csv(file_path, index_col=0)
        stocks_beta = pd.read_parquet('/home/ubuntu/group_porject/beta.parquet.gzip')
        stocks_beta.index = stocks_beta.index.astype(str)
        stocks_beta = pd.DataFrame(stocks_beta.unstack())
        stocks_beta = stocks_beta.rename(columns={0:'beta'})
        monthly_stocks_data = stocks_daily_price.join(stocks_beta, on=['sec_code', 'datetime'])

        if self.freq == 'monthly':
            monthly_stocks_data = monthly_stocks_data[monthly_stocks_data['datetime'].isin(self.month_list)]
        monthly_stocks_data = monthly_stocks_data.drop(columns=['openinterest'])
    
        # 计算收益率
        def cal_ret(x):
            # x['ret'] = (x['close'] - x['open'].shift()) / x['open'].shift()
            x['ret'] = x['close'].pct_change()
            
            return x

        monthly_stocks_data = monthly_stocks_data.groupby('sec_code').apply(lambda x : cal_ret(x)) 
        
        return monthly_stocks_data
    
    # 债券数据
    def bond_data(self):
        
        bond_data = pd.read_excel('/home/ubuntu/group_porject/raw_data/base/十年期国债即期收益率.xls', index_col=0, header=4)
        bond_data.index.name = 'date'
        bond_data = bond_data.rename(columns={'中债估值中心' : 'rf'}).dropna()
        bond_data.index = bond_data.index.astype(str).str[:10]
        if self.freq == 'monthly':
            bond_data = bond_data[bond_data.index.isin(self.month_list)]
        bond_data['rf'] = bond_data['rf']/100/12 
        
        return bond_data

    # 指数行情数据
    def index_data(self,file_path):
        index_price = pd.read_csv(file_path, index_col=0)
        if self.freq == 'monthly':
            index_price = index_price[index_price.index.isin(self.month_list)].drop(columns=['openinterest'])
        
        return index_price
    
    # 指数成分股权重数据
    def index_weight(self, file_path):
        index_weight = pd.read_csv(file_path, index_col=0)
        if self.freq == 'monthly':
            index_weight = index_weight[index_weight['trade_date'].isin(self.month_list)]
        
        return index_weight
    
    # 指数成分股权重数据拼接beta数据
    def stocks_beta(self, file_path, index_weight):
        stocks_beta = pd.read_parquet(file_path)
        stocks_beta.index = stocks_beta.index.astype(str)
        stocks_beta = pd.DataFrame(stocks_beta.unstack())
        stocks_beta = stocks_beta.rename(columns={0:'beta'})
        
        stocks_beta_weights = index_weight.join(stocks_beta, on=['sec_code', 'trade_date'])
        
        return stocks_beta_weights
    
    # 计算指数beta
    def index_beta(self):
        # 债券日收益率
        bond_data = pd.read_excel('/home/ubuntu/group_porject/raw_data/base/十年期国债即期收益率.xls', index_col=0, header=4)
        bond_data.index.name = 'date'
        bond_data = bond_data.rename(columns={'中债估值中心' : 'rf'}).dropna()
        bond_data.index = bond_data.index.astype(str).str[:10]
        bond_data['rf'] = bond_data['rf']/100/250
        # 指数日收益率
        index_price = pd.read_csv('/home/ubuntu/group_porject/raw_data/index_data/000905.XSHG_指数日行情.csv', index_col=0)
        # index_price['index_ret'] = (index_price['close'] - index_price['open']) / index_price['open']
        index_price['index_ret'] = index_price['close'].pct_change()
        index_ret = index_price.drop(columns=['open', 'close', 'high', 'low', 'volume', 'openinterest'])
        # 市场日收益率
        mkt_data = pd.read_csv('/home/ubuntu/group_porject/raw_data/index_data/000001.XSHG指数日行情.csv', index_col=0)
        # mkt_data['mkt_ret'] = (mkt_data['close'] - mkt_data['open']) / mkt_data['open']
        mkt_data['mkt_ret'] = mkt_data['close'].pct_change()
        reg_data = index_ret.join(bond_data).join(mkt_data['mkt_ret'])
        # reg_data = reg_data.rename(columns={'ret':'index_ret'})
        # 计算指数beta
        reg_data['mkt_excess_ret'] = reg_data['mkt_ret'] - reg_data['rf']
        reg_data['index_excess_ret'] = reg_data['index_ret'] - reg_data['rf']
        
        def regression(x, formula):
            return smf.ols(formula, data=x).fit().params
        
        index_beta = pd.DataFrame(columns={'beta'})
        for i in range(243, len(index_ret)):
            beta = regression(reg_data[i-243:i], 'index_excess_ret ~ mkt_excess_ret').mkt_excess_ret
            date = reg_data.index.tolist()[i]
            new = pd.DataFrame({'beta':beta}, index=[date])
            # index_beta = index_beta.append(new)
            index_beta = pd.concat([index_beta, new])
            
        return index_beta

    # 单期最优化权重计算
    def optimize_portfolio_weight(self, t, n, total_stocks_num, random_seed=None):
        '''
        t : which period
        n : back to t-n period
        total_stocks_num : 股票数量
        '''
        dates = self.month_list[t:t+n]
        # dates = self.month_list[t+24:t+24+n]
        if dates[-1] not in self.group_one["trade_date"].unique().tolist():
            return pd.DataFrame()
        # index_ret = self.month_index_ret[t:t+n]
        index_ret = self.month_index_ret[(self.month_index_ret.index >= dates[0]) & (self.month_index_ret.index <= dates[-1])]
        print('Index weight between date: %s and %s' % (dates[0], dates[-1]))
        
        # stocks_data = self.month_stocks_data[self.month_stocks_data['datetime'].isin(dates)].drop(columns=['open', 'close', 'high', 'low', 'volume'])
        stocks_data = self.month_stocks_data[
                            (self.month_stocks_data['datetime'] >= dates[0]) & (self.month_stocks_data['datetime'] <= dates[-1])
                        ].drop(
                                columns=['open', 'close', 'high', 'low', 'volume']
                            )
        # print(stocks_data)
        stocks_weight = self.month_index_weight[self.month_index_weight['trade_date']==dates[-1]]
        
        # # 预测收益率选股
        limit_stocks_weight = self.get_stocks_in_group_one(
            stocks_weight, dates[-1]
        )

        stock_list = limit_stocks_weight.index.tolist()
        stocks_data = stocks_data[stocks_data['sec_code'].isin(stock_list)]
        # stocks_data = stocks_data.drop_duplicates()
        
        ret_matrix = stocks_data.set_index(['sec_code', 'datetime']).unstack()['ret'].dropna()
        # beta_matrix = stocks_data.set_index(['sec_code', 'datetime']).unstack()['beta']
        # beta_matrix = beta_matrix[beta_matrix.index.isin(ret_matrix.index)]
        limit_stocks_weight = limit_stocks_weight[limit_stocks_weight.index.isin(ret_matrix.index)]
        limit_stocks_weight = limit_stocks_weight.sort_index()
        # print(limit_stocks_weight)
        # 行业权重计算
        test = limit_stocks_weight
        test['行业'] = test['sw_l1']
        test = test.set_index([limit_stocks_weight.index,'sw_l1']).unstack().reset_index().set_index('sec_code')
        industry_matrix = test['行业'].apply(lambda x : x.apply(lambda y : 0 if y is np.NaN else 1))
        industry_weight = stocks_weight.groupby('sw_l1').sum().weight

        # stocks_weight_upper, stocks_weight_boundary = self.adjusted_weight_with_MLmodel(
        #                                                   limit_stocks_weight, 
        #                                                   self.group_one, 
        #                                                   self.group_five,
        #                                                   adjusted_path=0.14
        #                                               )

        stocks_weight_list = limit_stocks_weight.weight.tolist()
        stocks_weight_upper = [i/100*(3)*(500/len(stocks_weight_list)) for i in stocks_weight_list]
        stocks_weight_boundary = [-i/100*(0.3)*(500/len(stocks_weight_list))  for i in stocks_weight_list]
        ### 求解最优化问题
        Q = (5/2) * matrix(np.array(ret_matrix.T.cov()))
        p = - (3/2) * matrix(np.array(ret_matrix.mean(axis=1)))
        # p = matrix(np.zeros((len(ret_matrix), 1)))
        ## 组合权重约束为1
        A = matrix(np.array([[1.0 for _ in range(len(ret_matrix))]]))
        b = matrix(1.0)
        ## 组合权重和beta约束为1
        # A = matrix(np.array([[1.0 for _ in range(len(ret_matrix))], beta_matrix[dates[-1]].tolist()]))
        # b = matrix([1.0, 1.0])
        ## beta约束
        # A = matrix(np.r_[np.array([beta_matrix[dates[-1]].tolist()])])
        # b = matrix([self.indexBeta.loc[dates[-1]].beta])
        ## 行业中性约束
        # A = matrix(np.float64(industry_matrix.T.values))
        # b = matrix([(industry_weight/100).tolist()])
        # 个股权重范围约束
        # G = matrix(
        #         np.vstack(
        #             (np.identity(len(ret_matrix)), -np.identity(len(ret_matrix)))
        #         )
        #     )
        # h = matrix(
        #         np.array(
        #             stocks_weight_upper + stocks_weight_boundary
        #         )
        #     )
        # 跟踪误差约束
        tracking_error_limit = 0.04
        G = matrix(
                np.vstack(
                        (np.identity(len(ret_matrix)), -np.identity(len(ret_matrix)), (ret_matrix - index_ret).T)
            )
        )
        h = matrix(
                np.vstack(
                    np.array(stocks_weight_upper + stocks_weight_boundary + [tracking_error_limit for i in range(0,n)])
                )
        )

        sol = solvers.qp(Q, p, G, h, A, b)
        weights = pd.DataFrame(np.array(sol['x']), index=ret_matrix.index, columns=[dates[-1]], dtype='double')
        
        return weights
    
    # 股票数量限制
    def get_stocks_list_limit(self, df_weight, date, n, by, ascending=False):
        ind_num = len(df_weight[df_weight['trade_date']==date]['sw_l1'].unique().tolist())
        
        def sort_top_n_stocks(df,n):
        
            df = df.sort_values(by=by, ascending=ascending)
            df = df[: int(round(n/500*len(df),0))]
            
            return df
        
        if n == len(df_weight):
            return df_weight
        else:
            return df_weight.groupby('sw_l1').apply(lambda x : sort_top_n_stocks(x, n)).droplevel('sw_l1')
        
    # 选择预测收益率高的股票
    def get_stocks_in_group_one(self, df_weight, date):
        group_one = self.group_one[self.group_one['trade_date'] == date]

        df_weight = df_weight[
                                df_weight.index.isin(group_one.sec_code.tolist())
                    ]
        
        return df_weight

    # 随机股票数量限制
    def get_stocks_list_random(self, df_weight, date, n, random_seed, target_weight=None):
        
        def random_industry_stocks(df,n):
            len_df = len(df)
            df = df.sort_values(by="weight", ascending=False)
            df = df[: (int(round(n/500*len(df),0)) + 4)]
            return df.sample(n=int(round(n/500*len_df, 0)), weights="weight", random_state=random_seed)
        
        if n == len(df_weight):
            return df_weight
        elif target_weight is None:
            return df_weight.groupby('sw_l1').apply(lambda x : random_industry_stocks(x, n)).droplevel('sw_l1')
        else:
            random_key = True
            while random_key:
                result =  df_weight.groupby('sw_l1').apply(lambda x : random_industry_stocks(x, n)).droplevel('sw_l1')
                stocks_in_index_weight = self.cal_strategy_stocks_in_index_weight(result, date)
                print(stocks_in_index_weight)
                if (stocks_in_index_weight < (0.1 + target_weight)) and  (stocks_in_index_weight > (-0.1 + target_weight)):
                    random_key = False
                    return result
            
    # 通过模型预测收益率情况，overweight预期收益率高的股票，underweight预期收益率低的股票
    def adjusted_weight_with_MLmodel(self, limit_stocks_weight, group_one, group_five, adjusted_path=0.01):
        trade_date = limit_stocks_weight.trade_date[0]
        limit_stocks_weight = limit_stocks_weight.reset_index()
        group_one = group_one[group_one['trade_date'] == trade_date]
        group_five = group_five[group_five['trade_date'] == trade_date]

        limit_stocks_weight['weight'] = limit_stocks_weight.apply(
                                            lambda x : x.weight + adjusted_path 
                                            if x.sec_code in group_one.sec_code.tolist() else x.weight, axis=1
                                        )
        limit_stocks_weight['weight'] = limit_stocks_weight.apply(
                                            lambda x : x.weight - adjusted_path 
                                            if x.sec_code in group_five.sec_code.tolist() else x.weight, axis=1
                                        )

        stocks_weight_list = limit_stocks_weight.weight.tolist()
        stocks_weight_upper = [i/100*(3)*(500/len(stocks_weight_list)) for i in stocks_weight_list]
        stocks_weight_boundary = [-i/100*(0.3)*(500/len(stocks_weight_list))  for i in stocks_weight_list]

        return stocks_weight_upper, stocks_weight_boundary

    def cal_strategy_stocks_in_index_weight(self, res_df, date):
        index_stocks_weight = self.month_index_weight[["trade_date", "weight"]]
        res_weight = pd.DataFrame(columns=["trade_date", "stocks weight"])
        
        index_stocks = index_stocks_weight[index_stocks_weight["trade_date"] == date].reset_index()
        stra_stocks = res_df[res_df["trade_date"] == date]
        stocks_in_index_weight = index_stocks.merge(stra_stocks, on=["sec_code"])["weight_x"].sum() / 100
        df = pd.DataFrame([[date, stocks_in_index_weight]], columns=["trade_date", "stocks weight"])
        res_weight = pd.concat([res_weight, df])

        return res_weight['stocks weight'].sum()


    # 多期计算
    def cal_strategy_weight(self, total_period, n, file_name, random_seed=None):

        weights_df = pd.DataFrame()
        for t in range(1, total_period):
            weights = self.optimize_portfolio_weight(t, n=n, total_stocks_num=self.n, random_seed=random_seed)
            weights_df = pd.concat([weights_df, weights.T])
            
        weights_df = weights_df.stack().reset_index()
        weights_df = weights_df.rename(columns={'level_0':'trade_date', 'level_1':'sec_code', 0:'weight'})
        
        weights_df.to_csv('/home/ubuntu/group_porject/raw_data/strategy/mv_TEconstraint_' + file_name + '.csv', index=True)


def run_single_strategy(i):
    track_strategy = enhanced_index('2012-01-01', '2022-01-01', freq='monthly', n=100, by='weight', ascending=False, group_one=i)
    track_strategy.cal_strategy_weight(109, 12, ('%s' % (i)), random_seed=None)


if __name__ == '__main__':
    t0 = time.time()
    pool_ = Pool(3)
    pool_.map(run_single_strategy, [
                                    "randomforest_group_one",
                                    "MLP_group_one",
                                    "elasticnet_group_one",
                                    "adaboost_group_one",
                                    "GBDT_group_one"
                                    "new_minmax_lasso_group_one"
                                    ])
    pool_.close()
    pool_.join()
    # track_strategy = enhanced_index('2012-01-01', '2022-01-01', freq='monthly', n=100, by='weight', ascending=False)
    # track_strategy.cal_strategy_weight(109, 9, 'monthly_test_100', random_seed=None)
    t1 = time.time()
    print("总耗时为: {0}".format(t1 - t0))
    print("完成时间为: %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
