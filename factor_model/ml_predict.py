import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import os
import datetime
import time
from multiprocessing import Pool
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
# from jqdatasdk import *


class ml_predict:

    @staticmethod
    def factor_selection(factor_data, ret, start_factors, predict_factor):
        """using LASSO to select useful factors from a starting factors pool

        Args:
            factor_data (dataframe): training data, X
            ret (dataframe): traing target, y
            start_factors (list): starting factors pool
            predict_factor (dataframe): predict data

        Returns:
            dataframe: traing data and predict data with only selected factors
        """
        pd.set_option('display.float_format',  '{:,.20f}'.format) 

        lasso = linear_model.LassoCV(
                    eps=0.001, 
                    n_alphas=100, 
                    alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                    cv=TimeSeriesSplit
                )
        lasso.fit(factor_data, ret)
        coef_df = pd.DataFrame(lasso.coef_, index=start_factors, columns=["coef"])
        coef_df = coef_df[coef_df.coef != 0]

        selected_factors = list(coef_df.T.columns)
        factor_data = factor_data[selected_factors]
        predict_factor = predict_factor[selected_factors]

        return factor_data, predict_factor

    @staticmethod
    def minmax_normalization(df_input):
        return df_input.apply(lambda x: (x-x.min())/ (x.max() - x.min()), axis=0)

    @staticmethod
    def predict_next_periodd_ret(factor_data, predict_factor, ret, date, method):
        """using different machine learning model to predict next period stock's return

        Args:
            factor_data (dataframe ): training data, X
            predict_factor (dataframe): predict data
            ret (dataframe): targeting data, y
            date (str): date string
            method (str): lasso, elasticnet, randomforest, adaboost, MLP, GBDT

        Returns:
            list: a list of stock quintiles
        """
        # 初始因子
        all_factors = factor_data.columns
        # 缺失数据
        factor_data = factor_data.fillna(factor_data.mean())
        # MINMAX标准化
        factor_data = ml_predict.minmax_normalization(factor_data)
        ret = ret.fillna(0)
        # 因子筛选
        factor_data, predict_factor = ml_predict.factor_selection(factor_data, ret, all_factors, predict_factor)
        print(predict_factor.shape)

        if method == 'lasso':
            ml_reg = linear_model.LassoCV(
                    eps=0.001, 
                    n_alphas=100, 
                    alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                    cv=TimeSeriesSplit
                )
            ml_reg.fit(factor_data, ret)
        elif method == 'elasticnet':
            ml_reg = linear_model.ElasticNetCV(
                    l1_ratio=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], 
                    eps=0.001, 
                    n_alphas=100, 
                    alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                    cv=TimeSeriesSplit
                )
            ml_reg.fit(factor_data, ret)
        elif method == 'randomforest':
            ml_reg = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=0
                )
            ml_reg.fit(factor_data, ret)
        elif method == 'adaboost':
            ml_reg = AdaBoostRegressor(
                    base_estimator=DecisionTreeRegressor(max_depth=3),
                    n_estimators=100,
                    learning_rate=0.8,
                    loss='linear'
                )
            ml_reg.fit(factor_data, ret)
        elif method == 'MLP':
            ml_reg = MLPRegressor(
                    hidden_layer_sizes=(100, 100, ),
                    activation='relu',
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    max_iter=5000,
                    random_state=1
                )
            ml_reg.fit(factor_data, ret)
        elif method == 'GBDT':
            ml_reg = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.8,
                    subsample=0.9
                )
            ml_reg.fit(factor_data, ret)

        # 回归系数
        # coef = pd.DataFrame(ml_reg.coef_, index=all_factors, columns=[date]).T

        # 预测下一期
        predict_factor = predict_factor.fillna(predict_factor.mean())
        predict_factor = predict_factor.fillna(0)
        result_index = predict_factor.index
        # predict_factor = ml_predict.minmax_normalization(predict_factor)
        predict_ret = ml_reg.predict(predict_factor)
        result = pd.DataFrame(predict_ret, index=result_index, columns=['ret'])
        # 按照收益率分五组
        result = result.sort_values(by='ret', ascending=False)
        stocks_result = result.index.tolist()
        five_group_ = [stocks_result[i:i+100] for i in range(0, len(stocks_result), 100)]

        return five_group_

    @staticmethod
    def predict_long_period(train_length, method):
        trade_dates_list = pd.read_csv('/home/ubuntu/group_porject/raw_data/base/stock_ret.csv').datetime.unique().tolist()
        trade_dates_list = sorted(trade_dates_list, reverse=False)
        index_weight = pd.read_csv(
                        '/home/ubuntu/group_porject/raw_data/index_data/000905.XSHG_成分股权重.csv', index_col=0
                )
        group_one = pd.DataFrame()
        group_two = pd.DataFrame()
        group_three = pd.DataFrame()
        group_four = pd.DataFrame()
        group_five = pd.DataFrame()
        factor_select = pd.DataFrame()
        for i in range(train_length, len(trade_dates_list)-2):
            train_data = pd.DataFrame()
            for j in range(i-train_length, i):
                month_data = pd.read_csv(
                            '/home/ubuntu/group_porject/raw_data/base/jqfactor/%s.csv' % trade_dates_list[j], index_col=0
                    )
                train_data = pd.concat([train_data, month_data])

            month_stocks = index_weight[index_weight['trade_date'] == trade_dates_list[i+1]].index.tolist()
            train_data = train_data[train_data.index.isin(month_stocks)]
            train_factor = train_data.iloc[:,:-2]
            train_ret = train_data['ret']
            
            # 训练集
            # train_factor = train_factor[train_factor.index.isin(month_stocks)]
            # train_ret = train_ret[train_ret.index.isin(month_stocks)]
            # print(train_ret)
            # 预测集
            predict_factor = pd.read_csv(
                            '/home/ubuntu/group_porject/raw_data/base/jqfactor/%s.csv' % trade_dates_list[i+1], index_col=0
                    ).iloc[:,:-2]
            print(trade_dates_list[i+1])
            predict_factor = predict_factor[predict_factor.index.isin(month_stocks)]
            # 预测，排序，分组
            five_groups_stocks = ml_predict.predict_next_periodd_ret(
                                            train_factor, predict_factor, train_ret, trade_dates_list[i+1], method
                                        )

            # 特征筛选情况
            # factor_select = pd.concat([factor_select, coef_df])
            # factor_select.to_csv('/home/ubuntu/group_porject/raw_data/result/lasso_factor_Select.csv')
            # 分组情况
            df_1 = pd.DataFrame(five_groups_stocks[0], columns=['sec_code'])
            df_1['trade_date'] = trade_dates_list[i+1]
            df_1['weight'] = 1/len(df_1)

            df_2 = pd.DataFrame(five_groups_stocks[1], columns=['sec_code'])
            df_2['trade_date'] = trade_dates_list[i+1]
            df_2['weight'] = 1/len(df_1)

            df_3 = pd.DataFrame(five_groups_stocks[2], columns=['sec_code'])
            df_3['trade_date'] = trade_dates_list[i+1]
            df_3['weight'] = 1/len(df_1)

            df_4 = pd.DataFrame(five_groups_stocks[3], columns=['sec_code'])
            df_4['trade_date'] = trade_dates_list[i+1]
            df_4['weight'] = 1/len(df_1)

            df_5 = pd.DataFrame(five_groups_stocks[4], columns=['sec_code'])
            df_5['trade_date'] = trade_dates_list[i+1]
            df_5['weight'] = 1/len(df_1)

            group_one = pd.concat([group_one, df_1])
            group_two = pd.concat([group_two, df_2])
            group_three = pd.concat([group_three, df_3])
            group_four = pd.concat([group_four, df_4])
            group_five = pd.concat([group_five, df_5])
        
        # 分组保存
        group_one.to_csv('/home/ubuntu/group_porject/raw_data/strategy/%s_group_one.csv' % method)
        group_two.to_csv('/home/ubuntu/group_porject/raw_data/strategy/%s_group_two.csv' % method)
        group_three.to_csv('/home/ubuntu/group_porject/raw_data/strategy/%s_group_three.csv' % method)
        group_four.to_csv('/home/ubuntu/group_porject/raw_data/strategy/%s_group_four.csv' % method)
        group_five.to_csv('/home/ubuntu/group_porject/raw_data/strategy/%s_group_five.csv' % method)
        # 特征筛选情况保存
        # factor_select.to_csv('/home/ubuntu/group_porject/raw_data/result/minmax_lasso_factor_Select.csv')


def run_single_strategy(i):
    ml_task = ml_predict.predict_long_period(train_length=36, method=i)

if __name__=='__main__':
    # ml_predict.predict_long_period(train_length=36, method="adaboost")
    t0 = time.time()
    pool_ = Pool(3)
    pool_.map(run_single_strategy, [
                                    # "randomforest", 'elasticnet', "MLP", "adaboost", "GBDT"
                                    "lasso"
                                    ])
    pool_.close()
    pool_.join()
    # track_strategy = full_replication('2012-01-01', '2022-01-01', freq='monthly', n=100, by='weight', ascending=False)
    # track_strategy.cal_strategy_weight(109, 9, 'monthly_test_100', random_seed=None)
    t1 = time.time()
    print("总耗时为: {0}".format(t1 - t0))
    print("完成时间为: %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
