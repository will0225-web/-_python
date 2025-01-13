import sys
import os
import csv
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。r
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
from math import *
import random
import openpyxl
model = 'Long_Short_model_剔除Cum只用Volume_'
model_name='Long_correlatcion_win_rate_0.8_train_1723_test_2324_96_480'
combinations = ['','_max_loss','_更改持有期間止盈止損','_max_loss_更改持有期間止盈止損',]
# 創建一個空的 DataFrame，用於儲存所有的結果
final_file_path = fr'C:\\Users\\User\\Documents\\Backtesting_修改出場邏輯後\\{model}\\All_Combined_Summary_{model_name}_沒有設定minmax.xlsx'


with pd.ExcelWriter(final_file_path, engine='openpyxl') as writer:
    for name in combinations:
        base_file_path = fr'C:\\Users\\User\\Documents\\Backtesting_修改出場邏輯後\\{model}\\Long_model\\Backtest_result_{model_name}{name}'
        file_path = os.path.join(
                                    base_file_path,
                                    f'A_Summary_{model_name}{name}.csv')

        strategy_data= pd.read_csv(file_path)
        print(strategy_data)


        def calculate_strategy_score(strategy_data, w1=0.4, w2=0.3, w3=0.1, w4=0.2):
            """
            計算策略的綜合評分
            :param strategy_data: 包含指標的 DataFrame (total_profit, sharpe_ratio, max_drawdown, common_ratio)
            :param w1: 總利潤權重
            :param w2: Sharpe 比率權重
            :param w3: 最大回撤權重（負的影響）
            :param w4: 通用比率權重
            :return: 帶有評分的 DataFrame
            """
            # 標準化指標以確保其值在 0-1 之間
            epsilon = 1e-6
            strategy_data['total_profit'] = strategy_data['Total Profit'] 
            strategy_data['sharpe_ratio'] = strategy_data['Strategy Sharpe Ratio']
            strategy_data['max_drawdown'] = strategy_data['Strategy Max Drawdown']
            # 使用 apply 方法逐行計算 common_ratio
            strategy_data['common_ratio'] = strategy_data.apply(
                lambda row: abs(row['sharpe_ratio'] / row['max_drawdown']) if row['max_drawdown'] != 0 else 1,
                axis=1
            )
            strategy_data['profit_norm'] = (strategy_data['total_profit'] - strategy_data['total_profit'].min()) / (strategy_data['total_profit'].max() - strategy_data['total_profit'].min()+epsilon)
            strategy_data['sharpe_norm'] = (strategy_data['sharpe_ratio'] - strategy_data['sharpe_ratio'].min()) / (strategy_data['sharpe_ratio'].max() - strategy_data['sharpe_ratio'].min()+epsilon)
            strategy_data['drawdown_norm'] = 1 - (abs(strategy_data['max_drawdown']) - abs(strategy_data['max_drawdown']).min()) / (abs(strategy_data['max_drawdown']).max() - abs(strategy_data['max_drawdown']).min() + epsilon)
            strategy_data['common_norm'] = (strategy_data['common_ratio'] - strategy_data['common_ratio'].min()) / (strategy_data['common_ratio'].max() - strategy_data['common_ratio'].min() + epsilon)
            #加權計算綜合評分
            strategy_data['score'] = (w1 * strategy_data['profit_norm'] +
                                    w2 * strategy_data['sharpe_norm'] +
                                    w3 * strategy_data['drawdown_norm'] +
                                    w4 * strategy_data['common_norm'])

            return strategy_data

        
        
        result = calculate_strategy_score(strategy_data)
        print(result)
        # 將當前組合的結果寫入不同的 worksheet
        sheet_name = f'Summary{name}' if name else 'Summary'
        result.to_excel(writer, sheet_name=sheet_name, index=False)



    # base_file_path = fr'C:\\Users\\User\\Documents\\Backtesting_修改出場邏輯後\\{model}'
    # result_name_up =f'Summary_{model_name}{name}.csv'
    # full_file_path_up = os.path.join(base_file_path, result_name_up) 
    # result.to_csv(full_file_path_up, index=False)