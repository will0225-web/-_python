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
model = 'Long_Short_Model_篩選(還未替除Cum_VP)'

model_name_combination=['Long','Short']
confindence_combination=[0.8]
for model_choice in model_name_combination:    
    for confindence in confindence_combination:
        
        model_name=f'{model_choice}_correlatcion_win_rate_{confindence}'
        combinations = ['','_max_loss','_更改持有期間止盈止損','_max_loss_更改持有期間止盈止損',]
        total_profit=0.5
        sharpe=0.2
        drawdown=0.2
        commom_ratio=0.1
        # 創建一個空的 DataFrame，用於儲存所有的結果

        final_file_path = fr'C:\\Users\\User\\Documents\\Backtesting_修改出場邏輯後\\{model}\\All_Combined_Summary_{model_name}_{total_profit}_{sharpe}_{drawdown}_{commom_ratio}.xlsx'

        # 創建一個空的字典來儲存所有組合的 min 和 max 值
        min_max_values = {
            'total_profit': {'min': float('inf'), 'max': float('-inf')},
            'sharpe_ratio': {'min': float('inf'), 'max': float('-inf')},
            'max_drawdown': {'min': float('inf'), 'max': float('-inf')},
            'common_ratio': {'min': float('inf'), 'max': float('-inf')}
        }

        # 先遍歷所有組合，找出每個指標的 min 和 max 值
        for name in combinations:
            base_file_path = fr'C:\Users\User\Documents\Backtesting_修改出場邏輯後\{model}\Long_Short_model\Backtest_result_{model_name}{name}'
            file_path = os.path.join(
                base_file_path,
                f'A_Summary_{model_name}{name}.csv')

            strategy_data = pd.read_csv(file_path)

            # 更新 min 和 max 值
            min_max_values['total_profit']['min'] = min(min_max_values['total_profit']['min'], strategy_data['Total Profit'].min())
            min_max_values['total_profit']['max'] = max(min_max_values['total_profit']['max'], strategy_data['Total Profit'].max())
            min_max_values['sharpe_ratio']['min'] = min(min_max_values['sharpe_ratio']['min'], strategy_data['Strategy Sharpe Ratio'].min())
            min_max_values['sharpe_ratio']['max'] = max(min_max_values['sharpe_ratio']['max'], strategy_data['Strategy Sharpe Ratio'].max())
            min_max_values['max_drawdown']['min'] = min(min_max_values['max_drawdown']['min'], abs(strategy_data['Strategy Max Drawdown']).min())
            min_max_values['max_drawdown']['max'] = max(min_max_values['max_drawdown']['max'], abs(strategy_data['Strategy Max Drawdown']).max())
            strategy_data['common_ratio'] = strategy_data.apply(
                lambda row: abs(row['Strategy Sharpe Ratio'] / row['Strategy Max Drawdown']) if row['Strategy Max Drawdown'] != 0 else 1,
                axis=1
            )
            min_max_values['common_ratio']['min'] = min(min_max_values['common_ratio']['min'], strategy_data['common_ratio'].min())
            min_max_values['common_ratio']['max'] = max(min_max_values['common_ratio']['max'], strategy_data['common_ratio'].max())

        # 打印所有組合的最終 min 和 max 值
        print("Final min and max values for all combinations:")
        for key, value in min_max_values.items():
            print(f"{key}: min = {value['min']}, max = {value['max']}")


        with pd.ExcelWriter(final_file_path, engine='openpyxl') as writer:
            for name in combinations:
                base_file_path = fr'C:\\Users\\User\\Documents\\Backtesting_修改出場邏輯後\\{model}\\Long_Short_model\\Backtest_result_{model_name}{name}'
                file_path = os.path.join(
                                            base_file_path,
                                            f'A_Summary_{model_name}{name}.csv')

                strategy_data= pd.read_csv(file_path)
                print(strategy_data)


                def calculate_strategy_score(strategy_data, w1=total_profit, w2=sharpe, w3=drawdown, w4=commom_ratio):
                    """
                    計算策略的綜合評分
                    :param strategy_data: 包含指標的 DataFrame (total_profit, sharpe_ratio, max_drawdown, common_ratio)
                    :param w1: 總利潤權重
                    :param w2: Sharpe 比率權重
                    :param w3: 最大回撤權重（負的影響）
                    :param w4: 通用比率權重
                    :return: 帶有評分的 DataFrame
                    """
                    # # 使用找到的 min 和 max 值進行標準化
                    profit_min = min_max_values['total_profit']['min']
                    profit_max = min_max_values['total_profit']['max']
                    sharpe_min = min_max_values['sharpe_ratio']['min']
                    sharpe_max = min_max_values['sharpe_ratio']['max']
                    drawdown_min = 0
                    drawdown_max = min_max_values['max_drawdown']['max']
                    common_min = min_max_values['common_ratio']['min']
                    common_max = min_max_values['common_ratio']['max']
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
                    # 如果 min/max 未設定，則使用 DataFrame 的最大最小值
                    profit_min = strategy_data['total_profit'].min() if profit_min is None else profit_min
                    profit_max = strategy_data['total_profit'].max() if profit_max is None else profit_max
                    sharpe_min = strategy_data['sharpe_ratio'].min() if sharpe_min is None else sharpe_min
                    sharpe_max = strategy_data['sharpe_ratio'].max() if sharpe_max is None else sharpe_max
                    drawdown_min = abs(strategy_data['max_drawdown']).min() if drawdown_min is None else drawdown_min
                    print(drawdown_min)
                    drawdown_max = abs(strategy_data['max_drawdown']).max() if drawdown_max is None else drawdown_max
                    common_min = strategy_data['common_ratio'].min() if common_min is None else common_min
                    common_max = strategy_data['common_ratio'].max() if common_max is None else common_max




                    # 標準化過程
                    strategy_data['profit_norm'] = (strategy_data['total_profit'] - profit_min) / (profit_max - profit_min + epsilon)
                    strategy_data['sharpe_norm'] = (strategy_data['sharpe_ratio'] - sharpe_min) / (sharpe_max - sharpe_min + epsilon)
                    strategy_data['drawdown_norm'] = 1 - (abs(strategy_data['max_drawdown']) - drawdown_min) / (drawdown_max - drawdown_min + epsilon)
                    strategy_data['common_norm'] = (strategy_data['common_ratio'] - common_min) / (common_max - common_min + epsilon)


                    # strategy_data['profit_norm'] = (strategy_data['total_profit'] - strategy_data['total_profit'].min()) / (strategy_data['total_profit'].max() - strategy_data['total_profit'].min()+epsilon)
                    # strategy_data['sharpe_norm'] = (strategy_data['sharpe_ratio'] - strategy_data['sharpe_ratio'].min()) / (strategy_data['sharpe_ratio'].max() - strategy_data['sharpe_ratio'].min()+epsilon)
                    # strategy_data['drawdown_norm'] = 1 - (abs(strategy_data['max_drawdown']) - abs(strategy_data['max_drawdown']).min()) / (abs(strategy_data['max_drawdown']).max() - abs(strategy_data['max_drawdown']).min() + epsilon)
                    # strategy_data['common_norm'] = (strategy_data['common_ratio'] - strategy_data['common_ratio'].min()) / (strategy_data['common_ratio'].max() - strategy_data['common_ratio'].min() + epsilon)
                    # 加權計算綜合評分
                    strategy_data['score'] = (w1 * strategy_data['profit_norm'] +
                                            w2 * strategy_data['sharpe_norm'] +
                                            w3 * strategy_data['drawdown_norm'] +
                                            w4 * strategy_data['common_norm'])

                    return strategy_data

                
                df = calculate_strategy_score(strategy_data)
                # 定義重要欄位順序
                important_columns = ['predict_lookahead', 'thresholds', 'ETP_threshold', 'SL','Max position','total_profit','sharpe_ratio','max_drawdown','common_ratio',
                                    'score','Number of Winning Trades','Number of Losing Trades','profit_norm','sharpe_norm','drawdown_norm','common_norm']
                # 將其餘欄位保留原來的順序
                remaining_columns = [col for col in df.columns if col not in important_columns]

                # 重新排列欄位順序
                new_column_order = important_columns + remaining_columns
                df_reordered = df[new_column_order]
                # 按照 'score' 欄位降序排列
                df_sorted = df_reordered.sort_values(by='score', ascending=False)
                # 將當前組合的結果寫入不同的 worksheet
                sheet_name = f'Summary{name}' if name else 'Summary'
                df_sorted.to_excel(writer, sheet_name=sheet_name, index=False)


                # base_file_path = fr'C:\\Users\\User\\Documents\\Backtesting_修改出場邏輯後\\{model}'
                # result_name_up =f'Summary_{model_name}{name}.csv'
                # full_file_path_up = os.path.join(base_file_path, result_name_up) 
                # result.to_csv(full_file_path_up, index=False)