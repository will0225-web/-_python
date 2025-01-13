import sys
import os
import csv
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
import itertools
# Define the ranges for the combinations
predict_lookaheads = [168]
thresholds = [0.01]
ETP_thresholds = [0.01]
stop_losses = [ -0.2]
max_positions = [20]


# Generate Multiple Combinations using tuples
Long_combinations = list(itertools.product(predict_lookaheads, thresholds, ETP_thresholds, stop_losses, max_positions))


predict_lookaheads = [120]
thresholds = [0.01]
ETP_thresholds = [0.01]
stop_losses = [ -0.1]
max_positions = [10]

Short_combinations = list(itertools.product(predict_lookaheads, thresholds, ETP_thresholds, stop_losses, max_positions))
model='Long_Short_Model_添加Volume_log'

name=''
date='1108'
long_model_name= f'Long_correlatcion_win_rate_0.8_train_1723_test_2324_96_480_{date}{name}'
base_long_file_path = fr'C:\\Users\\User\\Documents\\Spot_策略資料_{date}\\Backtest_result_{long_model_name}\\Daily_profit_invest'
Summary_long_file_path =  fr'C:\\Users\\User\\Documents\\Spot_策略資料_{date}\\Backtest_result_{long_model_name}'

# base_long_file_path = fr'C:\\Users\\User\\Documents\\Backtesting_修改出場邏輯後\\{model}\\Backtest_result_{long_model_name}\\Daily_profit_invest'
# Summary_long_file_path =  fr'C:\\Users\\User\\Documents\\Backtesting_修改出場邏輯後\\{model}\\Backtest_result_{long_model_name}'

name=''
short_model_name = f'Short_correlatcion_win_rate_0.8_train_1723_test_2324_96_480_{date}{name}'
base_short_file_path= fr'C:\Users\User\Documents\Futures_策略資料_{date}\\Backtest_result_{short_model_name}\Daily_profit_invest'
Summary_short_file_path =  fr'C:\Users\User\Documents\\Futures_策略資料_{date}\\Backtest_result_{short_model_name}'

# base_short_file_path= fr'C:\Users\User\Documents\Backtesting_修改出場邏輯後\\{model}\\Backtest_result_{short_model_name}\Daily_profit_invest'
# Summary_short_file_path =  fr'C:\Users\User\Documents\Backtesting_修改出場邏輯後\\{model}\\Backtest_result_{short_model_name}'

#folder_path = fr'C:\Users\User\Documents\Backtesting_修改出場邏輯後\{model}'
folder_path = fr'C:\Users\User\Documents'



sub_folder_name = f'最佳組合'
full_folder_path  = os.path.join(folder_path, sub_folder_name)
if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)


long_model_combinations = [
    {
        'predict_lookaheads': comb[0],
        'thresholds': comb[1],
        'ETP_thresholds': comb[2],
        'stop_losses': comb[3],
        'max_positions': comb[4]
    }
    for comb in Long_combinations
]

short_model_combinations = [
    {
        'predict_lookaheads': comb[0],
        'thresholds': comb[1],
        'ETP_thresholds': comb[2],
        'stop_losses': comb[3],
        'max_positions': comb[4]
    }
    for comb in Short_combinations
]
summary_file_path = os.path.join(full_folder_path, f'A_Summary_{name}.csv')

with open(summary_file_path,mode='w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['L_predict_lookahead','L_thresholds','L_ETP_threshold','L_SL','L_Max position','S_predict_lookahead','S_thresholds','S_ETP_threshold','S_SL','S_Max position','Total_Profit_Sum', 'Number_of_Winning_Trades_Sum', 'Number_of_Losing_Trades_Sum','Strategy_max_drawdown_sum','Sharpe_Ratio_sum',
                     'L_Total Profit','S_Total Profit', 'L_Strategy Max Drawdown', 'S_Strategy Max Drawdown', 
                     'L_Number of Winning Trades', 'L_Number of Losing Trades', 'L_Strategy Winning Losing Percentage', 'L_Strategy Sharpe Ratio', 'L_Total Return', 'L_Max Loss Per Trade', 'L_Max Consecutive Losses', 'L_Winning Weeks', 'L_Losing Weeks', 'L_Average Winning Amount', 'L_Average Losing Amount', 'L_Max No Trade Weeks', 'L_No Trade Weeks', 'L_Average Hold Count',
                     'S_Number of Winning Trades', 'S_Number of Losing Trades', 'S_Strategy Winning Losing Percentage', 'S_Strategy Sharpe Ratio', 'S_Total Return', 'S_Max Loss Per Trade', 'S_Max Consecutive Losses', 'S_Winning Weeks', 'S_Losing Weeks', 'S_Average Winning Amount', 'S_Average Losing Amount', 'S_Max No Trade Weeks', 'S_No Trade Weeks', 'S_Average Hold Count' ])
    for long_model in long_model_combinations:
        for short_model in short_model_combinations:
            # Long Model
            long_file_path = os.path.join(
                base_long_file_path,
                f"Backtesting_{long_model['predict_lookaheads']}_{long_model['thresholds']}_{long_model['ETP_thresholds']}_{long_model['stop_losses']}_{long_model['max_positions']}.csv"
            )
            data_long = pd.read_csv(long_file_path)
            print(data_long)
            S_long_file_path = os.path.join(
                Summary_long_file_path,
                f"A_Summary.csv"
            )
            Summary_long = pd.read_csv(S_long_file_path)
            print(Summary_long)
            filtered_row = Summary_long[
            (Summary_long['predict_lookahead'] == long_model['predict_lookaheads'])&
            (Summary_long['thresholds'] == long_model['thresholds']) &
            (Summary_long['ETP_threshold'] == long_model['ETP_thresholds']) &
            (Summary_long['SL'] == long_model['stop_losses']) &
            (Summary_long['Max position'] == long_model['max_positions'])]
            
            long_total_profit = filtered_row['Total Profit'].values[0]
            long_number_of_winning_trades = filtered_row['Number of Winning Trades'].iloc[0]
            long_number_of_losing_trades = filtered_row['Number of Losing Trades'].iloc[0]
            long_strategy_winning_losing_percentage = filtered_row['Strategy winning losing percentage'].iloc[0]
            long_strategy_sharpe_ratio = filtered_row['Strategy Sharpe Ratio'].iloc[0]
            long_strategy_max_drawdown = filtered_row['Strategy Max Drawdown'].iloc[0]
            long_total_return = filtered_row['total_return'].iloc[0]
            long_max_loss_per_trade = filtered_row['max_loss_per_trade'].iloc[0]
            long_max_consecutive_losses = filtered_row['max_consecutive_losses'].iloc[0]
            long_winning_weeks = filtered_row['winning_weeks'].iloc[0]
            long_losing_weeks = filtered_row['losing_weeks'].iloc[0]
            long_average_winning_amount = filtered_row['average_winning_amount'].iloc[0]
            long_average_losing_amount = filtered_row['average_losing_amount'].iloc[0]
            long_max_no_trade_weeks = filtered_row['max_no_trade_weeks'].iloc[0]
            long_no_trade_weeks = filtered_row['no_trade_weeks'].iloc[0]
            long_average_hold_count = filtered_row['average_hold_count'].iloc[0]
            # Short Model
            short_file_path = os.path.join(
                base_short_file_path,
                f"Backtesting_{short_model['predict_lookaheads']}_{short_model['thresholds']}_{short_model['ETP_thresholds']}_{short_model['stop_losses']}_{short_model['max_positions']}.csv"
            )
            data_short = pd.read_csv(short_file_path)
            print(data_short)
            S_short_file_path = os.path.join(
                            Summary_short_file_path,
                            f"A_Summary.csv"
                        )
            Summary_short = pd.read_csv(S_short_file_path)
            print(Summary_short)
            filtered_row = Summary_short[
            (Summary_short['predict_lookahead'] == short_model['predict_lookaheads'])&
            (Summary_short['thresholds'] == short_model['thresholds']) &
            (Summary_short['ETP_threshold'] == short_model['ETP_thresholds']) &
            (Summary_short['SL'] == short_model['stop_losses']) &
            (Summary_short['Max position'] == short_model['max_positions'])]
            short_total_profit = filtered_row['Total Profit'].values[0]
            short_number_of_winning_trades = filtered_row['Number of Winning Trades'].iloc[0]
            short_number_of_losing_trades = filtered_row['Number of Losing Trades'].iloc[0]
            short_strategy_winning_losing_percentage = filtered_row['Strategy winning losing percentage'].iloc[0]
            short_strategy_sharpe_ratio = filtered_row['Strategy Sharpe Ratio'].iloc[0]
            short_strategy_max_drawdown = filtered_row['Strategy Max Drawdown'].iloc[0]
            short_total_return = filtered_row['total_return'].iloc[0]
            short_max_loss_per_trade = filtered_row['max_loss_per_trade'].iloc[0]
            short_max_consecutive_losses = filtered_row['max_consecutive_losses'].iloc[0]
            short_winning_weeks = filtered_row['winning_weeks'].iloc[0]
            short_losing_weeks = filtered_row['losing_weeks'].iloc[0]
            short_average_winning_amount = filtered_row['average_winning_amount'].iloc[0]
            short_average_losing_amount = filtered_row['average_losing_amount'].iloc[0]
            short_max_no_trade_weeks = filtered_row['max_no_trade_weeks'].iloc[0]
            short_no_trade_weeks = filtered_row['no_trade_weeks'].iloc[0]
            short_average_hold_count = filtered_row['average_hold_count'].iloc[0]
            
            total_profit_sum = long_total_profit + short_total_profit
            number_of_winning_trades_sum = long_number_of_winning_trades + short_number_of_winning_trades
            number_of_losing_trades_sum = long_number_of_losing_trades + short_number_of_losing_trades

            # Convert the datetime columns to datetime objects for plotting
            data_long['datetime'] = pd.to_datetime(data_long['datetime'])
            data_short['datetime'] = pd.to_datetime(data_short['datetime'])

            combined_profit = pd.merge(data_long[['datetime', 'profit']],
                                    data_short[['datetime', 'profit']],
                                    on='datetime',
                                    suffixes=('_long', '_short'))

            combined_profit['combined_profit'] = combined_profit['profit_long'] + combined_profit['profit_short']
            combined_Strategy_Return = pd.merge(data_long[['datetime', 'Strategy_Return']],
                                    data_short[['datetime', 'Strategy_Return']],
                                    on='datetime',
                                    suffixes=('_long', '_short'))

            combined_Strategy_Return['combined_Strategy_Return'] = combined_Strategy_Return['Strategy_Return_long'] + combined_Strategy_Return['Strategy_Return_short']
            
            def calculate_max_drawdown(profits):
                cumulative_profit = profits.cumsum()  # 計算累積利潤
                peak = np.maximum.accumulate(cumulative_profit)  # 計算累積利潤的最高點
                # 檢查 peak 是否全為0或負值，這意味著沒有任何正的累積利潤
                drawdown = (cumulative_profit - peak) # 計算回撤百分比
                max_drawdown = drawdown.min()  # 取最小值作為最大回撤
                return max_drawdown
            Strategy_max_drawdown_sum = calculate_max_drawdown(combined_profit['combined_profit'])
            
            def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
                mean_return = np.mean(returns)  # 計算平均回報
                return_std = np.std(returns)  # 計算回報的標準差
                sharpe_ratio = ((mean_return - risk_free_rate)*365) / (return_std*sqrt(365)) if return_std != 0 else 0
                return sharpe_ratio
            sharpe_ratio_sum = calculate_sharpe_ratio(combined_Strategy_Return['combined_Strategy_Return'] )
            # Plotting total profit over time for long, short, and combined strategies
            plt.figure(figsize=(12, 8))

            # Assuming each DataFrame has 'total_profit' and 'datetime' columns
            plt.plot(data_long['datetime'], data_long['total_profit'], label='Long Position Total Profit', color='blue')
            plt.plot(data_short['datetime'], data_short['total_profit'], label='Short Position Total Profit', color='red')

            # Calculate combined profit (aligned by datetime)
            combined_data = pd.merge(data_long[['datetime', 'total_profit']],
                                    data_short[['datetime', 'total_profit']],
                                    on='datetime',
                                    suffixes=('_long', '_short'))

            combined_data['combined_total_profit'] = combined_data['total_profit_long'] + combined_data['total_profit_short']

            plt.plot(combined_data['datetime'], combined_data['combined_total_profit'], label='Combined Total Profit', linestyle='--', color='green')
            # Labels and title
            plt.xlabel('Time')
            plt.ylabel('Total Profit')
            plt.title('Long, Short, and Combined Total Profit Over Time')
            plt.legend()
            plt.grid(True)
            return_image_path = os.path.join(full_folder_path, f"Return_Long_{long_model['predict_lookaheads']}_{long_model['thresholds']}_{long_model['ETP_thresholds']}_{long_model['stop_losses']}_{long_model['max_positions']}_Short_{short_model['predict_lookaheads']}_{short_model['thresholds']}_{short_model['ETP_thresholds']}_{short_model['stop_losses']}_{short_model['max_positions']}.jpg")
            plt.savefig(return_image_path, dpi=150)  # Set dpi for higher resolution

            # Show the plot
            #plt.show()
            writer.writerow([long_model['predict_lookaheads'], long_model['thresholds'], long_model['ETP_thresholds'], long_model['stop_losses'], long_model['max_positions'], 
                             short_model['predict_lookaheads'], short_model['thresholds'], short_model['ETP_thresholds'], short_model['stop_losses'], short_model['max_positions'], 
                            total_profit_sum, number_of_winning_trades_sum, number_of_losing_trades_sum,Strategy_max_drawdown_sum,sharpe_ratio_sum,
                            long_total_profit, short_total_profit,long_strategy_max_drawdown,short_strategy_max_drawdown,
                            
                            long_number_of_winning_trades, long_number_of_losing_trades, long_strategy_winning_losing_percentage, long_strategy_sharpe_ratio, 
                            long_total_return, long_max_loss_per_trade, long_max_consecutive_losses, long_winning_weeks, long_losing_weeks, long_average_winning_amount, long_average_losing_amount,
                            long_max_no_trade_weeks, long_no_trade_weeks, long_average_hold_count,
                            
                            short_number_of_winning_trades, short_number_of_losing_trades, short_strategy_winning_losing_percentage, short_strategy_sharpe_ratio, 
                            short_total_return, short_max_loss_per_trade, short_max_consecutive_losses, short_winning_weeks, short_losing_weeks, short_average_winning_amount, short_average_losing_amount,
                            short_max_no_trade_weeks, short_no_trade_weeks, short_average_hold_count
                             ])






