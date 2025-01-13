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
def evaluate_model(df, predict_lookahead, TP,ETP, SL, max_open_positions, entry_confidence=None, folder_path=None):
    ###載入回測資料
    X_test = df[['Close','High','Low','prob_up','datetime']]
    # 初始化Signal列 
    X_test['Signal'] = 0  
    # 生成買入訊號，當prob_up > 0.9且prob_down < 0.5時，信號為1
    X_test.loc[(X_test['prob_up'] > entry_confidence), 'Signal'] = 1
    # 檢查多空訊號
    #print(X_test['Signal'].value_counts()[1])
    total_signals = X_test['Signal'].value_counts().get(1,0)
    # 確認計算 weekly_signal
    if total_signals > 0:
        weekly_signal = total_signals // 100
    else:
        weekly_signal = 0
    #以每周平均訊號數下單
    #max_open_positions =weekly_signal
    X_test['profit'] = 0.0
    X_test['Strategy_Return'] = 0.0
    X_test['Holding_Period'] = 0  # 新增一個欄位來記錄持有期天數
    X_test['Exit_Index'] = -1  # 用於記錄出場的位置
    
    # 初始化計算變數
    initial_investment = 100000/max_open_positions  # 每次交易的投入資金為10,000美金 
    # 初始化投資與交易變數
    initial_capital = 100000
    fee = 0.0010 #交易手續費
    total_investment = 0  # 當前的總投資金額
    total_profit = 0  # 總利潤
    profit_list = []  # 每480根K線記錄的利潤
    investment_list = []  # 每480根K線記錄的持倉金額
    return_list = []  # 每480根K線記錄的回報率
    slippage = 0.0005
    risk_per_trade = 0.1  # 每次交易風險1%
    # 用於記錄所有開倉的資訊
    open_trades = []
    signal_count_list = []  # 每480根K線記錄Signal次數
    trade_count_list = []  # 每480根K線記錄Trade次數
    signal_count = 0  # 訊號次數
    trade_count = 0  # 實際交易次數
    trade_log = [] # 建立一個交易記錄列表來儲存交易訊息
    # 記錄最近一個signal=1時的條件，用於更新所有倉位的止盈/止損
    last_entry_price = None  # 最後一次signal=1時的價格
    # 新增計算最大損失的變數
    max_loss = float('-inf')  # 初始化最大虧損
    current_consecutive_losses = 0
    max_consecutive_losses = 0
    
    # 新增統計數據
    winning_weeks = 0
    losing_weeks = 0
    no_trade_weeks = 0
    max_no_trade_weeks = 0
    no_trade_streak = 0
    total_holding_days = 0  # 用於累積每個倉位的持有期
    total_closed_trades = 0  # 用於計算完成交易的總數
    winning_amounts = []
    losing_amounts = []

    # 遍歷所有K線數據
    for i in range(0, len(X_test)):
        if open_trades:
            # 檢查每個持倉是否應該根據更新後的條件進行平倉
            for trade in open_trades[:]:
                    entry_price = trade['entry_price']
                    entry_index = trade['entry_index']
                    holding_days = i - entry_index
                    current_close = X_test.at[i, 'Close']
                    high_price = X_test.at[i, 'High']
                    low_price = X_test.at[i, 'Low']
                    # 使用目前價格來計算新的 TP 和 SL
                    # 計算新的止盈和止損條件
                    new_tp_price = last_entry_price * (1 + TP + ETP)
                    new_sl_price = last_entry_price * (1 + SL)
                    # 確保只在新條件更有利時更新
                    trade['tp_price'] = max(trade.get('tp_price', entry_price * (1 + TP + ETP)), new_tp_price)
                    trade['sl_price'] = min(trade.get('sl_price', entry_price * (1 + SL)), new_sl_price)
                    
                     # 檢查止盈條件  
                    if high_price >= trade['tp_price']:
                        exit_price = trade['tp_price']
                        profit = (exit_price - entry_price) * initial_investment / entry_price - fee * initial_investment
                        X_test.at[entry_index, 'profit'] += profit
                        X_test.at[entry_index, 'Strategy_Return'] += profit / initial_investment
                        X_test.at[entry_index, 'Holding_Period'] = holding_days
                        X_test.at[entry_index, 'Exit_Index'] = i
                        total_profit += profit
                        total_investment -= initial_investment
                        total_holding_days += holding_days  # 累積持有期
                        total_closed_trades += 1  # 計算完成的交易數量
                        open_trades.remove(trade)  # 平倉後移除該倉位
                        # 記錄交易訊息
                        trade_log.append(f'平倉: 止盈 at index {i}, Profit: {profit}')
                        # 更新連續虧損計數
                        continue
                        
                    elif low_price <= trade['sl_price']:
                        exit_price = trade['sl_price']
                        profit = (exit_price - entry_price) * initial_investment / entry_price - fee * initial_investment
                        X_test.at[entry_index, 'profit'] += profit
                        X_test.at[entry_index, 'Strategy_Return'] += profit / initial_investment
                        X_test.at[entry_index, 'Holding_Period'] = holding_days
                        X_test.at[entry_index, 'Exit_Index'] = i
                        total_profit += profit
                        total_investment -= initial_investment
                        total_holding_days += holding_days  # 累積持有期
                        total_closed_trades += 1  # 計算完成的交易數量
                        open_trades.remove(trade)  # 平倉後移除該倉位
                        trade_log.append(f'平倉: 止損 at index {i}, Profit: {profit}')
                        continue
                    elif holding_days >= predict_lookahead:
                        signal_found=False
                        for j in range(i-predict_lookahead , i):
                            if j>=0:
                                if X_test.at[j,'Signal']==1:
                                    signal_found=True
                                #print(f'新訊號出現，延長持有期 at index {j}')
                                break  # 停止檢查，重新延長持有期
                        if not signal_found:
                            new_tp_price = entry_price * (1 + TP)  # 捨去 ETP，只使用原本的止盈條件
                            new_sl_price = trade['sl_price'] * 1.05  # 例如，將止損價格提高 5%
                            # 更新止盈和止損價格
                            trade['tp_price'] = new_tp_price
                            # 更新止盈和止損價格
                            trade['sl_price'] = new_sl_price
                            if high_price > trade['tp_price']:  # 如果有正收益但沒達到止盈
                                exit_price = trade['tp_price'] 
                                profit = (exit_price - entry_price) * initial_investment / entry_price - fee * initial_investment
                                X_test.at[entry_index, 'profit'] += profit
                                X_test.at[entry_index, 'Strategy_Return'] += profit / initial_investment
                                X_test.at[entry_index, 'Holding_Period'] = holding_days
                                X_test.at[entry_index, 'Exit_Index'] = i
                                total_profit += profit
                                total_investment -= initial_investment
                                total_holding_days += holding_days  # 累積持有期
                                total_closed_trades += 1  # 計算完成的交易數量
                                open_trades.remove(trade)  # 平倉後移除該倉位
                                trade_log.append(f'平倉: 持有期結束獲利 at index {i}, Profit: {profit}')

                            elif low_price <= trade['sl_price']:   # 持有期滿的止損檢查
                                exit_price = trade['sl_price']                
                                profit = (exit_price - entry_price) * initial_investment / entry_price - fee * initial_investment
                                X_test.at[entry_index, 'profit'] += profit
                                X_test.at[entry_index, 'Strategy_Return'] += profit / initial_investment
                                X_test.at[entry_index, 'Holding_Period'] = holding_days
                                X_test.at[entry_index, 'Exit_Index'] = i
                                total_profit += profit
                                total_investment -= initial_investment
                                total_holding_days += holding_days  # 累積持有期
                                total_closed_trades += 1  # 計算完成的交易數量
                                open_trades.remove(trade)  # 平倉後移除該倉位
                                trade_log.append(f'平倉: 持有期結束於止損 index {i}, Profit: {profit}')
                        continue
        if X_test.at[i, 'Signal'] == 1 :
                signal_count += 1  # 每當出現Signal時增加一次訊號計數
                if len(open_trades) < max_open_positions:
                # 開新倉位
                    entry_price = X_test.at[i, 'Close'] # 考慮滑點影響
                    last_entry_price = entry_price  # 更新最後一次的entry_price
                    entry_index = i
                    # 記錄新倉位的信息
                    open_trades.append({
                        'entry_price': entry_price,
                        'entry_index': entry_index,
                        'tp_price': entry_price * (1 + TP + ETP),  # 初始止盈
                        'sl_price': entry_price * (1 + SL),  # 初始止損
                        'max_loss': 0  # 初始化最大損失
                    })
                    # 更新總投資金額
                    total_investment += initial_investment
                    trade_count += 1  # 每當實際進行交易時增加一次交易計數
                    trade_log.append(f'開新倉: at index {i}, Entry Price: {entry_price}')        
        
            # 每480根K線記錄一次利潤與回報
        if i % 480 == 0 and i > 0:
                profit_list.append(total_profit)
                investment_list.append(total_investment)
                return_rate = total_profit / total_investment if total_investment > 0 else 0
                return_list.append(return_rate)
                signal_count_list.append(signal_count)
                trade_count_list.append(trade_count)
                signal_count = 0
                trade_count = 0
                total_profit = 0
        
        if i % 480 == 0 and i > 0:
                weekly_profit = X_test['profit'].iloc[i - 480:i].sum()
                weekly_signal_count = X_test['Signal'].iloc[i - 480:i].sum()
                if weekly_signal_count == 0:
                    no_trade_streak += 1
                    no_trade_weeks += 1
                    if no_trade_streak > max_no_trade_weeks:
                        max_no_trade_weeks = no_trade_streak
                else:
                    no_trade_streak = 0

    # 計算贏單週數與輸單週數
    winning_weeks = sum(1 for profit in profit_list if profit > 0)
    losing_weeks = sum(1 for profit in profit_list if profit < 0)    
    # 計算平均持有期
    average_hold_count = total_holding_days / total_closed_trades if total_closed_trades > 0 else 0
    # 最後，可以將交易記錄輸出成DataFrame，或存到檔案
    # 計算平均數據
     # 計算平均數據
    average_winning_amount = np.mean([p for p in profit_list if p > 0]) if winning_weeks > 0 else 0
    average_losing_amount = np.mean([p for p in profit_list if p < 0]) if losing_weeks > 0 else 0
     
 
    df_trade_log = pd.DataFrame(trade_log, columns=['Trade_Log'])   
     # 打印結果
    print(f"幾個禮拜是贏單: {winning_weeks}")
    print(f"幾個禮拜是輸單: {losing_weeks}")
    print(f"平均贏單金額: {average_winning_amount}")
    print(f"平均輸單金額: {average_losing_amount}")
    print(f"最多沒下單週數: {max_no_trade_weeks}")
    print(f"平均沒下單週數: {no_trade_weeks}")
    print(f"平均持有期: {average_hold_count}")
    # 最後總結全部交易的利潤與回報
    df = pd.DataFrame({
        'Profit': profit_list,
        'Weekly Investment': investment_list,
        'Return' :return_list,
        'Signal Count': signal_count_list,
        'Trade Count': trade_count_list
    })
    # 返回結果與交易日誌
    # 統計最大單筆虧損
    max_loss_per_trade = X_test['profit'].min()

    # 統計最多連續虧損單數
    consecutive_losses = (X_test['profit'] < 0).astype(int).groupby((X_test['profit'] >= 0).cumsum()).cumsum()
    max_consecutive_losses = consecutive_losses.max()
    
    total_profit = df['Profit'].sum()
    print(f'Total Profit: {total_profit}, Final Return: {total_profit / total_investment if total_investment > 0 else 0}')
    # 確保 profit 列沒有空值或無效數據
    if X_test['profit'].isnull().any():
        raise ValueError("Profit column contains NaN values. Check your data.")

    # 確保有非零的利潤記錄
    if X_test['profit'].sum() == 0:
        print("Warning: No profits recorded. Check your trading signals or data.")
        
    # 計算 公式Max Drawdown和公式Sharpe Ratio
    # 基於累積利潤來計算 Max Drawdown
    def calculate_max_drawdown(profits):
        cumulative_profit = profits.cumsum()  # 計算累積利潤
        peak = np.maximum.accumulate(cumulative_profit)  # 計算累積利潤的最高點
        # 檢查 peak 是否全為0或負值，這意味著沒有任何正的累積利潤
        drawdown = (cumulative_profit - peak) # 計算回撤百分比
        max_drawdown = drawdown.min()  # 取最小值作為最大回撤
        return max_drawdown
    
    Strategy_max_drawdown = calculate_max_drawdown(X_test['profit'])
    print(Strategy_max_drawdown)
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        mean_return = np.mean(returns)  # 計算平均回報
        return_std = np.std(returns)  # 計算回報的標準差
        sharpe_ratio = ((mean_return - risk_free_rate)*365) / (return_std*sqrt(365)) if return_std != 0 else 0
        return sharpe_ratio
    sharpe_ratio = calculate_sharpe_ratio(X_test['Strategy_Return'])
    Commom_Ratio = abs(sharpe_ratio/Strategy_max_drawdown) if Strategy_max_drawdown != 0 else 0
    print(f"Strategy Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Strategy Max Drawdown: {Strategy_max_drawdown:.5f}")
    
    ###計算策略和市場回報
    X_test['total_profit'] = X_test['profit'].cumsum()
    X_test['Cumulative_Strategy_Return'] = X_test['Strategy_Return'].cumsum()
    X_test['Next_Close'] = X_test['Close'].shift(-1)
    X_test['Market_Return'] = (X_test['Next_Close'] - X_test['Close']) / X_test['Close']
    X_test['Cumulative_Market_Return'] = X_test['Market_Return'].cumsum()
    
    #計算Market Drawdown
    def max_drawdown(return_series):
        cum_return = (1 + return_series).cumprod() 
        peak = cum_return.cummax()
        drawdown = (cum_return - peak) / peak
        return drawdown.min()
    market_max_drawdown = max_drawdown(X_test['Market_Return'])
    print(f"Market Max Drawdown: {market_max_drawdown:.2%}")
    
    # 定義贏單和輸單（基於百分比回報）
    X_test['Win'] = (X_test['Strategy_Return'] > 0).cumsum()
    X_test['Loss'] = (X_test['Strategy_Return'] < 0).cumsum()
    # 計算贏單和輸單的數量
    num_wins = X_test['Win'].iloc[-1]
    num_losses = X_test['Loss'].iloc[-1]
    # 輸出贏單和輸單數量
    print(f"Number of Winning Trades: {num_wins}")
    print(f"Number of Losing Trades: {num_losses}")
    
    # 自訂義夏普比率的計算
    total_winning_return = X_test[X_test['Strategy_Return'] > 0]['Strategy_Return'].sum() 
    total_losing_return = X_test[X_test['Strategy_Return'] < 0]['Strategy_Return'].sum() 
    # 計算總回報（基於初始資金）
    total_return = X_test['Strategy_Return'].sum()
    print(f'total_return:{total_return}')       
    # 計算贏單和輸單的總比例（使用實際資金）
    total_winning_percentage = total_winning_return / total_return if total_return != 0 else 0
    total_losing_percentage = total_losing_return / total_return if total_return != 0 else 0
    # 計算 Sharpe Ratio
    strategy_sharpe = total_winning_percentage / abs(total_losing_percentage) if total_losing_percentage != 0 else 0
    print(f"Strategy winning losing percentage: {strategy_sharpe:.2f}")
    
    # 畫圖，並使用實際的累積回報
    plt.figure(figsize=(14, 7))
    plt.plot(X_test['Cumulative_Strategy_Return'], label='Strategy Return')
    plt.plot(X_test['Cumulative_Market_Return'], label='Market Return')
    plt.legend()
    # 使用 folder_path 來保存圖像
    return_image_path = os.path.join(folder_path, f'Return_{predict_lookahead}_{TP}_{ETP}_{SL}_{max_open_positions}.jpg')
    plt.title('Cumulative Return Comparison')
    plt.savefig(return_image_path, dpi=150)  # 設置 dpi 來提高圖像解析度
    plt.close()
    plt.show()
    
    # 畫圖，並使用實際的累積Profit
    plt.figure(figsize=(14, 7))
    plt.plot(X_test['total_profit'], label='profit')
    plt.legend()
    profit_image_path = os.path.join(folder_path,f'Profit_{predict_lookahead}_{TP}_{ETP}_{SL}_{max_open_positions}.jpg')
    plt.savefig(profit_image_path, dpi=150)  # 設置 dpi 來提高圖像解析度
    plt.title('profit')
    plt.close()
    plt.show()

    if folder_path is None:
        raise ValueError("Folder path must be provided.")
    
    ###儲存每個週期的利潤與回報
    sub_folder_name_week = 'Weekly_profit_invest'
    full_folder_path_week = os.path.join(folder_path, sub_folder_name_week)
    if not os.path.exists(full_folder_path_week):
        os.makedirs(full_folder_path_week)
    file_name_up =f'Weekly_profit_invest_{predict_lookahead}_{TP}_{ETP}_{SL}_{max_open_positions}.csv'
    full_file_path_up = os.path.join(full_folder_path_week, file_name_up) 
    df.to_csv(full_file_path_up, index=False)
    
    ###儲存全部測試K線集
    sub_folder_name_daily = 'Daily_profit_invest'
    full_folder_path_daily  = os.path.join(folder_path, sub_folder_name_daily )
    if not os.path.exists(full_folder_path_daily ):
        os.makedirs(full_folder_path_daily )
    X_test_up = f'Backtesting_{predict_lookahead}_{TP}_{ETP}_{SL}_{max_open_positions}.csv'
    X_test_path_up = os.path.join(full_folder_path_daily, X_test_up) 
    X_test.to_csv(X_test_path_up, index=False)

    ###儲存交易平倉、開倉、止營、止損數據
    sub_folder_name_log= 'Daily_profit_log'
    full_folder_path_log  = os.path.join(folder_path, sub_folder_name_log )
    if not os.path.exists(full_folder_path_log ):
        os.makedirs(full_folder_path_log )
    summary_df_name_log = f'df_trade_log_{predict_lookahead}_{TP}_{ETP}_{SL}_{max_open_positions}.csv'
    full_file_path_summary_df= os.path.join(full_folder_path_log, summary_df_name_log) 
    df_trade_log.to_csv(full_file_path_summary_df, index=False)
    
    return total_profit, num_wins, num_losses, strategy_sharpe, sharpe_ratio, Strategy_max_drawdown, Commom_Ratio, total_return, max_loss_per_trade, max_consecutive_losses, winning_weeks, losing_weeks, average_winning_amount, average_losing_amount, max_no_trade_weeks, no_trade_weeks, average_hold_count

###Combinations組合: predict_lookahead、TP、SL、Max open position
# combinations = [(144,0.01,-0.1,10),(144,0.01,-0.15,10),(144,0.01,-0.05,10),(144,0.01,-0.1,20),(144,0.01,-0.15,20),(144,0.01,-0.05,20),
#                 (168,0.01,-0.1,10),(168,0.01,-0.15,10),(168,0.01,-0.05,10),(168,0.01,-0.1,20),(168,0.01,-0.15,20),(168,0.01,-0.05,20),]

# predict_lookaheads = [144,168]
# thresholds = [round(i * 0.005, 3) for i in range(1,6)] 
# combinations = [(96, 0.005), (96, 0.01), (120, 0.005), (120, 0.01), (144, 0.005), (144, 0.01), (144, 0.015), (168, 0.005), (168, 0.01), (168, 0.015), (192, 0.005), (192, 0.01), (192, 0.015), (216, 0.005), (216, 0.01), (216, 0.015), (240, 0.005), (240, 0.01), (240, 0.015), (240, 0.02), (264, 0.005), (264, 0.01), (264, 0.015), (264, 0.02), (288, 0.005), (288, 0.01), (288, 0.015), (288, 0.02), (312, 0.005), (312, 0.01), (312, 0.015), (312, 0.02), (336, 0.005), (336, 0.01), (336, 0.015), (336, 0.02), (336, 0.025), (360, 0.005), (360, 0.01), (360, 0.015), (360, 0.02), (360, 0.025), (384, 0.005), (384, 0.01), (384, 0.015), (384, 0.02), (384, 0.025), (408, 0.005), (408, 0.01), (408, 0.015), (408, 0.02), (408, 0.025), (432, 0.005), (432, 0.01), (432, 0.015), (432, 0.02), (432, 0.025), (456, 0.005), (456, 0.01), (456, 0.015), (456, 0.02), (456, 0.025), (480, 0.005), (480, 0.01), (480, 0.015), (480, 0.02), (480, 0.025), (480, 0.03)]
# combinations = [(96, 0.005), (96, 0.01), (120, 0.005), (120, 0.01), (144, 0.005),  (144, 0.01), (144, 0.015), (168, 0.005), (168, 0.01), (168, 0.015),  (192, 0.005), (192, 0.01), (192, 0.015), (216, 0.005), (216, 0.01),  (216, 0.015), (216, 0.02), (240, 0.005), (240, 0.01), (240, 0.015),  (240, 0.02), (264, 0.005), (264, 0.01), (264, 0.015), (264, 0.02),  (288, 0.005), (288, 0.01), (288, 0.015), (288, 0.02), (312, 0.005),  (312, 0.01), (312, 0.015), (312, 0.02), (312, 0.025), (336, 0.005),  (336, 0.01), (336, 0.015), (336, 0.02), (336, 0.025), (360, 0.005),  (360, 0.01), (360, 0.015), (360, 0.02), (360, 0.025), (384, 0.005),  (384, 0.01), (384, 0.015), (384, 0.02), (384, 0.025), (408, 0.005),  (408, 0.01), (408, 0.015), (408, 0.02), (408, 0.025), (432, 0.005),  (432, 0.01), (432, 0.015), (432, 0.02), (432, 0.025), (432, 0.03),  (456, 0.005), (456, 0.01), (456, 0.015), (456, 0.02), (456, 0.025),  (480, 0.005), (480, 0.01), (480, 0.015), (480, 0.02), (480, 0.025),  (480, 0.03)]
combinations =[(96, 0.005), (96, 0.01), (120, 0.005), (120, 0.01), (120, 0.015),  (144, 0.005), (144, 0.01), (144, 0.015), (168, 0.005), (168, 0.01),  (168, 0.015), (192, 0.005), (192, 0.01), (192, 0.015), (216, 0.005),  (216, 0.01), (216, 0.015), (240, 0.005), (240, 0.01), (240, 0.015),  (240, 0.02), (264, 0.005), (264, 0.01), (264, 0.015), (264, 0.02),  (288, 0.005), (288, 0.01), (288, 0.015), (288, 0.02), (312, 0.005),  (312, 0.01), (312, 0.015), (312, 0.02), (336, 0.005), (336, 0.01),  (336, 0.015), (336, 0.02), (360, 0.005), (360, 0.01), (360, 0.015),  (360, 0.02), (360, 0.025), (384, 0.005), (384, 0.01), (384, 0.015),  (384, 0.02), (384, 0.025), (408, 0.005), (408, 0.01), (408, 0.015),  (408, 0.02), (408, 0.025), (432, 0.005), (432, 0.01), (432, 0.015),  (432, 0.02), (432, 0.025), (456, 0.005), (456, 0.01), (456, 0.015),  (456, 0.02), (456, 0.025), (480, 0.005), (480, 0.01), (480, 0.015),  (480, 0.02), (480, 0.025), (480, 0.03)]

ETP_thresholds = [0,0.01,0.02,0.03]
stop_losses = [-0.02,-0.05,-0.1,-0.2]
max_positions = [10,20] 

name = 'Long_correlatcion_win_rate_0.9_train_1723_test_2324_96_480_添加volume_log'

# 設置file_path變數，這邊是需要特定lookahed和TP的每一個測試資料檔案
base_file_path = fr'C:\\Users\\User\\Documents\\Save_Model\\Long_Short_Model_添加Volume_log\\Save_model_{name}'

# 定義輸出結果檔案的完整路徑
base_folder_path = r'C:\\Users\\User\\Documents\\Backtesting_修改出場邏輯後'
sub_folder_name = f'Backtest_result_{name}_更改持有期間止盈止損'
# 完整資料夾路徑
full_folder_path = os.path.join(base_folder_path, sub_folder_name)
# 如果資料夾不存在則建立
if not os.path.exists(full_folder_path):
    os.makedirs(full_folder_path)

summary_file_path = os.path.join(full_folder_path, f'A_Summary_{name}_更改持有期間止盈止損.csv')

with open(summary_file_path,mode='w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['predict_lookahead','thresholds','ETP_threshold','SL','Max position','Total Profit', 'Number of Winning Trades','Number of Losing Trades','Strategy winning losing percentage','Strategy Sharpe Ratio','Strategy Max Drawdown', 'Commom_Ratio','total_return', 'max_loss_per_trade', 'max_consecutive_losses', 'winning_weeks', 'losing_weeks', 'average_winning_amount', 'average_losing_amount', 'max_no_trade_weeks', 'no_trade_weeks', 'average_hold_count'])
    
    # for predict_lookahead in predict_lookaheads:
    #         for threshold in thresholds:
    for predict_lookahead, threshold in combinations:
                for stop_loss in stop_losses:
                    for max_open_position in max_positions:
                        for ETP_threshold in ETP_thresholds:
    # for predict_lookahead, threshold,ETP_thresholds, stop_loss, max_open_position in combinations:
                            file_path = os.path.join(
                            base_file_path,
                            f'Long_Short_Matrix_test_{predict_lookahead}_{threshold}_up.csv' )
                            df= pd.read_csv(file_path)
                            ###儲存結果資料夾
                            folder_path = full_folder_path  
                            
                            total_profit, num_wins, num_losses, strategy_sharpe, sharpe_ratio, Strategy_max_drawdown, Commom_Ratio, total_return, max_loss_per_trade, max_consecutive_losses, winning_weeks, losing_weeks, average_winning_amount, average_losing_amount, max_no_trade_weeks, no_trade_weeks, average_hold_count= evaluate_model(df,predict_lookahead, threshold,ETP_threshold, stop_loss, max_open_position,
                            entry_confidence=0.9, folder_path=folder_path)
                            
                            writer.writerow([ predict_lookahead, threshold, ETP_threshold, stop_loss, max_open_position, total_profit, num_wins, num_losses, 
                                            strategy_sharpe, sharpe_ratio, Strategy_max_drawdown, Commom_Ratio, total_return, max_loss_per_trade, max_consecutive_losses, 
                                            winning_weeks, losing_weeks, average_winning_amount, average_losing_amount, max_no_trade_weeks, no_trade_weeks, average_hold_count])



    # 開始檢查每根K線
    # for i in range(0, len(X_test)):
    #     # 只有當 Signal = 1 時才進行投資
    #     if open_trades:
    #         # 如果存在任何持倉，檢查每一筆交易的條件是否滿足
    #         all_close = False  # 標記是否觸發全倉平倉
    #         for trade in open_trades[:]:
    #             entry_price = trade['entry_price']
    #             entry_index = trade['entry_index']
    #             holding_days = i - entry_index
                
    #             # 計算當前的價格變動
    #             price_change = (X_test.at[i, 'High'] - entry_price) / entry_price
    #             price_sl = (X_test.at[i, 'Close'] - entry_price) / entry_price
                
    #             if price_change >= TP + ETP:
    #                 exit_price = entry_price * (1 + TP + ETP)
    #                 profit = (exit_price - entry_price) * initial_investment / entry_price - fee * initial_investment
    #                 X_test.at[entry_index, 'profit'] += profit
    #                 X_test.at[entry_index, 'Strategy_Return'] += profit / initial_investment
    #                 X_test.at[entry_index, 'Holding_Period'] = holding_days
    #                 X_test.at[entry_index, 'Exit_Index'] = i
    #                 total_profit += profit
    #                 total_investment -= initial_investment
    #                 open_trades.remove(trade)  # 平倉後移除該倉位
    #                 # 記錄交易訊息
    #                 trade_log.append(f'平倉: 止盈 at index {i}, Profit: {profit}')
                
    #             elif price_sl <= SL:
    #                 exit_price = entry_price * (1 + SL)
    #                 profit = (exit_price - entry_price) * initial_investment / entry_price - fee * initial_investment
    #                 X_test.at[entry_index, 'profit'] += profit
    #                 X_test.at[entry_index, 'Strategy_Return'] += profit / initial_investment
    #                 X_test.at[entry_index, 'Holding_Period'] = holding_days
    #                 X_test.at[entry_index, 'Exit_Index'] = i
    #                 total_profit += profit
    #                 total_investment -= initial_investment
    #                 open_trades.remove(trade)  # 平倉後移除該倉位
    #                 trade_log.append(f'平倉: 止損 at index {i}, Profit: {profit}')
                
    #             # 檢查持有期結束時是否有盈利可平倉
    #             elif holding_days >= predict_lookahead:
    #                 signal_found=False
    #                 for j in range(i-predict_lookahead , i):
    #                     if j>=0:
    #                         if X_test.at[j,'Signal']==1:
    #                             signal_found=True
    #                         #print(f'新訊號出現，延長持有期 at index {j}')
    #                         break  # 停止檢查，重新延長持有期
    #                 if not signal_found:
    #                     if price_change > TP:  # 如果有正收益但沒達到止盈
    #                         exit_price = entry_price * (1 + TP ) 
    #                         profit = (exit_price - entry_price) * initial_investment / entry_price - fee * initial_investment
    #                         X_test.at[entry_index, 'profit'] += profit
    #                         X_test.at[entry_index, 'Strategy_Return'] += profit / initial_investment
    #                         X_test.at[entry_index, 'Holding_Period'] = holding_days
    #                         X_test.at[entry_index, 'Exit_Index'] = i
    #                         total_profit += profit
    #                         total_investment -= initial_investment
    #                         open_trades.remove(trade)  # 平倉後移除該倉位
    #                         trade_log.append(f'平倉: 持有期結束 at index {i}, Profit: {profit}')
            
    #     #檢查是否有新信號出現並進入新倉位
    #     if X_test.at[i, 'Signal'] == 1 :
    #         signal_count += 1  # 每當出現Signal時增加一次訊號計數
    #         if len(open_trades) < max_open_positions:
    #         # 開新倉位
    #             entry_price = X_test.at[i, 'Close']
    #             entry_index = i
    #             # 記錄新倉位的信息
    #             open_trades.append({
    #                 'entry_price': entry_price,
    #                 'entry_index': entry_index,
    #             })
    #             # 更新總投資金額
    #             total_investment += initial_investment
    #             trade_count += 1  # 每當實際進行交易時增加一次交易計數
    #             trade_log.append(f'開新倉: at index {i}, Entry Price: {entry_price}')     