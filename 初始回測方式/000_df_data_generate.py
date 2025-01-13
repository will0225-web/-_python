import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam

from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, PReLU, Conv1D, MaxPooling1D, Flatten, LeakyReLU, ReLU, Bidirectional, Attention, LayerNormalization, Input, Activation, RepeatVector, Permute, Multiply, Layer, concatenate, Concatenate
from tensorflow.keras.initializers import GlorotUniform

from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

from keras.callbacks import ReduceLROnPlateau, TensorBoard
from datetime import datetime
from scipy import stats

from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression

import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import pyarrow.csv as pv
####生出原始資料
def prepare_features_and_targets_1(lookahead_value,tp_value):
    file_path = r'C:\\Users\\User\\Documents\\df_time_segment_21_07_01.csv'
    df= pv.read_csv(file_path).to_pandas()
    df= df[(df['lookahead'] == lookahead_value) & (df['TP'] == tp_value)].copy()
    file_path = r'C:\\Users\\User\\Documents\\df_time_segment_24_09_03.csv'
    df_2= pv.read_csv(file_path).to_pandas()
    df_2= df_2[(df_2['lookahead'] == lookahead_value) & (df_2['TP'] == tp_value)].copy()
    df_time= pd.concat([df, df_2], ignore_index=True)
    return df_time
def generate_new_orders(df):
    new_orders=[]
    for i in range(len(df)):
        if df['order'].iloc[i] == -1:
            if df['flag_down'].iloc[i]==1 and df['flag_up'].iloc[i]==0:
                 new_orders.append(0)  ##先往下
            elif df['flag_down'].iloc[i]==0 and df['flag_up'].iloc[i]==1:
                new_orders.append(1)  ###先往上
            else:
                new_orders.append(3)  ###都沒有
        elif df['order'].iloc[i] == 0:
                new_orders.append(0)
        elif df['order'].iloc[i] == 1:
             new_orders.append(1)
        else:
            new_orders.append(2)  # 同一根K線滿足上下
    df.loc[:,'new_orders'] = new_orders
    # 將結果返回到數據框中
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'] + pd.DateOffset(hours=8)
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    return df
standard_features = ['Upper Band', 'Lower Band', 'Middle Band', 'VWAP']
def generate_orders(filtered_df , lookahead_value, tp_value):
    filtered_df.reset_index(drop=True, inplace=True)  # 重新設定索引
    orders=[]
    for idx in range(len(filtered_df)-lookahead_value):
        if filtered_df['flag_up'].iloc[idx] == 1 and filtered_df['flag_down'].iloc[idx] == 1:
                # 根據時間順序來判斷 order
            first_up_time = (filtered_df['High'].iloc[idx+1:idx+lookahead_value+1] >= filtered_df['Close'].iloc[idx] * (1 + tp_value[0])).idxmax()
            first_down_time = (filtered_df['Low'].iloc[idx+1:idx+lookahead_value+1] <= filtered_df['Close'].iloc[idx] * (1 - tp_value[0])).idxmax()
            #print(f"lookahead: {lookahead_value}, TP: {tp_value}, first_up_time: {first_up_time}, first_down_time: {first_down_time}")
            if first_up_time == first_down_time:
                orders.append(2)  # 同一根K線同時達到，標記為 2
            elif first_up_time < first_down_time:
                orders.append(1)  # 先達到向上的目標，標記為 1
            else:
                orders.append(0)  # 先達到向下的目標，標記為 0
        else:
                orders.append(-1)  # 不同時滿足條件，標記為 -1
    # 將結果返回到數據框中
    if len(orders) < len(filtered_df):
        orders.extend([-1] * (len(filtered_df) - len(orders)))
    filtered_df.loc[:,'order'] = orders
    #print(f"lookahead: {lookahead_value}, TP: {tp_value}, first_up_time: {first_up_time}, first_down_time: {first_down_time}")
    print(f"Appending filtered_df for lookahead={lookahead_value}, TP={tp_value}, Length: {len(filtered_df)}")
    return filtered_df






def prepare_features_and_targets(file_path,max_lookahead, cols,thresholds):
    
    df= pv.read_csv(file_path).to_pandas()
    features = []
    targets_up=[]
    targets_down = []
    for i in range(len(df) - max_lookahead):
            current_close = df['Close'].iloc[i]
            high_window = df['High'].iloc[i+1:i+max_lookahead+1].max()
            low_window = df['Low'].iloc[i+1:i+max_lookahead+1].min()
            for threshold in thresholds:
                features.append([
                df['datetime'].iloc[i],
                df['Open'].iloc[i],
                df['High'].iloc[i],
                df['Low'].iloc[i],
                df['Close'].iloc[i],
                df['Open_Volatility'].iloc[i],
                df['High_Volatility'].iloc[i],
                df['Low_Volatility'].iloc[i],
                df['Close_Volatility'].iloc[i],
                df['Cumulative_VP'].iloc[i],
                df['Cumulative_Volume'].iloc[i],
                df['Volume'].iloc[i],
                df['Volume_log'].iloc[i],
                df['Cumulative_Volume_log'].iloc[i],
                df['fibonacci_0.382'].iloc[i],
                df['fibonacci_0.5'].iloc[i],
                df['fibonacci_0.618'].iloc[i],
                df['fibonacci_1'].iloc[i],
                df['fibonacci_0'].iloc[i],
                df['Open_Close_pct'].iloc[i],
                df['High_Low_pct'].iloc[i],
                df['Up_Shadow_pct'].iloc[i],
                df['Down_Shadow_pct'].iloc[i],
                df['EMA_7'].iloc[i],
                df['EMA_25'].iloc[i],
                df['EMA_99'].iloc[i],
                df['BB_Width'].iloc[i],
                df['RSI'].iloc[i],
                df['ATR'].iloc[i],
                df['VWAP'].iloc[i],
                df['Upper Band'].iloc[i],
                df['Lower Band'].iloc[i],
                df['Middle Band'].iloc[i],
                max_lookahead,
                threshold,
            ])
                targets_down.append(int((current_close - low_window) / current_close >= threshold))
                targets_up.append(int((high_window - current_close) / current_close >= threshold))
    return pd.DataFrame(features, columns=cols), np.array(targets_down),np.array(targets_up)

def generate_data(file_path, max_lookahead,thresholds):
    cols = [
            'datetime','Open', 'High', 'Low', 'Close', 'Open_Volatility', 'High_Volatility', 'Low_Volatility', 'Close_Volatility', 'Cumulative_VP','Cumulative_Volume',
            'Volume','Volume_log','Cumulative_Volume_log', 'fibonacci_0.382','fibonacci_0.5', 'fibonacci_0.618','fibonacci_1', 'fibonacci_0', 'Open_Close_pct', 'High_Low_pct', 
            'Up_Shadow_pct', 'Down_Shadow_pct', 'EMA_7', 'EMA_25', 'EMA_99', 'BB_Width', 'RSI', 'ATR', 'VWAP','Upper Band','Lower Band', 'Middle Band','lookahead', 'TP'
        ]

    df,y_down,y_up= prepare_features_and_targets(file_path, max_lookahead, cols,thresholds)
    df['flag_down'] = y_down
    df['flag_up'] =y_up
    #result_df = generate_new_orders(generate_orders(df,max_lookahead,thresholds))
    result_df = df
    print(result_df)
    #生成 CSV 檔案的名稱
    file_name_down = f'df_data_{max_lookahead}_{thresholds[0]}.csv'  
    # 完整的檔案路徑
    full_file_path_down = os.path.join(base_folder_path, file_name_down)
    # 儲存 DataFrame 為 CSV
    result_df.to_csv(full_file_path_down, index=False)
    
     
#name = 'Spot'
name = 'Futures' 
date = '1108'


file_path = fr'C:\\Users\\User\\Documents\\{name}_策略資料_{date}\\New_{name}_data_{date}.csv'
base_folder_path = fr'C:\\Users\\User\\Documents\\{name}_策略資料_{date}'

thresholds = [round(i * 0.005, 3) for i in range(2, 3)]
for lookahead in range(120,121,24):
        for d in thresholds:
            generate_data(file_path,lookahead,[d])

   
#file_path = r'C:\\Users\\User\\Documents\\Original_data_0916.csv'
#df= pd.read_csv(file_path)
#print(df)
#print(df.columns)
# 計算每個特徵與目標的相關性
