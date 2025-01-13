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

file_path = r'C:\\Users\\User\\Documents\\Original_test_data.csv'
df= pd.read_csv(file_path)


df['Future_High'] = df['Close'] * 1.008
df['Future_Low'] = df['Close'] * 0.992
cols = [
        'Open', 'High', 'Low', 'Close', 'Future_High', 'Future_Low', 'Volume', 'fibonacci_0.382', 
        'fibonacci_0.5', 'fibonacci_0.618', 'Open_Close_pct', 'High_Low_pct', 
        'Up_Shadow_pct', 'Down_Shadow_pct', 'EMA_7', 'EMA_25', 'EMA_99', 'MACD_Cross', 'RSI',
          'Middle Band', 'Upper Band', 'Lower Band','lookahead','TP'
    ]
# 最大前瞻窗口
max_lookahead = 24
thresholds = [i * 0.001 for i in range(1, 21)]

# 准备特征和目标变量
def prepare_features_and_targets(df, max_lookahead, cols,thresholds):
    features = []
    targets_up=[]
    targets_down = []
    for i in range(len(df) - max_lookahead):
        current_close = df['Close'].iloc[i]

        for j in range(1, max_lookahead + 1):
            high_window = df['High'].iloc[i+1:i+j+1].max()
            low_window = df['Low'].iloc[i+1:i+j+1].min()
            for threshold in thresholds:
                features.append([
                df['Open'].iloc[i],
                df['High'].iloc[i],
                df['Low'].iloc[i],
                df['Close'].iloc[i],
                df['Future_High'].iloc[i],
                df['Future_Low'].iloc[i],
                df['Volume'].iloc[i],
                df['fibonacci_0.382'].iloc[i],
                df['fibonacci_0.5'].iloc[i],
                df['fibonacci_0.618'].iloc[i],
                df['Open_Close_pct'].iloc[i],
                df['High_Low_pct'].iloc[i],
                df['Up_Shadow_pct'].iloc[i],
                df['Down_Shadow_pct'].iloc[i],
                df['EMA_7'].iloc[i],
                df['EMA_25'].iloc[i],
                df['EMA_99'].iloc[i],
                df['MACD_Cross'].iloc[i],
                df['RSI'].iloc[i],
                df['Middle Band'].iloc[i],
                df['Upper Band'].iloc[i],
                df['Lower Band'].iloc[i],
                j,
                threshold,
            ])
                targets_down.append(int((current_close - low_window) / current_close >= threshold))
                targets_up.append(int((high_window - current_close) / current_close >= threshold))


    return pd.DataFrame(features, columns=cols), np.array(targets_down),np.array(targets_up)
# 生成特征和目标
df,y_down,y_up= prepare_features_and_targets(df, max_lookahead, cols,thresholds)

df['flag_down'] = y_down
df['flag_up'] =y_up
#df.to_csv('testing_data_多空.csv', index=False)
print(df)

def generate_orders(df, lookahead_value, tp_value):
    filtered_df = df[(df['lookahead'] == lookahead_value) & (df['TP'] == tp_value)].copy()
    filtered_df.reset_index(drop=True, inplace=True)  # 重新設定索引
    orders=[]
    for idx in range(len(filtered_df)-lookahead_value):
        if filtered_df['flag_up'].iloc[idx] == 1 and filtered_df['flag_down'].iloc[idx] == 1:
                # 根據時間順序來判斷 order
            first_up_time = (filtered_df['High'].iloc[idx+1:idx+lookahead_value+1] >= filtered_df['Close'].iloc[idx] * (1 + tp_value)).idxmax()
            first_down_time = (filtered_df['Low'].iloc[idx+1:idx+lookahead_value+1] <= filtered_df['Close'].iloc[idx] * (1 - tp_value)).idxmax()

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


    return filtered_df

all_df = []


# 顯示合併後的 DataFrame
lookahead_range= range(1,25)
thresholds = [i * 0.001 for i in range(1, 21)]
for lookahead_value in lookahead_range:
    for tp_value in thresholds:
        result_df = generate_orders(df, lookahead_value, tp_value)
        # 為了區分不同組合的結果，可以加入識別列
        all_df.append(result_df)


# 將所有結果合併成一個大的 DataFrame
final_df = pd.concat(all_df, ignore_index=True)
print(final_df)
# 顯示合併後的 DataFrame
print(final_df['order'].value_counts())
final_df.to_csv('filtered_df_多空_測試資料.csv', index=False)