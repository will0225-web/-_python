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

from modules import data as modules_data
from modules import signals as signals
from modules import indicators as indicators
from modules import model as model_process
from modules import custom_model_fit_indicators as custom_model_fit_indicators
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from datetime import datetime
from scipy import stats

from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression

import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


file_path = r'C:\\Users\\User\\Documents\\Original_data_0816.csv'
df= pd.read_csv(file_path)
### 整理所需要欄位

df['Future_High'] = df['Close'] * 1.008
df['Future_Low'] = df['Close'] * 0.992
cols = [
        'Open', 'High', 'Low', 'Close','datetime' ,'Future_High', 'Future_Low', 'Volume', 'fibonacci_0.382', 
        'fibonacci_0.5', 'fibonacci_0.618', 'Open_Close_pct', 'High_Low_pct', 
        'Up_Shadow_pct', 'Down_Shadow_pct', 'EMA_7', 'EMA_25', 'EMA_99', 'MACD_Cross', 'RSI', 'Middle Band', 'Upper Band', 'Lower Band','lookahead','TP','flag'
    ]
# 最大前瞻窗口
max_lookahead = 24
thresholds = [i * 0.001 for i in range(1, 21)]
# 准备特征和目标变量
def prepare_features_and_targets(df, max_lookahead, cols,thresholds):
    features = []
    data = df[['Open', 'High', 'Low', 'Close','datetime' , 'Future_High', 'Future_Low', 'Volume',
               'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'Open_Close_pct',
               'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'EMA_7', 'EMA_25',
               'EMA_99', 'MACD_Cross', 'RSI', 'Middle Band', 'Upper Band', 'Lower Band']].values
    for i in range(len(data) - max_lookahead):
        current_close = data[i, 3]
        for j in range(2, max_lookahead + 1):
            low_window = data[i+1:i+j,2].min()
            for threshold in thresholds:
                flag = 1 if (current_close - low_window) / current_close >= threshold else 0
                features.append([
                    *data[i],
                    j,
                    threshold,
                    flag
                ])
    return pd.DataFrame(features, columns=cols)
# 生成特征和目标
X= prepare_features_and_targets(df, max_lookahead, cols,thresholds)
print(X)
# 输出到 CSV 文件
import dask.dataframe as dd
# 分塊寫入CSV文件
chunk_size = 100000  # 根據數據量調整塊大小
for i in range(0, X.shape[0], chunk_size):
    X.iloc[i:i + chunk_size].to_csv('Short_TP_data_.csv', index=False, mode='a', header=(i == 0))







