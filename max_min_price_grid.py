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

from scipy import stats

from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression

import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import csv
from sklearn.preprocessing import OneHotEncoder

def customized_specific_period_col(df):
    customized_cols_infos = []
    
    # 紀錄原先的cols
    original_cols = set(df.columns)
    price_look_back = 672
    
    # indicator_look_back = 288
    
    # df, lower_low_higher_high_info = indicators.add_lower_low_higher_high(df, 0.04, look_back=indicator_look_back, is_need_return_function_info=True)
    # customized_cols_infos.append(lower_low_higher_high_info)
    
    # df, price_indicator_info = indicators.add_price_indicator(df, look_back=price_look_back, is_need_return_function_info=True)
    # customized_cols_infos.append(price_indicator_info)

    # df, look_back_24h_min_max_price_info = indicators.add_24h_min_max_price(df, look_back=96, is_need_return_function_info=True)
    # customized_cols_infos.append(look_back_24h_min_max_price_info)

    df = calculate_volatility(df, 20)
    di_len = 14
    adx_len = 14
    df = calculate_adx(df, di_len, adx_len)
    df = calculate_moving_averages(df)
    df = calculate_bollinger_bands(df)
    df = calculate_momentum(df)
    df = calculate_ppo(df)
    # df = calculate_trend(df, 288, 0.04)
    df = indicators.trend_min_max_price_indicator(df, 192)
    df = indicators.calculate_candelstick_patterns(df)
    df = calculate_VWAP(df, window=20)

    # # 對數收益率
    # df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    # df['Volatility'] = df['Log_Returns'].rolling(window=20).std()
    
    # df['SMA_5'] = indicators.calculate_sma(df['Close'], 5)
    # df['SMA_10'] = indicators.calculate_sma(df['Close'], 10)

    df['EMA_7'] = indicators.calculate_ema(df['Close'], 7)
    df['EMA_25'] = indicators.calculate_ema(df['Close'], 25)
    df['EMA_99'] = indicators.calculate_ema(df['Close'], 99)

    
    # 紀錄新的cols
    modified_cols = set(df.columns)
    # 篩選多出來的cols
    new_cols = list(modified_cols - original_cols)
    return df, customized_cols_infos, new_cols

def calculate_VWAP(df, window=20):
    df['Typical_Price'] = (df['Close'] + df['High'] + df['Low']) / 3
    # df['Typical_Price'] = df['Close']
    df['VP'] = df['Typical_Price'] * df['Volume']

    # df['Cumulative_VP'] = df['VP'].cumsum()
    # df['Cumulative_Volume'] = df['Volume'].cumsum()
    # 使用滚动窗口计算 VP 和 Volume 的累积和
    df['Cumulative_VP'] = df['VP'].rolling(window=window).sum()
    df['Cumulative_Volume'] = df['Volume'].rolling(window=window).sum()

    df['VWAP'] = df['Cumulative_VP'] / df['Cumulative_Volume']

    return df


def calculate_dm(df):
    df['up'] = df['High'] - df['High'].shift(1)
    df['down'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['-DM'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
    return df

def calculate_rma(series, period):
    rma = series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return rma

def calculate_di(df, period):
    df['TR_sum'] = calculate_rma(df['TR'], period)
    df['+DM_sum'] = calculate_rma(df['+DM'], period)
    df['-DM_sum'] = calculate_rma(df['-DM'], period)
    df['+DI'] = 100 * (df['+DM_sum'] / df['TR_sum'])
    df['-DI'] = 100 * (df['-DM_sum'] / df['TR_sum'])
    return df

def calculate_dx(df):
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    return df

def calculate_adx(df, di_len, adx_len):
    df = calculate_dm(df)
    df = calculate_di(df, di_len)
    df = calculate_dx(df)
    df['ADX'] = calculate_rma(df['DX'], adx_len)
    return df

def calculate_volatility(df, look_back):
    df['Close_Volatility'] = df['Close'].rolling(window=look_back).std()
    df['High_Volatility'] = df['High'].rolling(window=look_back).std()
    df['Low_Volatility'] = df['Low'].rolling(window=look_back).std()
    df['Open_Volatility'] = df['Open'].rolling(window=look_back).std()
    return df

def calculate_moving_averages(df, short_window=50, long_window=200):
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    return df

def calculate_bollinger_bands(df):
    df['BB_Width'] = df['Upper Band'] - df['Lower Band']
    df['BB_Pos'] = (df['Close'] - df['Lower Band']) / (df['Upper Band'] - df['Lower Band'])
    return df

def calculate_momentum(df, window=10):
    df['Momentum'] = df['Close'].diff(window)
    return df

def calculate_ppo(df, short_window=12, long_window=26):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['PPO'] = (short_ema - long_ema) / long_ema * 100
    return df

# 假设df是你的数据框，包含所有需要的列
def calculate_rolling_stats(df, window=20):
    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()
    return rolling_mean, rolling_std

def set_y_label(df, lookahead):
    for i in range(len(df) - lookahead):
        future_high_window = df['High'].iloc[i+1:i+lookahead+1]
        # 設置未來lookahead內high的最高價格
        df.at[i, 'y'] = future_high_window.max()
    return df

def validation(test_data, model, preprocessor, close_scaler, cols, lookahead, price_related_features, robust_features):
    # 预测并存储结果
    model.set_params(device='cuda')

    input_data = test_data.iloc[:len(test_data) - lookahead].copy()
    input_data = input_data[cols]
    input_data = set_y_label(input_data, lookahead)
    y = input_data['y']
    input_data.drop('y', axis=1, inplace=True)
    

    

    preprocessed_data = preprocessor.transform(input_data)

    # 预测lookahead内的最高价格
    # preds = model.predict(preprocessed_data)
    # probs = model.predict_proba(preprocessed_data)
    predicted_values = model.predict(preprocessed_data)

    input_data['predicted_max_price'] = predicted_values
    input_data['y'] = y

    # 先筛选出 Volume 大于 10,000 的数据
    # input_data = input_data[input_data['Volume'] < 15000]
    

    denominator = len(input_data)

    # 定义正确预测的条件：predicted_class 在 y 的 ±0.2% 范围内
    accuracy_threshold = 0.002  # 0.2% = 0.002
    input_data['is_correct'] = abs(input_data['predicted_max_price'] - input_data['y']) / input_data['y'] <= accuracy_threshold

    numerator = input_data['is_correct'].sum()

    if denominator > 0:
        total_accuracy = numerator / denominator
    else:
        total_accuracy = 0

    residuals = abs(predicted_values - y)
    percentage_residuals = (residuals / y) * 100  # 将残差转换为百分比
    avg_percentage_residual = percentage_residuals.mean()  # 平均百分比误差
    return total_accuracy, avg_percentage_residual
    

def get_test_data():
    # predict目前最新的資料
    end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

    end_time_seconds = end_time / 1000
    end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
    end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, 60000, '', is_need_save_original_data=False, is_need_calculated=True)

    # 将 'datetime' 列转换为 datetime 类型
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 将 'datetime' 列转换为 UTC+8
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

    # 如果需要移除时区信息，可以使用 .dt.tz_localize(None)
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    df, customized_cols_infos, new_cols = customized_specific_period_col(df)

    drop_front_data_count = 1000
    df = df[drop_front_data_count:]
    df.reset_index(drop=True, inplace=True)

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.interpolate(method='linear', inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 检查 NaN 值的数量
    nan_counts = df[numeric_cols].isna().sum()
    print("NaN Counts:\n", nan_counts)

    inf_counts = np.isinf(df[numeric_cols]).sum()
    print("Infinity Counts:\n", inf_counts)

    total_nan_inf = nan_counts + inf_counts
    print("Total NaN and Infinity Counts:\n", total_nan_inf)

    # 检查所有列中 NaN 值的行
    nan_volume_rows = df[df[numeric_cols].isna().any(axis=1)]
    inf_volume_rows = df[np.isinf(df[numeric_cols]).any(axis=1)]
    print("Rows with NaN column:\n", nan_volume_rows)
    print("Rows with inf column:\n", inf_volume_rows)
    df.reset_index(drop=True, inplace=True)

    return df

def scaler(df, price_related_features, robust_features, close_scaler=None, preprocessor=None):
    robust_features = robust_features
    standard_features = []
    minMax_features = []

    # 假设 X 是你的数据集
    # 定义需要以 Close 尺度缩放的特征
    close_feature = ['Close']
    price_related_features = price_related_features

    if close_scaler is None:
        # 初始化 StandardScaler 并仅在 Close 上进行拟合
        close_scaler = StandardScaler()
        X_close_scaled = close_scaler.fit_transform(df[close_feature])
    else:
        X_close_scaled = close_scaler.transform(df[close_feature])

    X_scaled_without_price = df.drop(price_related_features + close_feature, axis=1)
    if preprocessor is None:
        # 列出每个缩放器/转换器对应的特征
        preprocessor = ColumnTransformer(
            transformers=[
                ('robust', RobustScaler(), robust_features),
                ('standard', StandardScaler(), standard_features),
                ('minMax', MinMaxScaler(feature_range=(0, 1)), minMax_features),
            ],
            remainder='passthrough'  # 不需要缩放的特征保持原样
        )
        X_scaled = preprocessor.fit_transform(X_scaled_without_price)
    else:
        X_scaled = preprocessor.transform(X_scaled_without_price)

    # 使用 Close 的均值和标准差手动缩放其他相关特征
    price_related_scaled = (df[price_related_features] - close_scaler.mean_[0]) / close_scaler.scale_[0]

    # 将缩放后的 price_related_features 替换回 X_scaled
    X_scaled_df = pd.DataFrame(X_scaled, columns=preprocessor.get_feature_names_out())

    # 将 price_related_scaled 替换回 X_scaled_df 中
    for i, feature in enumerate(price_related_features):
        X_scaled_df[feature] = price_related_scaled.iloc[:, i]

    X_scaled_df['Close'] = X_close_scaled
    # 导出特定列到 CSV 文件
    X_scaled = X_scaled_df.to_numpy()

    return X_scaled, close_scaler, preprocessor


symbol = "ETHUSDT"
interval = "5m"
look_back = 1 #使用回看n根數據
epochs = 150
batch_size = 128
total_klines = 450000
get_local_file_name = 'ETHUSDT_15m_2023-12-31_23-59-59_450000_calculated.csv'

input_model_infos = []

# end_time = int(datetime.datetime.timestamp(datetime.datetime.now())) * 1000

# end_time_seconds = end_time / 1000
# end_datetime = datetime.datetime.fromtimestamp(end_time_seconds)
# end_time_string = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# end_time_string = "2023-12-31 23:59:59"
# df = modules_data.get_binance_klines_backward(symbol, interval, end_time_string, total_klines, get_local_file_name, is_need_save_original_data=False, is_need_calculated=True)
# df, customized_cols_infos, new_cols = customized_specific_period_col(df)

# # 将 'datetime' 列转换为 datetime 类型
# df['datetime'] = pd.to_datetime(df['datetime'])

# # 将 'datetime' 列转换为 UTC+8
# df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')

# # 如果需要移除时区信息，可以使用 .dt.tz_localize(None)
# df['datetime'] = df['datetime'].dt.tz_localize(None)

# df.fillna(0, inplace=True)
# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df.interpolate(method='linear', inplace=True)

# numeric_cols = df.select_dtypes(include=[np.number]).columns

# # 检查 NaN 值的数量
# nan_counts = df[numeric_cols].isna().sum()
# print("NaN Counts:\n", nan_counts)

# inf_counts = np.isinf(df[numeric_cols]).sum()
# print("Infinity Counts:\n", inf_counts)

# total_nan_inf = nan_counts + inf_counts
# print("Total NaN and Infinity Counts:\n", total_nan_inf)

# # 检查所有列中 NaN 值的行
# nan_volume_rows = df[df[numeric_cols].isna().any(axis=1)]
# inf_volume_rows = df[np.isinf(df[numeric_cols]).any(axis=1)]
# print("Rows with NaN column:\n", nan_volume_rows)
# print("Rows with inf column:\n", inf_volume_rows)
# df.reset_index(drop=True, inplace=True)

# df.to_csv('original_df.csv', index=False)

df = pd.read_csv('original_df.csv')

lookahead = 12
cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0',
        'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct',
        'EMA_7', 'EMA_25', 'EMA_99', 
        'BB_Width', 'Upper Band', 'Lower Band', 'Middle Band',
        'Close_Volatility', 'High_Volatility', 'Low_Volatility', 'Open_Volatility', 'ATR', 'VWAP', 'Cumulative_VP', 'Cumulative_Volume'
    ]
X = df[cols]

X = set_y_label(X, lookahead)

drop_front_data_count = 1000
X = X[drop_front_data_count:-300]
X.reset_index(drop=True, inplace=True)


# X = set_y_multiple_label(X, 96, lookahead, percentage)
# X.to_csv('X.csv', index=False)

y = X['y']
X.drop('y', axis=1, inplace=True)

robust_features = ['Volume', 'Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'ATR', 'BB_Width', 'Close_Volatility', 'High_Volatility', 'Low_Volatility', 'Open_Volatility', 'Cumulative_VP', 'Cumulative_Volume']
# standard_features = []
standard_features = ['Close', 'High', 'Low', 'Open', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0', 'EMA_7', 'EMA_25', 'EMA_99', 'Upper Band', 'Lower Band', 'Middle Band', 'VWAP']
minMax_features = []

# 假设 X 是你的数据集
# 定义需要以 Close 尺度缩放的特征
### 這部分得到的成果沒有比較好
# close_feature = ['Close']
# price_related_features = ['High', 'Low', 'Open', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0', 'EMA_7', 'EMA_25', 'EMA_99', 'Upper Band', 'Lower Band', 'Middle Band', 'VWAP', 'up_percentage', 'down_percentage']
# X_scaled, close_scaler, preprocessor = scaler(X, price_related_features, robust_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('robust', RobustScaler(), robust_features),
        ('standard', StandardScaler(), standard_features),
        ('minMax', MinMaxScaler(feature_range=(0, 1)), minMax_features),
    ],
    remainder='passthrough'  # 不需要缩放的特征保持原样
)
X_scaled = preprocessor.fit_transform(X)

test_data = get_test_data()
# 打开CSV文件准备写入
with open('model_results_linear.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入列名
    writer.writerow(['n_estimators', 'max_depth', 'learning_rate', 'total_accuracy', 'avg_percentage_residual'])

    for n in range(500, 1100, 100):
        for d in range(2, 6, 1):
            for lr in np.arange(0.001, 0.011, 0.001):
                # model_big_trend = xgb.XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr, random_state=42, device='cuda', verbosity=1)
                model_big_trend = xgb.XGBRegressor(n_estimators=n, max_depth=d, learning_rate=lr, random_state=42, device='cuda', verbosity=1)

                model_big_trend.fit(X_scaled, y)

                # total_accuracy, class_accuracies, class_denominators, class_numerators, total_profit_percentage = validation(test_data, model_big_trend, preprocessor, close_scaler, cols, lookahead, percentage, price_related_features, robust_features)
                total_accuracy, avg_percentage_residual = validation(test_data, model_big_trend, preprocessor, None, cols, lookahead, None, robust_features)

                writer.writerow([n, d, 0, lr, total_accuracy, avg_percentage_residual])

                print(f"n_estimators: {n}, max_depth: {d}, leaves: {0}, learning_rate: {lr}, total_accuracy: {total_accuracy}, avg_percentage_residual: {avg_percentage_residual}")