import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from scipy import stats
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV,TimeSeriesSplit, cross_val_score

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier

from joblib import Memory
import pyarrow.csv as pv
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, roc_curve
 
from sklearn.metrics import accuracy_score, classification_report
param_log = [] # 建立一個交易記錄列表來儲存交易訊息
trade_log = []
def evaluate_model( predict_lookahead, TP, entry_confidence, base_folder_path=None, sub_folder_name=None):    
    file_path = os.path.join(
    r'C:\\Users\\User\\Documents\\Futures_策略資料_1019',
    f'df_data_{predict_lookahead}_{TP}.csv'
        )
    df= pv.read_csv(file_path).to_pandas()
    cols = [ 
            'Open', 'High', 'Low', 'Close', 'Open_Volatility', 'High_Volatility', 'Low_Volatility', 'Close_Volatility'
            ,'fibonacci_0.382','fibonacci_0.5', 'fibonacci_0.618','fibonacci_1', 'fibonacci_0', 'Open_Close_pct', 'High_Low_pct', 
            'Up_Shadow_pct', 'Down_Shadow_pct','RSI','Lower Band','Cumulative_VP','Volume_log','Cumulative_Volume_log'
            ]
    y_up= df['flag_up']
    y_down= df['flag_down']
    X_train_o, X_test_o, y_up_train, y_up_test,y_down_train, y_down_test = train_test_split(df, y_up, y_down, test_size=0.2, random_state=42, shuffle=False)
    # 計算切割點的索引
    # split_idx = int(len(df) * 0.7)  # 計算前 70% 的索引位置
    # X_train_o = df[split_idx:]  # 後面 30% 作為訓練資料
    # X_test_o = df[:split_idx]   # 前面 70% 作為測試資料
    # y_up_train = y_up[split_idx:]  # y_up 的後面 30% 作為訓練資料
    # y_up_test = y_up[:split_idx]   # y_up 的前面 70% 作為測試資料
    # total_len = len(df)
    # train_split_idx = int(total_len * 0.7)  # 前 40% 作為訓練集
    # test_split_idx = int(total_len * 0.7)   # 中間 30% 作為測試集 (前 70% 作為訓練 + 測試，剩下 30% 為驗證集)
    # # 分割資料集
    # X_train_o = df[:train_split_idx]               # 前 40% 作為訓練集
    # X_test_o = df[train_split_idx:test_split_idx]  # 中間 30% 作為測試集         
    # 設定時間分割點，這裡用例子表示你想要的訓練、測試、驗證時間範圍
    # train_start_time = '2024-01-01 00:00:00'
    # train_end_time = '2024-09-07 00:00:00'  # 訓練資料的結束時間
    # test_start_time = '2021-01-01 00:00:00'
    # test_end_time = '2022-07-01 00:00:00'   # 測試資料的結束時間，驗證資料從這之後開始
    # # 分割資料集，基於時間範圍
    # X_train_o = df[(df['datetime']> train_start_time) & (df['datetime'] <= train_end_time)]  # 這裡用來選擇訓練集    
    # X_test_o = df[(df['datetime'] > test_start_time) & (df['datetime'] <= test_end_time)]  # 測試集
    # # 分割對應的 y_up 和 y_down
    # y_up_train = y_up[(df['datetime']> train_start_time) & (df['datetime'] <= train_end_time)]  # 訓練集的 y_up
    # y_up_test = y_up[(df['datetime'] > test_start_time) & (df['datetime'] <= test_end_time)]  # 測試集的 y_up
    # y_down_train =y_down[(df['datetime']> train_start_time) & (df['datetime'] <= train_end_time)]  # 訓練集的 y_up
    # y_down_test= y_up[(df['datetime'] > test_start_time) & (df['datetime'] <= test_end_time)]  # 測試集的 y_up
    # y_down_train = y_down[split_idx:]  # y_down 的後面 30% 作為訓練資料
    # y_down_test = y_down[:split_idx]   # y_down 的前面 70% 作為測試資料
    X_train=X_train_o[cols]
    X_test=X_test_o[cols]
    print(X_train)
    print(X_test)
    full_folder_path = os.path.join(base_folder_path, sub_folder_name)
    # 如果資料夾不存在則建立
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)
    
    def scaler(X_train,X_test):
        robust_features = ['Open_Close_pct', 'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'Close_Volatility', 'High_Volatility', 'Low_Volatility', 'Open_Volatility','Cumulative_VP']
        standard_features = ['Close', 'High', 'Low', 'Open', 'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0','Lower Band', 'Volume_log', 'Cumulative_Volume_log']
        minMax_features = []
        preprocessor = ColumnTransformer(
            transformers=[
                ('price', RobustScaler(), robust_features),
                ('percent', StandardScaler(), standard_features),
                ('bounded', MinMaxScaler(feature_range=(0, 1)), minMax_features),
            ],
            remainder='passthrough'  # 不需要缩放的特征保持原样
        )
        X_train_scaler = preprocessor.fit_transform(X_train)
        X_test_scaler = preprocessor.transform(X_test)
        with open(os.path.join(full_folder_path,f'scaler_{predict_lookahead}_{TP}_down.pkl'), 'wb') as f:
            pickle.dump(preprocessor, f)
        
        return  X_train_scaler,X_test_scaler
    
    X_valid, X_valid_2 = scaler(X_train,X_test)
    # y_up= y_up_train.to_numpy()
    y_down= y_down_train.to_numpy()
    # y_test_up= y_up_test.to_numpy()
    y_test_down= y_down_test.to_numpy()
    df_2 = X_test_o
    print(df_2)
    
    def find_best_para(threshold):
        param_grid = {
            'n_estimators': [50,100,200,250],
            'max_depth': [3,5,7,9],
            'learning_rate': [0.05,0.1],
            'alpha': [0.1,0.5,1],
            'gamma':[0.1,1], 
            'reg_lambda': [0.1,1]
        }
        best_params_down = None
        best_score_down = 0
        best_model_down =None
        for params in ParameterGrid(param_grid):
            model_down = xgb.XGBClassifier(tree_method="hist", device="cuda",n_estimators=params['n_estimators'],
                                    max_depth=params['max_depth'],
                                    learning_rate=params['learning_rate'],reg_alpha = params['alpha'], gamma = params['gamma'],reg_lambda=params['reg_lambda'],
                                    random_state=42)
            model_down.fit(X_valid,y_down, verbose=True)
        
            ###對測試資料進行模型預測
            y_pred_down=model_down.predict(X_valid_2)
            y_new_pred_proba_down = model_down.predict_proba(X_valid_2)
            close= df_2['Close']
            high = df_2[ 'High']
            low = df_2[ 'Low']
            lookahead = df_2['lookahead']
            threshold_value=df_2['TP']
            date_time = df_2['datetime']
            ##找UP model
            Result_df_2 = pd.DataFrame({
                'Close' :close,
                'High':high,
                'Low':low,
                'datetime' :date_time,
                'lookahead' :lookahead,
                'TP': threshold_value,
                'prob_down' : y_new_pred_proba_down[:, 1],
                'prob_up' : y_new_pred_proba_down[:, 0],
                'True_down_label' :y_test_down,
                'Predict_down_label': y_pred_down,
            })
            c = Result_df_2[Result_df_2['prob_down'] > threshold].shape[0]
            d = Result_df_2[(Result_df_2['prob_down'] > threshold) & (Result_df_2['True_down_label'] == 1)].shape[0]
            win_rate_down = d/c if c != 0 else 0
            report_down = classification_report(y_test_down, y_pred_down, output_dict=True)
            df_report_down = pd.DataFrame(report_down).transpose()
            print(df_report_down)
            f1_score_down_class_1 = report_down['1']['f1-score']
            trade_log.append(f"Params down: {params},Avg Win Rate: {win_rate_down:.4f},F1 score:{f1_score_down_class_1:.4f}") 

            ####找最大勝率
            if win_rate_down>= best_score_down:
                best_score_down = win_rate_down
                best_params_down = params
                best_result_df_down=Result_df_2
                best_y_pred_down = y_pred_down
                best_model_down =model_down

            print(f"df_data_{predict_lookahead}_{TP}_Params: {params}, F1 up : {f1_score_down_class_1:.4f}")
            print(f"Best Params down: {best_params_down}, Best Avg Win Rate: {best_score_down:.4f}")
        
        return best_model_down,best_score_down,best_params_down,best_result_df_down,best_y_pred_down
    
    best_model_down,best_score_down,best_params_down,best_result_df_down,best_y_pred_down =find_best_para(entry_confidence)
    
    param_log.append(f'df_data_{predict_lookahead}_{TP}_Best Params: {best_params_down}, Best Avg Win Rate: {best_score_down:.4f}')

    # 生成 CSV 檔案的名稱
    file_name_down = f'Short_Matrix_test_{predict_lookahead}_{TP}_down.csv'    
    # 完整的檔案路徑
    full_file_path_down = os.path.join(full_folder_path, file_name_down)
    # 儲存 DataFrame 為 CSV
    best_result_df_down.to_csv(full_file_path_down, index=False)
    # 使用已訓練的模型預測 X_valid 的機率
    
    plt.figure(figsize=(12, 6))
    plt.plot(best_result_df_down['datetime'], best_result_df_down['Close'], label='Close Price', color='blue')
    # Marking the points where prob_up > 0.8, Predict_up_label == 1, and True_up_label == 1 (correct predictions) with green dots
    plt.scatter(best_result_df_down[(best_result_df_down['Predict_down_label'] == 1) & 
                                    (best_result_df_down['True_down_label'] == 1) & 
                                    (best_result_df_down['prob_down'] > entry_confidence)]['datetime'], 
                best_result_df_down[(best_result_df_down['Predict_down_label'] == 1) & 
                                    (best_result_df_down['True_down_label'] == 1) & 
                                    (best_result_df_down['prob_down'] > entry_confidence)]['Close'], 
                color='green', label=f'Correct Prediction (prob_down > {entry_confidence})', zorder=5)

    # Marking the points where prob_up > 0.8, Predict_up_label == 1, and True_up_label == 0 (false positives) with red dots
    plt.scatter(best_result_df_down[(best_result_df_down['Predict_down_label'] == 1) & 
                                    (best_result_df_down['True_down_label'] == 0) & 
                                    (best_result_df_down['prob_down'] > entry_confidence)]['datetime'], 
                best_result_df_down[(best_result_df_down['Predict_down_label'] == 1) & 
                                    (best_result_df_down['True_down_label'] == 0) & 
                                    (best_result_df_down['prob_down'] > entry_confidence)]['Close'], 
                color='red', label=f'False Positive (prob_down > {entry_confidence})', zorder=5)

    # Adding labels and title
    plt.xlabel('Datetime')
    plt.ylabel('Close Price')
    plt.title(f'Price vs. Time with Correct and False Predictions (prob_up > {entry_confidence})')
    # Adding legend
    plt.legend()
    # Display the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    return_image_path = os.path.join(full_folder_path, f'Return_{predict_lookahead}_{TP}.jpg')
    plt.savefig(return_image_path, dpi=150)  # 設置 dpi 來提高圖像解析度

    ####保存模型
    # folder_path = r'C:\\Users\\User\\Documents\\Save_Model_correlation_win_rate_08_資料倒訓'
    os.makedirs(full_folder_path, exist_ok=True)
    
    # with open(os.path.join(folder_path,f'xgb_model_{predict_lookahead}_{TP}_up.pkl'), 'wb') as f:
    #     pickle.dump(best_model_up, f)
    with open(os.path.join(full_folder_path, f'xgb_model_{predict_lookahead}_{TP}_down.pkl'), 'wb') as f:
       pickle.dump(best_model_down, f)
    
    y_new_pred_proba_down = best_model_down.predict_proba(X_valid_2)
    prob_down_class_1 = y_new_pred_proba_down[:, 1]
    df_2['prob_down'] = prob_down_class_1
    thresholds_down = [0.95,0.9, 0.8, 0.7,0.6,0.5]
    results_down = {"Threshold": [], "Win Rate": [],"Win Count":[],"Total Count":[]}
    #print(df_down)
    for threshold in thresholds_down:
        c = best_result_df_down[best_result_df_down['prob_down'] > threshold].shape[0]
        d = best_result_df_down[(best_result_df_down['prob_down'] > threshold) & (best_result_df_down['True_down_label'] == 1)].shape[0]
        win_rate_down = d/c if c != 0 else 0
        results_down["Threshold"].append(f"Prob_down_over_{threshold}")
        results_down["Win Rate"].append(win_rate_down)
        results_down["Win Count"].append(d)
        results_down["Total Count"].append(c)
        print(f'Prob_down_over_{threshold}_win_rate[{win_rate_down}]')
    df_down = pd.DataFrame(results_down)
    #print(df_up)
    accuracy_down = accuracy_score(y_test_down, best_y_pred_down)
    # print(f'Accuracy_up: {accuracy_up:.2f}')
    print(f'Accuracy_down: {accuracy_down:.2f}')
    auc_score_down = roc_auc_score(y_test_down, prob_down_class_1 )
    #print(f"AUC-ROC Score: {auc_score_up:.4f}")
    print(f"AUC-ROC Score: {auc_score_down:.4f}")
    # 打印每個閾值對應的 FPR 和 TPR
    # 找到 FPR 和 TPR 比較平衡的閾值
    #print(f"Optimal Threshold: {optimal_threshold_up:.2f}")  
    #print(f"Optimal Threshold: { optimal_threshold_down:.2f}")
    # 顯示詳細的分類報告
    #print(classification_report(y_test_up, best_y_pred_up))
    # 計算分類報告並轉換為DataFrame
    report_down = classification_report(y_test_down, best_y_pred_down, output_dict=True)
    df_report_down = pd.DataFrame(report_down).transpose()
    print(df_report_down)
    f1_score_down_class_1 = report_down['1']['f1-score']
    #print(classification_report(y_test_down, best_y_pred_down))
# 創建一個ExcelWriter物件
    return f1_score_down_class_1,df_down,df_report_down,accuracy_down,auc_score_down


import csv
thresholds = [round(i * 0.005, 3) for i in range(1,11)] 
# with open('output_results_0926_F1_score.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['predict_lookahead','thresholds','Accuracy_up','auc_score_up','f1_score_up_class_1'])
#     for lookahead in range(24,504,24):
#         for d in thresholds:
#             f1_score_up_class_1,df_up,df_report_up,accuracy_up, auc_score_up  = evaluate_model(lookahead,d)
#             writer.writerow([lookahead,d,accuracy_up, auc_score_up ,f1_score_up_class_1 ])
name = 'Short_correlatcion_win_rate_0.8_train_1723_test_2324_96_480_添加volume_log'

# 定義輸出結果檔案的完整路徑
folder_path = r'C:\\Users\\User\\Documents\\Output_result'
 # 使用當前時間生成子資料夾名稱，或根據其他參數自定義名稱
summary_file_path = os.path.join(folder_path , f'output_results_每一個評估結果_{name}.csv')

with open(summary_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['','','','','','predict_lookahead','thresholds','Accuracy_down','auc_score_down','f1_score_down_class_1'])
    for lookahead in range(96,482,24):
        for d in thresholds:
            # 主資料夾路徑
            base_folder_path = r'C:\\Users\\User\\Documents\\Save_Model'        
            # 使用當前時間生成子資料夾名稱，或根據其他參數自定義名稱
            sub_folder_name = f'Save_model_{name}'
            f1_score_down_class_1,df_down,df_report_down,accuracy_down,auc_score_down = evaluate_model(lookahead,d, 0.8,base_folder_path=base_folder_path,sub_folder_name=sub_folder_name)
            writer.writerow(['','','','','',lookahead,d,accuracy_down, auc_score_down ,f1_score_down_class_1 ])
            writer.writerow([f'評估結果_{lookahead}_{d}'])
            # 空行作為分隔
            # 寫入 df_down 的標題和內容
            writer.writerow(['','df_down Results'])
            df_down.insert(0,'','') 
            df_down.to_csv(file, index=False)
            writer.writerow(['','Classification Report down'])
            df_report_down.insert(0,'', '') 
            df_report_down.to_csv(file,index=False) 

df_folder_path =r'C:\\Users\\User\\Documents\\Output_result\\df_para_log'
# 最後，可以將交易記錄輸出成DataFrame，或存到檔案
df_para_log = pd.DataFrame(param_log, columns=['Trade_Log'])
summary_df_name = f'df_para_log_{name}.csv'
full_file_path_summary_df= os.path.join(df_folder_path, summary_df_name) 
df_para_log.to_csv(full_file_path_summary_df, index=False)


df_folder_path =r'C:\\Users\\User\\Documents\\Output_result\\df_trade_log'
# 最後，可以將交易記錄輸出成DataFrame，或存到檔案
df_trade_log = pd.DataFrame(trade_log, columns=['Trade_Log'])
summary_df_name = f'df_trad_log_{name}.csv'
full_file_path_summary_df= os.path.join(df_folder_path, summary_df_name) 
df_trade_log.to_csv(full_file_path_summary_df, index=False)


