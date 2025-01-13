import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
 
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight

from keras.callbacks import ReduceLROnPlateau, TensorBoard
from datetime import datetime
from scipy import stats

from sklearn.linear_model import LogisticRegression

import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import sys
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
# 使用sys.path.append()將父目錄添加到系統路徑中。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV,TimeSeriesSplit, cross_val_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import learning_curve
from joblib import Memory
import pyarrow.csv as pv
import dask.dataframe as dd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, classification_report
from xgboost import plot_tree
scaler = StandardScaler() 
import pickle
import subprocess
import sys
import joblib
# 使用 pickle 从文件加载模型
import sklearn
thresholds = [round(i * 0.005, 3) for i in range(2,3)] 
for predict_lookahead  in range(168,169,24):
        for TP  in thresholds:
            entry_confidence = 0.8
            folder_path = r'C:\\Users\\User\\Documents\\Save_Model\\Long_Short_Model_添加Volume_log\\Save_model_Long_correlatcion_win_rate_0.8_train_1723_test_2324_96_480'
            model_file_path = os.path.join(folder_path , f'xgb_model_{predict_lookahead}_{TP}_up.pkl')

            with open(model_file_path, 'rb') as f:
                model_up= pickle.load(f)


            scaler_file_path = os.path.join(folder_path , f'scaler_{predict_lookahead}_{TP}_up.pkl')

            with open(scaler_file_path, 'rb') as f:
                loaded_scaler = pickle.load(f)
            # subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "scikit-learn"])
            # subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.5.0"])
            
            new_folder_path = r'C:\\Users\\User\\Documents\\Spot_策略資料_1108'
            new_scaler_file_path = os.path.join(new_folder_path , f'new_scaler_{predict_lookahead}_{TP}_up.pkl')
            with open(new_scaler_file_path, 'wb') as f:
                pickle.dump(loaded_scaler, f)
            
            
            cols = [ 
                        'Open', 'High', 'Low', 'Close', 'Open_Volatility', 'High_Volatility', 'Low_Volatility', 'Close_Volatility'
                        ,'fibonacci_0.382','fibonacci_0.5', 'fibonacci_0.618','fibonacci_1', 'fibonacci_0', 'Open_Close_pct', 'High_Low_pct', 
                        'Up_Shadow_pct', 'Down_Shadow_pct','RSI','Lower Band','Cumulative_VP','Volume_log','Cumulative_Volume_log'
                        ]
            file_path = os.path.join(
                r'C:\\Users\\User\\Documents\\Spot_策略資料_1108',
                f'New_Spot_data_1108.csv'
                    )
            df= pv.read_csv(file_path).to_pandas()
            print(df)


            #y_up= df['flag_up']
            # y_test_up = y_up.to_numpy()
            df_2=df[cols]
            #y_test_up= y_up.to_numpy()
            print(df_2)
            # # 使用載入的縮放器對新的資料集進行縮放
            X_valid_2  = loaded_scaler.transform(df_2)
            # # 假設 X_test_scaler 是 Numpy 陣列，先轉為 DataFrame
            print(X_valid_2)
            y_pred_up=model_up.predict(X_valid_2)
            y_new_pred_proba_up = model_up.predict_proba(X_valid_2)
            close= df['Close']
            high = df[ 'High']
            low = df[ 'Low']
            df['lookahead'] = predict_lookahead
            lookahead = df['lookahead']
            df['TP'] =TP
            threshold_value=df['TP']
            date_time = df['datetime']
            Result_df_2 = pd.DataFrame({
                            'Close' :close,
                            'High':high,
                            'Low':low,
                            'datetime' :date_time,
                            'lookahead' :lookahead,
                            'TP': threshold_value,
                            'prob_down' : y_new_pred_proba_up[:, 0],
                            'prob_up' : y_new_pred_proba_up[:, 1],
                            #'True_down_label' :y_test_down,
                            # 'True_up_label' :y_test_up,
                            'Predict_up_label': y_pred_up,
                        })
            # c = Result_df_2[Result_df_2['prob_up'] > entry_confidence].shape[0]
            # d = Result_df_2[(Result_df_2['prob_up'] > entry_confidence) & (Result_df_2['True_up_label'] == 1)].shape[0]
            # win_rate_up = d/c if c != 0 else 0
            # report_up = classification_report(y_test_up, y_pred_up, output_dict=True)
            #df_report_up = pd.DataFrame(report_up).transpose()
           # print(df_report_up)
            #f1_score_up_class_1 = report_up['1']['f1-score']

            # 生成 CSV 檔案的名稱
            file_name_up = f'Long_Short_Matrix_test_{predict_lookahead}_{TP}_up.csv'    
            # 完整的檔案路徑
            full_file_path_up = os.path.join(new_folder_path, file_name_up)
            # 儲存 DataFrame 為 CSV
            Result_df_2.to_csv(full_file_path_up, index=False)

            best_result_df_up = Result_df_2


            plt.figure(figsize=(12, 6))
            # Plotting the Close price as a line
            plt.plot(best_result_df_up['datetime'], best_result_df_up['Close'], label='Close Price', color='blue')
            # Marking the points where prob_up > 0.8, Predict_up_label == 1, and True_up_label == 1 (correct predictions) with green dots
            plt.scatter(best_result_df_up[(best_result_df_up['Predict_up_label'] == 1) & 
                                            
                                            (best_result_df_up['prob_up'] > entry_confidence)]['datetime'], 
                        best_result_df_up[(best_result_df_up['Predict_up_label'] == 1) & 
                                            
                                            (best_result_df_up['prob_up'] > entry_confidence)]['Close'], 
                        color='green', label=f'Correct Prediction (prob_up > {entry_confidence})', zorder=5)

            # Marking the points where prob_up > 0.8, Predict_up_label == 1, and True_up_label == 0 (false positives) with red dots
            # plt.scatter(best_result_df_up[(best_result_df_up['Predict_up_label'] == 1) & 
            #                                 (best_result_df_up['True_up_label'] == 0) & 
            #                                 (best_result_df_up['prob_up'] > entry_confidence)]['datetime'], 
            #             best_result_df_up[(best_result_df_up['Predict_up_label'] == 1) & 
            #                                 (best_result_df_up['True_up_label'] == 0) & 
            #                                 (best_result_df_up['prob_up'] > entry_confidence)]['Close'], 
            #             color='red', label=f'False Positive (prob_up > {entry_confidence})', zorder=5)

            # Adding labels and title
            plt.xlabel('Datetime')
            plt.ylabel('Close Price')
            plt.title(f'Price vs. Time with Correct and False Predictions (prob_up > {entry_confidence})')
            # Adding legend
            plt.legend()
            # Display the plot
            plt.xticks(rotation=45)
            plt.tight_layout()
            return_image_path = os.path.join(new_folder_path, f'Return_{predict_lookahead}_{TP}.jpg')
            plt.savefig(return_image_path, dpi=150)  # 設置 dpi 來提高圖像解析度