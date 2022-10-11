""" 勾配ブースティングによる月齢の予測精度を評価 """

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error

import sys
sys.path.append('../')

# データロード
N_tdv = pd.read_pickle('../pickle_folder/paper/questions_bow_arr_paper.pkl')
Sex_ds = pd.read_pickle('../pickle_folder/paper/children_sex_count_paper.pkl')
Con_ds = pd.read_pickle('../pickle_folder/paper/additional_info_paper.pkl')
Q_tds = pd.read_pickle('../pickle_folder/paper/questions_meta_arr_paper.pkl')
Q_td = Q_tds[:,:,2].astype('uint32')
Age_td = Q_tds[:,:,1]

k = 5                        # 分割数
n = int(N_tdv.shape[1]/k)    # 各区間のデータ数
mse_train_all = 0.0            # 5 分割の lda の訓練 mse 平均
mse_test_all = 0.0            # 5 分割の lda のテスト mse 平均

for i in range(0, N_tdv.shape[1], n):
    
    # データロード
    d_test = range(i, i+n)
    d_train = list(set(range(N_tdv.shape[1])) ^ set(d_test))
    z_train = pd.read_pickle(f'../pickle_folder/paper/lda_train_{i}_{i+n}_z_dk_var.pkl')
    z_test = pd.read_pickle(f'../pickle_folder/paper/lda_test_{i}_{i+n}_z_dk_var.pkl')
    y_true_train = Age_td[0:1, d_train].reshape(-1, 1)
    y_true_train = np.reshape(y_true_train,(-1))
    y_true_test = Age_td[0:1, d_test].reshape(-1, 1)
    y_true_test = np.reshape(y_true_test,(-1))
    
    # 学習: RandomForest
    rg = RFR(n_jobs=-1, random_state=1998)
    rg.fit(z_train, y_true_train)
    y_pred_train = rg.predict(z_train)
    # 訓練データ評価
    mse_train = mean_squared_error(y_true_train, y_pred_train)
    print(f'RandomForest_mse_train_var_{i}_{i+n}: {mse_train}')
    mse_train_all += mse_train
    
    # テストデータ評価
    y_pred_test = rg.predict(z_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    print(f'RandomForest_mse_test_{i}_{i+n}: {mse_test}')
    mse_test_all += mse_test    
# end for

mse_train_all /= k
mse_test_all /= k

print(f'mse_train_all: {mse_train_all}')
print(f'mse_test_all: {mse_test_all}')