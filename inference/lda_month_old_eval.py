""" LDA の月齢の予測精度を評価 """

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import sys
sys.path.append('../')
from cython_folder.LDA.lda import LDA

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
    
    # LDA
    d_test = range(i, i+n)
    d_train = list(set(range(N_tdv.shape[1])) ^ set(d_test))
    z_train = pd.read_pickle(f'../pickle_folder/paper/lda_train_{i}_{i+n}_z_dk_var.pkl')
    y_true_train = Age_td[0:1, d_train].reshape(-1, 1)
    y_true_test = Age_td[0:1, d_test].reshape(-1, 1)
    
    # 学習
    lr = LinearRegression()
    lr.fit(z_train, y_true_train)
    y_pred_train = lr.predict(z_train)
    # 訓練データ評価
    mse_train = mean_squared_error(y_true_train, y_pred_train)
    print(f'lda_mse_train_var_{i}_{i+n}: {mse_train}')
    mse_train_all += mse_train
    
    # テストデータの説明変数構築
    lda_test = LDA(
                    K = 25,
                    L = 1,
                    N_tdv = N_tdv[0:1, d_test, :],
                    Sex_ds = Sex_ds[d_test, :],
                    Con_ds = Con_ds[d_test, :],
                    Q_td = Q_td[0:1, d_test],
                    Age_td = Age_td[0:1, d_test]
                    )
    lda_test.lda_estimate(iteration=1000, sample_size=500)
    y_pred_test = lr.predict(lda_test.z_tdk_var[0])
    # テストデータ評価
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    print(f'lda_mse_test_var_{i}_{i+n}: {mse_test}')
    mse_test_all += mse_test
    
    # 結果を保存
    # ユーザー情報
    pd.to_pickle(d_test, f'../pickle_folder/paper/d_test_{i}_{i+n}.pkl')
    pd.to_pickle(d_train, f'../pickle_folder/paper/d_train_{i}_{i+n}.pkl')
    # lda
    pd.to_pickle(np.array(lda_test.z_tdk_var[0]), f'../pickle_folder/paper/lda_test_{i}_{i+n}_z_dk_var.pkl')
# end for

mse_train_all /= k
mse_test_all /= k

print(f'mse_train_all: {mse_train_all}')
print(f'mse_test_all: {mse_test_all}')