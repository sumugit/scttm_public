""" SCTTM と LDA において月齢を k 分割 Closs-Validation で学習 """

import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('../')
from cython_folder.LDA.lda import LDA
from cython_folder.SCTTM.scttm import SCTTM

# データロード
N_tdv = pd.read_pickle('../pickle_folder/paper/questions_bow_arr_paper.pkl')
Sex_ds = pd.read_pickle('../pickle_folder/paper/children_sex_count_paper.pkl')
Con_ds = pd.read_pickle('../pickle_folder/paper/additional_info_paper.pkl')
Q_tds = pd.read_pickle('../pickle_folder/paper/questions_meta_arr_paper.pkl')
Q_td = Q_tds[:,:,2].astype('uint32')
Age_td = Q_tds[:,:,1]

k = 5                        # 分割数
n = int(N_tdv.shape[1]/k)    # 各区間のデータ数
mse_k_scttm = 0.0            # 5 分割の scttm の訓練 mse 平均

for i in range(0, N_tdv.shape[1], n):
    
    d_test = range(i, i+n)
    d_train = list(set(range(N_tdv.shape[1])) ^ set(d_test))
    # SCTTM_訓練データ
    scttm_train = SCTTM(
                    K = 25,
                    L = 1,
                    N_tdv = N_tdv[0:1, d_train, :],
                    Sex_ds = Sex_ds[d_train, :],
                    Con_ds = Con_ds[d_train, :],
                    Q_td = Q_td[0:1, d_train],
                    Age_td = Age_td[0:1, d_train],
                    )
    # LDA_訓練データ
    lda_train = LDA(
                    K = 25,
                    L = 1,
                    N_tdv = N_tdv[0:1, d_train, :],
                    Sex_ds = Sex_ds[d_train, :],
                    Con_ds = Con_ds[d_train, :],
                    Q_td = Q_td[0:1, d_train],
                    Age_td = Age_td[0:1, d_train]
                    )
    # 推論: scttm
    scttm_train.scttm_estimate(iteration=1000, sample_size=500)
    print(f'scttm_{i}_{i+n}: mse_train = {scttm_train.mse}')
    mse_k_scttm += scttm_train.mse
    # 推論: lda
    lda_train.lda_estimate(iteration=1000, sample_size=500)
    
    # 結果を保存
    # ユーザー情報
    pd.to_pickle(d_test, f'../pickle_folder/paper/d_test_{i}_{i+n}.pkl')
    pd.to_pickle(d_train, f'../pickle_folder/paper/d_train_var_{i}_{i+n}.pkl')
    # scttm
    pd.to_pickle(np.array(scttm_train.eta_k), f'../pickle_folder/paper/scttm_train_{i}_{i+n}_eta_k.pkl')
    pd.to_pickle(scttm_train.sigma, f'../pickle_folder/paper/scttm_train_{i}_{i+n}_sigma.pkl')
    # lda
    pd.to_pickle(np.array(lda_train.z_tdk_var[0]), f'../pickle_folder/paper/lda_train_{i}_{i+n}_z_dk_var.pkl')
# end for

mse_k_scttm /= k
print(f'scttm_mse_train: {mse_k_scttm}')