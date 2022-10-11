""" SCTTM における月齢の予測精度を評価 """

import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('../')
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
mse_k_scttm = 0.0            # 5 分割の scttm のテスト mse 平均

for i in range(0, N_tdv.shape[1], n):
    
    d_test = range(i, i+n)
    eta_k = pd.read_pickle(f'../pickle_folder/paper/scttm_train_{i}_{i+n}_eta_k.pkl')
    sigma = pd.read_pickle(f'../pickle_folder/paper/scttm_train_{i}_{i+n}_sigma.pkl')
    
    # SCTTM_テストデータ
    scttm_test = SCTTM(
                    K = 25,
                    L = 1,
                    N_tdv = N_tdv[0:1, d_test, :],
                    Sex_ds = Sex_ds[d_test, :],
                    Con_ds = Con_ds[d_test, :],
                    Q_td = Q_td[0:1, d_test],
                    Age_td = Age_td[0:1, d_test],
                    eta_k = eta_k,
                    sigma = sigma
                    )
    
    # 評価: scttm
    scttm_test.scttm_estimate(iteration=1000, sample_size=500)
    print(f'scttm_{i}_{i+n}: mse_test = {scttm_test.mse}')
    mse_k_scttm += scttm_test.mse    
# end for

mse_k_scttm /= k
print(f'scttm_mse_test: {mse_k_scttm}')