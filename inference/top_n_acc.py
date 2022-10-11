import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('../')
from cython_folder.evaluation.eval import Eval


# データロード
N_tdv = pd.read_pickle('../pickle_folder/paper/questions_bow_arr_paper.pkl')
# lda の Top-N-Accuracy
# lda_theta_tdk = pd.read_pickle('../pickle_folder/paper/all_lda/lda_theta.pkl')
# lda_phi_tkv = pd.read_pickle('../pickle_folder/paper/all_lda/lda_phi.pkl')
# scttm の Top-N-Accuracy
scttm_theta_tdk = pd.read_pickle('../pickle_folder/paper/all_ttm/ttm_theta.pkl')
scttm_phi_tkv = pd.read_pickle('../pickle_folder/paper/all_ttm/ttm_phi.pkl')

lda_lst = []
scttm_lst = []

# 質問回数 5, 10, 20 回のユーザー
for i in [5, 10, 20]:
    # n
    for n in range(1, 4):
        # LDA の Top-N-accuracy
        """
        if i == 5:
            theta_tdk = lda_theta_tdk[0:5, 0:250, :]
            phi_tkv = lda_phi_tkv[0:5, :, :]
            lda = Eval(
                    theta_tdk=theta_tdk,
                    phi_tkv=phi_tkv,
                    N_tdv=N_tdv[0:5, 0:250, :],
                    )
        elif i == 10:
            theta_tdk = lda_theta_tdk[0:10, 250:500, :]
            phi_tkv = lda_phi_tkv[0:10, :, :]
            lda = Eval(
                    theta_tdk=theta_tdk,
                    phi_tkv=phi_tkv,
                    N_tdv=N_tdv[0:10, 250:500, :],
                    )
        else:
            theta_tdk = lda_theta_tdk[0:20, 500:750, :]
            phi_tkv = lda_phi_tkv[0:20, :, :]
            lda = Eval(
                    theta_tdk=theta_tdk,
                    phi_tkv=phi_tkv,
                    N_tdv=N_tdv[0:20, 500:750, :],
                    )
        lda.top_N_accuracy(n)
        lda_lst.append(np.array(lda.accuracy))
        print(f'lda_{n}_{i}: {np.array(lda.accuracy)}')
        print(f'average_lda_{n}_{i}: {np.sum(np.array(lda.accuracy))/(i-1)}')
        """
        # SCTTM の Top-N-accuracy
        if i == 5:
            theta_tdk = scttm_theta_tdk[0:5, 0:250, :]
            phi_tkv = scttm_phi_tkv[0:5, :, :]
            scttm = Eval(
                    theta_tdk=theta_tdk,
                    phi_tkv=phi_tkv,
                    N_tdv=N_tdv[0:5, 0:250, :],
                    )
        elif i == 10:
            theta_tdk = scttm_theta_tdk[0:10, 250:500, :]
            phi_tkv = scttm_phi_tkv[0:10, :, :]
            scttm = Eval(
                    theta_tdk=theta_tdk,
                    phi_tkv=phi_tkv,
                    N_tdv=N_tdv[0:10, 250:500, :],
                    )
        else:
            theta_tdk = scttm_theta_tdk[0:20, 500:750, :]
            phi_tkv = scttm_phi_tkv[0:20, :, :]
            scttm = Eval(
                    theta_tdk=theta_tdk,
                    phi_tkv=phi_tkv,
                    N_tdv=N_tdv[0:20, 500:750, :],
                    )       
        scttm.top_N_accuracy(n)
        scttm_lst.append(np.array(scttm.accuracy))
        print(f'ttm_{n}_{i}: {np.array(scttm.accuracy)}')
        print(f'average_ttm_{n}_{i}: {np.sum(np.array(scttm.accuracy))/(i-1)}')
    # end for
# end for


# with open('../pickle_folder/paper/lda_lst.pickle', 'wb') as f:
#    pickle.dump(lda_lst, f)

with open('../pickle_folder/paper/ttm_lst.pickle', 'wb') as f:
    pickle.dump(scttm_lst, f)