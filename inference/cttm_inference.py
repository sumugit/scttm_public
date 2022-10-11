import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append('../')
from cython_folder.cTTM.cttm import CTTM
PATH = '../pickle_folder/paper/all_cttm'

# 単語集合
""" コンペ """ 
"""
N_tdv = pd.read_pickle('../pickle_folder/q30/questions_bow_arr.pkl')
Sex_ds = pd.read_pickle('../pickle_folder/auxiliary/children_sex_count.pkl')
Con_ds = pd.read_pickle('../pickle_folder/auxiliary/additional_info.pkl')
Q_tds = pd.read_pickle('../pickle_folder/auxiliary/questions_meta_arr.pkl')
Q_td = Q_tds[:,:,2].astype('uint32')
"""

""" 論文 """
N_tdv = pd.read_pickle('../pickle_folder/paper/questions_bow_arr_paper.pkl')
Sex_ds = pd.read_pickle('../pickle_folder/paper/children_sex_count_paper.pkl')
Con_ds = pd.read_pickle('../pickle_folder/paper/additional_info_paper.pkl')
Q_tds = pd.read_pickle('../pickle_folder/paper/questions_meta_arr_paper.pkl')
Q_td = Q_tds[:,:,2].astype('uint32')

# 学習
# k: トピック数, L: 依存時間数, N_tdv: 時刻・ユーザー毎のBOW行列,
# Sex_ds:ユーザー毎の子供性別のBOW行列, Con_ds: ユーザー属性情報, Q_td: 質問カテゴリ情報
# Age_td: ユーザー毎の子供の月齢
cttm = CTTM(K=25, L=1, N_tdv=N_tdv, Sex_ds=Sex_ds, Con_ds=Con_ds, Q_td=Q_td)
# cttm_estimate の引数: 反復数
cttm.cttm_estimate(iteration=1000, sample_size=500)


# データ保存
pd.to_pickle(np.array(cttm.theta_tdk), os.path.join(PATH, 'cttm_theta.pkl'))
pd.to_pickle(np.array(cttm.phi_tkv), os.path.join(PATH, 'cttm_phi.pkl'))
pd.to_pickle(np.array(cttm.psi_sex_tks), os.path.join(PATH, 'cttm_psi_sex.pkl'))
pd.to_pickle(np.array(cttm.psi_chi_tks), os.path.join(PATH, 'cttm_psi_chi.pkl'))
pd.to_pickle(np.array(cttm.psi_gen_tks), os.path.join(PATH, 'cttm_psi_gen.pkl'))
pd.to_pickle(np.array(cttm.psi_pre_tks), os.path.join(PATH, 'cttm_psi_pre.pkl'))
pd.to_pickle(np.array(cttm.psi_q_tks), os.path.join(PATH, 'cttm_psi_q.pkl'))
pd.to_pickle(np.array(cttm.alpha_tdl), os.path.join(PATH, 'cttm_alpha.pkl'))
pd.to_pickle(np.array(cttm.beta_tkl), os.path.join(PATH, 'cttm_beta.pkl'))
pd.to_pickle(np.array(cttm.gamma_sex_t), os.path.join(PATH, 'cttm_gamma_sex_t.pkl'))
pd.to_pickle(np.array(cttm.gamma_chi_t), os.path.join(PATH, 'cttm_gamma_chi_t.pkl'))
pd.to_pickle(np.array(cttm.gamma_gen_t), os.path.join(PATH, 'cttm_gamma_gen_t.pkl'))
pd.to_pickle(np.array(cttm.gamma_pre_t), os.path.join(PATH, 'cttm_gamma_pre_t.pkl'))
pd.to_pickle(np.array(cttm.gamma_q_t), os.path.join(PATH, 'cttm_gamma_q_t.pkl'))
