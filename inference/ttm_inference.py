import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append('../')
from cython_folder.TTM.ttm import TTM
PATH = '../pickle_folder/paper/all_ttm'

# 単語集合
""" コンペ """ 
"""
N_tdv = pd.read_pickle('../pickle_folder/q30/questions_bow_arr.pkl')
"""

""" 論文 """
N_tdv = pd.read_pickle('../pickle_folder/paper/questions_bow_arr_paper.pkl')

# 学習
# k: トピック数, L: 依存時間数, N_tdv: 時刻・ユーザー毎のBOW行列,
# Sex_ds:ユーザー毎の子供性別のBOW行列, Con_ds: ユーザー属性情報, Q_td: 質問カテゴリ情報
# Age_td: ユーザー毎の子供の月齢
ttm = TTM(K=25, L=1, N_tdv=N_tdv)
# ttm_estimate の引数: 反復数
ttm.ttm_estimate(iteration=1000, sample_size=500)


# データ保存
pd.to_pickle(np.array(ttm.theta_tdk), os.path.join(PATH, 'ttm_theta.pkl'))
pd.to_pickle(np.array(ttm.phi_tkv), os.path.join(PATH, 'ttm_phi.pkl'))
pd.to_pickle(np.array(ttm.alpha_tdl), os.path.join(PATH, 'ttm_alpha.pkl'))
pd.to_pickle(np.array(ttm.beta_tkl), os.path.join(PATH, 'ttm_beta.pkl'))
