import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('../')
from cython_folder.LDA.lda import LDA

# 単語集合
""" コンペ """ 
"""
N_tdv = pd.read_pickle('../pickle_folder/q30/questions_bow_arr.pkl')
Sex_ds = pd.read_pickle('../pickle_folder/auxiliary/children_sex_count.pkl')
Con_ds = pd.read_pickle('../pickle_folder/auxiliary/additional_info.pkl')
Q_tds = pd.read_pickle('../pickle_folder/auxiliary/questions_meta_arr.pkl')
Q_td = Q_tds[:,:,2].astype('uint32')
Age_td = Q_tds[:,:,1]
"""

""" 論文 """
N_tdv = pd.read_pickle('../pickle_folder/paper/questions_bow_arr_paper.pkl')
Sex_ds = pd.read_pickle('../pickle_folder/paper/children_sex_count_paper.pkl')
Con_ds = pd.read_pickle('../pickle_folder/paper/additional_info_paper.pkl')
Q_tds = pd.read_pickle('../pickle_folder/paper/questions_meta_arr_paper.pkl')
Q_td = Q_tds[:,:,2].astype('uint32')
Age_td = Q_tds[:,:,1]

# 学習
# k: トピック数, L: 依存時間数, N_tdv: 時刻・ユーザー毎のBOW行列,
# Sex_ds:ユーザー毎の子供性別のBOW行列, Con_ds: ユーザー属性情報, Q_td: 質問カテゴリ情報
# Age_td: ユーザー毎の子供の月齢
lda = LDA(K=25, L=1, N_tdv=N_tdv, Sex_ds=Sex_ds, Con_ds=Con_ds, Q_td=Q_td, Age_td=Age_td)
# lda_estimate の引数: 反復数
lda.lda_estimate(iteration=1000, sample_size=500)


# データ保存
pd.to_pickle(np.array(lda.theta_tdk), '../pickle_folder/paper/all_lda/lda_theta.pkl')
pd.to_pickle(np.array(lda.phi_tkv), '../pickle_folder/paper/all_lda/lda_phi.pkl')