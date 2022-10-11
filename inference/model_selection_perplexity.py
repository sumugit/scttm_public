""" モデル選択 """

import sys
sys.path.append('../')
from cython_folder.evaluation.eval import Eval
import pandas as pd
from cython_folder.SCTTM.scttm import SCTTM

# 単語の訓練, テストデータ 
N_tdv_train = pd.read_pickle('../pickle_folder/paper/questions_bow_arr_paper_train.pkl')
N_tdv_test = pd.read_pickle('../pickle_folder/paper/questions_bow_arr_paper_test.pkl')
# 補助情報データ
Sex_ds = pd.read_pickle('../pickle_folder/paper/children_sex_count_paper.pkl')
Con_ds = pd.read_pickle('../pickle_folder/paper/additional_info_paper.pkl')
Q_tds = pd.read_pickle('../pickle_folder/paper/questions_meta_arr_paper.pkl')
Q_td = Q_tds[:,:,2].astype('uint32')
Age_td = Q_tds[:,:,1]

# 比較トピック数
topics = range(5, 40, 5)

for k in topics:
    # 訓練データで SCTTM 推論
    scttm = SCTTM(K=k, L=1, N_tdv=N_tdv_train[0:1], Sex_ds=Sex_ds, Con_ds=Con_ds, Q_td=Q_td[0:1], Age_td=Age_td[0:1])
    # scttm_estimate の引数: 反復数
    scttm.scttm_estimate(iteration=1000, sample_size=500)
    
    # トピック数を評価
    topic_eval = Eval(theta_tdk=scttm.theta_tdk, phi_tkv=scttm.phi_tkv, N_tdv=N_tdv_test, Age_td=Age_td)
    perplexity = topic_eval.model_select_perplexity()
    print(f'トピック: {k}, perplexity: {perplexity}')
    