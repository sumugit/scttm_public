"""Correspondence TTM with Cython"""

from libcpp.vector cimport vector
import numpy as np
import sys
import math
import cython
cimport numpy as np
from scipy.special import digamma
from libc.math cimport pow as c_pow
from threading import Thread
from cython.parallel import prange
from cython.parallel import threadid
from tqdm import tqdm
from sklearn.metrics import r2_score
import warnings

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval)

srand48(1234)
warnings.filterwarnings('error')

# 境界チェックを無視
@cython.boundscheck(False)
@cython.wraparound(False)

cdef class CTTM():

    cdef public unsigned int K, D, V, S, T, L, Chi, Gen, Pre, Q
    cdef public vector[vector[vector[int]]] N_tdv
    cdef public vector[vector[vector[int]]] N_tdk
    cdef public vector[vector[vector[int]]] N_tkv
    cdef public vector[vector[int]] N_tk
    cdef public vector[vector[int]] N_td

    # 補助情報: 子供の性別
    cdef public vector[vector[int]] Sex_ds
    cdef public vector[vector[vector[int]]] Sex_tdk
    cdef public vector[vector[vector[int]]] Sex_tks
    cdef public vector[vector[int]] Sex_tk
    cdef public vector[int] Sex_d

    # 補助情報: 子供の数, 世代 id, 都道府県 id
    cdef public vector[vector[int]] Con_ds
    cdef public unsigned int Con_d

    # 補助情報: 子供の数
    cdef public vector[vector[vector[int]]] Chi_tdk
    cdef public vector[vector[vector[int]]] Chi_tks
    cdef public vector[vector[int]] Chi_tk

    # 補助情報: 世代 id
    cdef public vector[vector[vector[int]]] Gen_tdk
    cdef public vector[vector[vector[int]]] Gen_tks
    cdef public vector[vector[int]] Gen_tk

    # 補助情報: 都道府県 id
    cdef public vector[vector[vector[int]]] Pre_tdk
    cdef public vector[vector[vector[int]]] Pre_tks
    cdef public vector[vector[int]] Pre_tk

    # 補助情報: 質問カテゴリ id
    cdef public vector[vector[int]] Q_td
    cdef public vector[vector[vector[int]]] Q_tdk
    cdef public vector[vector[vector[int]]] Q_tks
    cdef public vector[vector[int]] Q_tk
    cdef public unsigned int Q_d

    # トピック
    cdef public int[:,:,:] z_tdn
    cdef public int[:,:,:] y_sex_tdm
    cdef public int[:,:,:] y_chi_tdm
    cdef public int[:,:,:] y_gen_tdm
    cdef public int[:,:,:] y_pre_tdm
    cdef public int[:,:,:] y_q_tdm

    # 推論するパラメータ
    cdef public double[:] first_alpha
    cdef public double first_beta
    cdef public double[:] gamma_sex_t
    cdef public double[:] gamma_chi_t
    cdef public double[:] gamma_gen_t
    cdef public double[:] gamma_pre_t
    cdef public double[:] gamma_q_t
    cdef public double[:,:,:] alpha_tdl
    cdef public double[:,:,:] beta_tkl
    cdef public double[:,:,:] theta_tdk
    cdef public double[:,:,:] phi_tkv
    cdef public double[:,:,:] psi_sex_tks
    cdef public double[:,:,:] psi_chi_tks
    cdef public double[:,:,:] psi_gen_tks
    cdef public double[:,:,:] psi_pre_tks
    cdef public double[:,:,:] psi_q_tks

    # 確率分布
    cdef public double[:,:] p_ztd_n
    cdef public double[:,:] p_ytd_m


    def __init__(self, unsigned int K, unsigned int L, vector[vector[vector[int]]] N_tdv,
                vector[vector[int]] Sex_ds, vector[vector[int]] Con_ds,
                vector[vector[int]] Q_td):
        """ 初期化 """
        # N : T×D×V

        cdef unsigned int t
        cdef unsigned int d

        self.K = K # トピック数
        self.L = L # 依存する過去の期間
        self.N_tdv = N_tdv # 単語のBOW行列
        self.V = self.N_tdv.at(0).at(0).size()  # 単語数
        self.S = 3 # 子供の性別の要素数
        self.Chi = 7 # 子供数
        self.Gen = 8 # 世代数
        self.Pre = 48 # 都道府県数
        self.Q = 16 # 質問カテゴリ数
        self.Sex_ds = Sex_ds # ユーザー属性: Sex の BOW 行列  
        self.Con_ds = Con_ds # 他のユーザー属性
        self.Q_td = Q_td # ユーザーの質問カテゴリ
        self.T = self.N_tdv.size() # 期間
        self.D = self.N_tdv.at(0).size()  # 文書数
        self.N_td = np.zeros(shape=(self.T, self.D)).astype('uint32') # 時刻 t , ユーザー d の語彙数
        self.Sex_d = np.zeros(shape=(self.D)).astype('uint32') # ユーザー d の子供の性別カウント
        self.Con_d = 1 # ユーザー d の子供数, 世代, 都道府県 のカウント (1つしか登録できないので 1)
        self.Q_d = 1 # ユーザーの質問カテゴリ数 (1回の質問で1つのみ)
        
        for t in range(self.T):
            for d in range(self.D):
                self.N_td[t][d] = sum(self.N_tdv[t][d])
        
        for d in range(self.D):
            self.Sex_d[d] = sum(self.Sex_ds[d])

        # ハイパーパラメータ alpha, betaの初期化
        # t = 0 の時のハイパラのみ別処理
        self.first_alpha = np.full((self.K), 2.0) # 非一様
        self.first_beta = 2.0 # 一様
        self.gamma_sex_t = np.full((self.T), 2.0) # 一様
        self.gamma_chi_t = np.full((self.T), 2.0) # 一様
        self.gamma_gen_t = np.full((self.T), 2.0) # 一様
        self.gamma_pre_t = np.full((self.T), 2.0) # 一様
        self.gamma_q_t = np.full((self.T), 2.0) # 一様
        self.alpha_tdl = np.full((self.T, self.D, self.L), 2.0) # 一様
        self.beta_tkl = np.full((self.T, self.K, self.L), 2.0) # 一様

        # トピック分布, 単語分布, 補助情報分布の初期化
        self.theta_tdk = np.full((self.T, self.D, self.K), 1.0/self.K)
        self.phi_tkv = np.full((self.T, self.K, self.V), 1.0/self.V)
        self.psi_sex_tks = np.full((self.T, self.K, self.S), 1.0/self.S)
        self.psi_chi_tks = np.full((self.T, self.K, self.Chi), 1.0/self.Chi)
        self.psi_gen_tks = np.full((self.T, self.K, self.Gen), 1.0/self.Gen)
        self.psi_pre_tks = np.full((self.T, self.K, self.Pre), 1.0/self.Pre)
        self.psi_q_tks = np.full((self.T, self.K, self.Q), 1.0/self.Q)

        # 時刻 t, ユーザー d, 語彙 v に割り当てられたトピック (z_tdn) の集合
        self.z_tdn = np.full((self.T, self.D, self.V), -1).astype('int32')
        # 時刻 t , ユーザー d においてトピック k が割り当てられた単語数の集合
        self.N_tdk = np.zeros(shape=(self.T, self.D, self.K)).astype('uint32')
        # 時刻 t のユーザー全体でトピック k が割り当てられた語彙 v の出現回数の集合
        self.N_tkv = np.zeros(shape=(self.T, self.K, self.V)).astype('uint32')
        # 時刻 t の全ユーザーでトピック k が割り当てられた単語数のベクトル
        self.N_tk = np.zeros(shape=(self.T, self.K)).astype('uint32')
        # 時刻 t の単語トピックのサンプリング確率
        self.p_ztd_n = np.zeros(shape=(self.T, self.K))

        # 時刻 t, ユーザー d, 子供の性別 m に割り当てられたトピック (y_sex_tdm) の集合
        self.y_sex_tdm = np.full((self.T, self.D, self.S), -1).astype('int32')
        # 時刻 t, ユーザー d においてトピック k に割り当てられた子供の性別数の集合
        self.Sex_tdk = np.zeros(shape=(self.T, self.D, self.K)).astype('uint32')
        # 時刻 t のユーザー全体でトピック k が割り当てられた子供の性別 s の出現回数の集合
        self.Sex_tks = np.zeros(shape=(self.T, self.K, self.S)).astype('uint32')
        # 時刻 t の全ユーザーでトピック k が割り当てられた子供の性別数のベクトル
        self.Sex_tk = np.zeros(shape=(self.T, self.K)).astype('uint32')
        # 時刻 t の子供の性別トピックのサンプリング確率
        self.p_ytd_m = np.zeros(shape=(self.T, self.K))

        # 時刻 t, ユーザー d, 子供の数 m に割り当てられたトピック (y_chi_tdm) の集合
        self.y_chi_tdm = np.full((self.T, self.D, self.Chi), -1).astype('int32')
        self.Chi_tdk = np.zeros(shape=(self.T, self.D, self.K)).astype('uint32')
        self.Chi_tks = np.zeros(shape=(self.T, self.K, self.Chi)).astype('uint32')
        self.Chi_tk = np.zeros(shape=(self.T, self.K)).astype('uint32')

        # 時刻 t, ユーザー d, 世代 m に割り当てられたトピック (y_gen_tdm) の集合
        self.y_gen_tdm = np.full((self.T, self.D, self.Chi), -1).astype('int32')
        self.Gen_tdk = np.zeros(shape=(self.T, self.D, self.K)).astype('uint32')
        self.Gen_tks = np.zeros(shape=(self.T, self.K, self.Gen)).astype('uint32')
        self.Gen_tk = np.zeros(shape=(self.T, self.K)).astype('uint32')

        # 時刻 t, ユーザー d, 都道府県 m に割り当てられたトピック (y_pre_tdm) の集合
        self.y_pre_tdm = np.full((self.T, self.D, self.Pre), -1).astype('int32')
        self.Pre_tdk = np.zeros(shape=(self.T, self.D, self.K)).astype('uint32')
        self.Pre_tks = np.zeros(shape=(self.T, self.K, self.Pre)).astype('uint32')
        self.Pre_tk = np.zeros(shape=(self.T, self.K)).astype('uint32')

        # 時刻 t, ユーザー d, 質問カテゴリ m に割り当てられたトピック (y_q_tdm) の集合
        self.y_q_tdm = np.full((self.T, self.D, self.Q), -1).astype('int32')
        self.Q_tdk = np.zeros(shape=(self.T, self.D, self.K)).astype('uint32')
        self.Q_tks = np.zeros(shape=(self.T, self.K, self.Q)).astype('uint32')
        self.Q_tk = np.zeros(shape=(self.T, self.K)).astype('uint32')
    
    
    def cttm_estimate(self, unsigned int iteration, unsigned int sample_size):
        """崩壊型ギブスサンプリングで推定"""

        cdef unsigned int t, d, k, l, _l, v, s, num_chi, gen_id, pre_id, q_id, index
        cdef unsigned int itr
        cdef int topic, new_topic
        cdef double tmp_alpha_td, tmp_beta_tk, tmp_sum, mse
        cdef double[:] tmp_first_alpha, eta_right, eta_temp, pred_y
        cdef double[:,:] eta_left
        cdef double[:,:,:] tmp_alpha, tmp_beta
        cdef double tmp_alpha_numer, tmp_alpha_denom, tmp_alpha_sum, tmp_first_alpha_sum
        cdef double tmp_beta_numer, tmp_beta_denom, tmp_beta_sum, tmp_first_beta
        cdef double p_alpha, p_beta_numer, p_beta_denom, p_gamma_numer, p_gamma_denom, p_aid
        cdef double alpha_theta, beta_phi
        cdef double alpha_numer1, alpha_numer, alpha_denom1, alpha_denom, alpha_numer_digamma
        cdef double beta_numer1, beta_numer, beta_denom1, beta_denom, beta_numer_digamma
        cdef double gamma_numer1, gamma_numer, gamma_denom1, gamma_denom, tmp_gamma
        cdef list del_list

        # 期間でループ
        for t in range(self.T):
            # repeat
            for itr in tqdm(range(iteration)):

                # ユーザー数でループ
                for d in range(self.D):
                    """単語トピックのサンプリング"""
                    # ユーザー d の語彙数でループ
                    for v in range(self.V):
                        for num in range(self.N_tdv[t][d][v]):
                        
                            # 時刻 t におけるユーザー d の語彙 v に割り当てられたトピックを取得
                            topic = self.z_tdn[t][d][v]
                            

                            # 初回を飛ばす処理
                            if topic >= 0:
                                # カウントから topic の割当分を引く
                                self.N_tdk[t][d][topic] -= 1
                                self.N_tkv[t][topic][v] -= 1
                                self.N_tk[t][topic] -= 1
                            # end if


                            # 最初の時刻のサンプリング式
                            if t == 0:
                                for k in range(self.K):
                                    # サンプリング確率を計算
                                    p_alpha = self.N_tdk[t][d][k] + self.first_alpha[k]
                                    p_beta_numer = self.N_tkv[t][k][v] + self.first_beta
                                    p_beta_denom = self.N_tk[t][k] + self.first_beta * self.V
                                    index = self.Sex_tdk[t][d][k] + self.Chi_tdk[t][d][k] + self.Gen_tdk[t][d][k] + self.Pre_tdk[t][d][k] + self.Q_tdk[t][d][k]
                                    if self.N_tdk[t][d][k] == 0 and index > 0: p_aid = 100000 # 擬似的に無限大を作る (np.infを使うとnanが返る)
                                    elif self.N_tdk[t][d][k] == 0 and index == 0: p_aid = 1.0
                                    else: p_aid = c_pow((self.N_tdk[t][d][k] + 1.0)/(self.N_tdk[t][d][k]), index)
                                    self.p_ztd_n[t][k] = p_alpha * (p_beta_numer / p_beta_denom) * p_aid
                                # end for k
                        
                            
                            # 最初から数えて L 時刻目までのサンプリング式
                            elif self.L > t > 0:
                                for k in range(self.K):
                                    alpha_theta = sum([self.alpha_tdl[t][d][l] * self.theta_tdk[t-l-1][d][k] for l in range(t)])
                                    beta_phi = sum([self.beta_tkl[t][k][l] * self.phi_tkv[t-l-1][k][v] for l in range(t)])

                                    p_alpha = self.N_tdk[t][d][k] + alpha_theta
                                    p_beta_numer = self.N_tkv[t][k][v] + beta_phi
                                    p_beta_denom = self.N_tk[t][k] + sum(self.beta_tkl[t][k][:t])
                                    index = self.Q_tdk[t][d][k]
                                    if self.N_tdk[t][d][k] == 0 and index > 0: p_aid = 100000
                                    elif self.N_tdk[t][d][k] == 0 and index == 0: p_aid = 1.0
                                    else: p_aid = c_pow((self.N_tdk[t][d][k] + 1.0)/(self.N_tdk[t][d][k]), index)
                                    self.p_ztd_n[t][k] = p_alpha * (p_beta_numer / p_beta_denom) * p_aid
                                # end for k


                            # L 時刻目以降のサンプリング式
                            else:
                                for k in range(self.K):
                                    alpha_theta = sum([self.alpha_tdl[t][d][l] * self.theta_tdk[t-l-1][d][k] for l in range(self.L)])
                                    beta_phi = sum([self.beta_tkl[t][k][l] * self.phi_tkv[t-l-1][k][v] for l in range(self.L)])

                                    p_alpha = self.N_tdk[t][d][k] + alpha_theta
                                    p_beta_numer = self.N_tkv[t][k][v] + beta_phi
                                    p_beta_denom = self.N_tk[t][k] + sum(self.beta_tkl[t][k])
                                    index = self.Q_tdk[t][d][k]
                                    if self.N_tdk[t][d][k] == 0 and index > 0: p_aid = 100000
                                    elif self.N_tdk[t][d][k] == 0 and index == 0: p_aid = 1.0
                                    else: p_aid = c_pow((self.N_tdk[t][d][k] + 1.0)/(self.N_tdk[t][d][k]), index)
                                    self.p_ztd_n[t][k] = p_alpha * (p_beta_numer / p_beta_denom) * p_aid
                                # end for k
                            # end if

                            # 減らした分を元に戻す
                            if topic >= 0:
                                # カウントから topic の割当分を引く
                                self.N_tdk[t][d][topic] += 1
                                self.N_tkv[t][topic][v] += 1
                                self.N_tk[t][topic] += 1
                            
                            
                            tmp_sum = sum(self.p_ztd_n[t])
                            if math.isclose(tmp_sum, 0.0):
                                print('ユーザー ', d,' で欠測データが続きました.')
                                for k in range(self.K):
                                    self.p_ztd_n[t][k] = 1.0/self.K
                                # end for k
                            else:
                                for k in range(self.K):
                                    self.p_ztd_n[t][k] /= tmp_sum
                                # end for k
                            
                            # トピックをサンプリング
                            try:
                                self.z_tdn[t][d][v] = np.random.multinomial(n=1, pvals=self.p_ztd_n[t], size=1).argmax()
                            except:
                                print(np.array(self.p_ztd_n[t]))
                            # カウントに新たに割り当てたトピックの分を加える
                            new_topic = self.z_tdn[t][d][v]
                            self.N_tdk[t][d][new_topic] += 1
                            self.N_tkv[t][new_topic][v] += 1
                            self.N_tk[t][new_topic] += 1
                        # end for num
                    # end for v
                # end for d

                # 補助情報トピックの推定
                if t == 0:

                    for d in range(self.D):   
                        # 単語数 0 のユーザーは除く
                        if self.N_td[t][d] == 0: continue                          
                        """ 補助情報: 子供の性別に応じたトピックのサンプリング """
                        # 補助情報でループ
                        for s in range(self.S):
                            for num in range(self.Sex_ds[d][s]):
                            
                                # 時刻 t におけるユーザー d の補助情報 s に割り当てられたトピックを取得
                                topic = self.y_sex_tdm[t][d][s]

                                # 初回を飛ばす処理
                                if topic >= 0:
                                    # カウントから topic の割当分を引く
                                    self.Sex_tks[t][topic][s] -= 1
                                    self.Sex_tk[t][topic] -= 1
                                # end if

                                # 時刻に依らず同じサンプリング式
                                for k in range(self.K):
                                    # サンプリング確率を計算
                                    p_gamma_numer = self.Sex_tks[t][k][s] + self.gamma_sex_t[t]
                                    p_gamma_denom = self.Sex_tk[t][k] + self.gamma_sex_t[t] * self.S
                                    self.p_ytd_m[t][k] = self.N_tdk[t][d][k] * (p_gamma_numer / p_gamma_denom)
                                # end for k

                                # 減らした分を元に戻す
                                if topic >= 0:
                                    # カウントから topic の割当分を引く
                                    self.Sex_tks[t][topic][s] += 1
                                    self.Sex_tk[t][topic] += 1
                                    
                                    
                                # 正規化してサンプリング確率に変換
                                tmp_sum = sum(self.p_ytd_m[t])
                                for k in range(self.K):
                                    self.p_ytd_m[t][k] /= tmp_sum
                                
                                # トピックをサンプリング
                                self.y_sex_tdm[t][d][s] = np.random.multinomial(n=1, pvals=self.p_ytd_m[t], size=1).argmax()                         
                                # カウントに新たに割り当てたトピックの分を加える
                                new_topic = self.y_sex_tdm[t][d][s]
                                self.Sex_tdk[t][d][new_topic] += 1
                                self.Sex_tks[t][new_topic][s] += 1
                                self.Sex_tk[t][new_topic] += 1
                            # end for num
                        # end for s
                    # end for d

                    for d in range(self.D):   
                        # 単語数 0 のユーザーは除く
                        if self.N_td[t][d] == 0: continue
                        """ 補助情報: 子供数に応じたトピックのサンプリング """
                        num_chi = self.Con_ds[d][0]

                        # 時刻 t におけるユーザー d の子供数に割り当てられたトピックを取得
                        topic = self.y_chi_tdm[t][d][num_chi]

                        # 初回を飛ばす処理
                        if topic >= 0:
                            # カウントから topic の割当分を引く
                            self.Chi_tks[t][topic][num_chi] -= 1
                            self.Chi_tk[t][topic] -= 1
                        # end if 

                        # 時刻に依らず同じサンプリング式
                        for k in range(self.K):
                            # サンプリング確率を計算
                            p_gamma_numer = self.Chi_tks[t][k][num_chi] + self.gamma_chi_t[t]
                            p_gamma_denom = self.Chi_tk[t][k] + self.gamma_chi_t[t] * self.Chi
                            self.p_ytd_m[t][k] = self.N_tdk[t][d][k] * (p_gamma_numer / p_gamma_denom)
                        # end for k

                        # 減らした分を元に戻す
                        if topic >= 0:
                            # カウントから topic の割当分を引く
                            self.Chi_tks[t][topic][num_chi] += 1
                            self.Chi_tk[t][topic] += 1
                            
                            
                        # 正規化してサンプリング確率に変換
                        tmp_sum = sum(self.p_ytd_m[t])
                        for k in range(self.K):
                            self.p_ytd_m[t][k] /= tmp_sum
                        
                        # トピックをサンプリング
                        self.y_chi_tdm[t][d][num_chi] = np.random.multinomial(n=1, pvals=self.p_ytd_m[t], size=1).argmax()                         
                        # カウントに新たに割り当てたトピックの分を加える
                        new_topic = self.y_chi_tdm[t][d][num_chi]
                        self.Chi_tdk[t][d][new_topic] += 1
                        self.Chi_tks[t][new_topic][num_chi] += 1
                        self.Chi_tk[t][new_topic] += 1
                    # end for
                    

                    for d in range(self.D):   
                        # 単語数 0 のユーザーは除く
                        if self.N_td[t][d] == 0: continue                                              
                        """ 補助情報: 世代に応じたトピックのサンプリング """
                        gen_id = self.Con_ds[d][1]

                        # 時刻 t におけるユーザー d の世代idに割り当てられたトピックを取得
                        topic = self.y_gen_tdm[t][d][gen_id]

                        # 初回を飛ばす処理
                        if topic >= 0:
                            # カウントから topic の割当分を引く
                            self.Gen_tks[t][topic][gen_id] -= 1
                            self.Gen_tk[t][topic] -= 1
                        # end if 

                        # 時刻に依らず同じサンプリング式
                        for k in range(self.K):
                            # サンプリング確率を計算
                            p_gamma_numer = self.Gen_tks[t][k][gen_id] + self.gamma_gen_t[t]
                            p_gamma_denom = self.Gen_tk[t][k] + self.gamma_gen_t[t] * self.Gen
                            self.p_ytd_m[t][k] = self.N_tdk[t][d][k] * (p_gamma_numer / p_gamma_denom)
                        # end for k

                        # 減らした分を元に戻す
                        if topic >= 0:
                            # カウントから topic の割当分を引く
                            self.Gen_tks[t][topic][gen_id] += 1
                            self.Gen_tk[t][topic] += 1
                            
                            
                        # 正規化してサンプリング確率に変換
                        tmp_sum = sum(self.p_ytd_m[t])
                        for k in range(self.K):
                            self.p_ytd_m[t][k] /= tmp_sum
                        
                        # トピックをサンプリング
                        self.y_gen_tdm[t][d][gen_id] = np.random.multinomial(n=1, pvals=self.p_ytd_m[t], size=1).argmax()                         
                        # カウントに新たに割り当てたトピックの分を加える
                        new_topic = self.y_gen_tdm[t][d][gen_id]
                        self.Gen_tdk[t][d][new_topic] += 1
                        self.Gen_tks[t][new_topic][gen_id] += 1
                        self.Gen_tk[t][new_topic] += 1
                    # end for d

                    for d in range(self.D):   
                        # 単語数 0 のユーザーは除く
                        if self.N_td[t][d] == 0: continue
                        """ 補助情報: 都道府県に応じたトピックのサンプリング """
                        pre_id = self.Con_ds[d][2]

                        # 時刻 t におけるユーザー d の世代idに割り当てられたトピックを取得
                        topic = self.y_pre_tdm[t][d][pre_id]

                        # 初回を飛ばす処理
                        if topic >= 0:
                            # カウントから topic の割当分を引く
                            self.Pre_tks[t][topic][pre_id] -= 1
                            self.Pre_tk[t][topic] -= 1
                        # end if 

                        # 時刻に依らず同じサンプリング式
                        for k in range(self.K):
                            # サンプリング確率を計算
                            p_gamma_numer = self.Pre_tks[t][k][pre_id] + self.gamma_pre_t[t]
                            p_gamma_denom = self.Pre_tk[t][k] + self.gamma_pre_t[t] * self.Pre
                            self.p_ytd_m[t][k] = self.N_tdk[t][d][k] * (p_gamma_numer / p_gamma_denom)
                        # end for k

                        # 減らした分を元に戻す
                        if topic >= 0:
                            # カウントから topic の割当分を引く
                            self.Pre_tks[t][topic][pre_id] += 1
                            self.Pre_tk[t][topic] += 1
                            
                            
                        # 正規化してサンプリング確率に変換
                        tmp_sum = sum(self.p_ytd_m[t])
                        for k in range(self.K):
                            self.p_ytd_m[t][k] /= tmp_sum
                        
                        # トピックをサンプリング
                        self.y_pre_tdm[t][d][pre_id] = np.random.multinomial(n=1, pvals=self.p_ytd_m[t], size=1).argmax()                       
                        # カウントに新たに割り当てたトピックの分を加える
                        new_topic = self.y_pre_tdm[t][d][pre_id]
                        self.Pre_tdk[t][d][new_topic] += 1
                        self.Pre_tks[t][new_topic][pre_id] += 1
                        self.Pre_tk[t][new_topic] += 1
                    # end for d

                for d in range(self.D):   
                    # 単語数 0 のユーザーは除く
                    if self.N_td[t][d] == 0: continue
                    """ 補助情報: 質問カテゴリ id に応じたトピックのサンプリング """
                    q_id = self.Q_td[t][d]

                    # 時刻 t におけるユーザー d の質問カテゴリ id に割り当てられたトピックを取得
                    topic = self.y_q_tdm[t][d][q_id]

                    # 初回を飛ばす処理
                    if topic >= 0:
                        # カウントから topic の割当分を引く
                        self.Q_tks[t][topic][q_id] -= 1
                        self.Q_tk[t][topic] -= 1
                    # end if 

                    # 時刻に依らず同じサンプリング式
                    for k in range(self.K):
                        # サンプリング確率を計算
                        p_gamma_numer = self.Q_tks[t][k][q_id] + self.gamma_q_t[t]
                        p_gamma_denom = self.Q_tk[t][k] + self.gamma_q_t[t] * self.Q
                        self.p_ytd_m[t][k] = self.N_tdk[t][d][k] * (p_gamma_numer / p_gamma_denom)
                    # end for k

                    # 減らした分を元に戻す
                    if topic >= 0:
                        # カウントから topic の割当分を引く
                        self.Q_tks[t][topic][q_id] += 1
                        self.Q_tk[t][topic] += 1
                        
                        
                    # 正規化してサンプリング確率に変換
                    tmp_sum = sum(self.p_ytd_m[t])
                    for k in range(self.K):
                        self.p_ytd_m[t][k] /= tmp_sum
                    
                    # トピックをサンプリング
                    self.y_q_tdm[t][d][q_id] = np.random.multinomial(n=1, pvals=self.p_ytd_m[t], size=1).argmax()                       
                    # カウントに新たに割り当てたトピックの分を加える
                    new_topic = self.y_q_tdm[t][d][q_id]
                    self.Q_tdk[t][d][new_topic] += 1
                    self.Q_tks[t][new_topic][q_id] += 1
                    self.Q_tk[t][new_topic] += 1
                # end for d


                """ハイパーパラメータの更新"""
                # 最初の時刻の処理
                if t == 0:
                    
                    tmp_first_alpha = self.first_alpha.copy()
                    tmp_first_alpha_sum = sum(tmp_first_alpha)
                    tmp_first_beta = np.asarray([self.first_beta]).copy()[0]
                    beta_numer1 = 0.0
                    beta_denom1 = 0.0
                                            
                    # alpha の更新
                    for k in range(self.K):

                        # nan 回避
                        if math.isclose(tmp_first_alpha[k], 0.0):
                            self.first_alpha[k] = tmp_first_alpha[k]
                            continue
                        # end if

                        alpha_numer1 = 0.0
                        alpha_denom1 = 0.0
                        
                        for d in range(self.D):
                            alpha_numer1 += digamma(self.N_tdk[t][d][k] + tmp_first_alpha[k])
                            # 発散を防ぐため itr + 1 する
                            alpha_denom1 += digamma(self.N_td[t][d] * (itr + 1.0) + tmp_first_alpha_sum)
                        # end for d

                        alpha_numer = alpha_numer1 - self.D * digamma(tmp_first_alpha[k])
                        alpha_denom = alpha_denom1 - self.D * digamma(tmp_first_alpha_sum)
                        # nan の回避
                        try:
                            self.first_alpha[k] = tmp_first_alpha[k] * alpha_numer / alpha_denom
                        except ZeroDivisionError as err:
                            print("t=0の時のalphaの更新でゼロ除算発生. alphaが発散している可能性.", err)
                            return
                    # end for k

                    # beta の更新
                    for k in range(self.K):
                        for v in range(self.V):
                            beta_numer1 += digamma(self.N_tkv[t][k][v] + tmp_first_beta)
                        # end for v
                        beta_denom1 += digamma(self.N_tk[t][k] + tmp_first_beta * self.V)
                    # end for k
                    beta_numer = beta_numer1 - self.K * self.V * digamma(tmp_first_beta)
                    beta_denom = self.V * beta_denom1 - self.K * self.V * digamma(tmp_first_beta * self.V)
                    try:
                        self.first_beta = tmp_first_beta * beta_numer / beta_denom
                    except ZeroDivisionError as err:
                        print("t=0の時のbetaの更新でゼロ除算発生. betaが発散している可能性.", err)
                        return
                    
                    # トピック分布の点推定
                    if itr >= iteration - sample_size:
                        for d in range(self.D):
                            for k in range(self.K):
                                self.theta_tdk[t][d][k] = (self.N_tdk[t][d][k] + self.first_alpha[k]) / (self.N_td[t][d] * (itr + 1.0) + sum(self.first_alpha))
                                # サンプルサイズの平均を求める
                                if itr == iteration - 1:
                                    self.theta_tdk[t][d][k] /= sample_size
                            # end for k
                        # end for d
                        
                        # 単語分布の点推定
                        for k in range(self.K):
                            for v in range(self.V):
                                self.phi_tkv[t][k][v] = (self.N_tkv[t][k][v] + self.first_beta) / (self.N_tk[t][k] + self.first_beta * self.V)
                                # サンプリングサイズの平均を求める
                                if itr == iteration - 1:
                                    self.phi_tkv[t][k][v] /= sample_size
                            # end for v
                        # end for k
                    # end if
                    
                    # gamma_sex_t の更新
                    tmp_gamma = np.asarray([self.gamma_sex_t[t]]).copy()[0]
                    gamma_numer1 = 0.0
                    gamma_denom1 = 0.0

                    for k in range(self.K):
                        for s in range(self.S):
                            gamma_numer1 += digamma(self.Sex_tks[t][k][s] + tmp_gamma)
                        # end for s
                        gamma_denom1 += digamma(self.Sex_tk[t][k] + tmp_gamma * self.S)
                    # end for k
                    gamma_numer = gamma_numer1 - self.K * self.S * digamma(tmp_gamma)
                    gamma_denom = self.S * gamma_denom1 - self.K * self.S * digamma(tmp_gamma * self.S)
                    try:
                        self.gamma_sex_t[t] = tmp_gamma * gamma_numer / gamma_denom
                    except ZeroDivisionError as err:
                        print("t=0 の時の gamma_sex_t の更新でゼロ除算発生. gamma_sex_t が発散している可能性.", err)
                        return

                    # gamma_chi_t の更新
                    tmp_gamma = np.asarray([self.gamma_chi_t[t]]).copy()[0]
                    gamma_numer1 = 0.0
                    gamma_denom1 = 0.0
                    
                    for k in range(self.K):
                        for s in range(self.Chi):
                            gamma_numer1 += digamma(self.Chi_tks[t][k][s] + tmp_gamma)
                        # end for s
                        gamma_denom1 += digamma(self.Chi_tk[t][k] + tmp_gamma * self.Chi)
                    # end for k
                    gamma_numer = gamma_numer1 - self.K * self.Chi * digamma(tmp_gamma)
                    gamma_denom = self.Chi * gamma_denom1 - self.K * self.Chi * digamma(tmp_gamma * self.Chi)
                    try:
                        self.gamma_chi_t[t] = tmp_gamma * gamma_numer / gamma_denom
                    except ZeroDivisionError as err:
                        print("t=0 の時の gamma_chi_t の更新でゼロ除算発生. gamma_chi_t が発散している可能性.", err)
                        return

                    # gamma_gen_t の更新
                    tmp_gamma = np.asarray([self.gamma_gen_t[t]]).copy()[0]
                    gamma_numer1 = 0.0
                    gamma_denom1 = 0.0
                    
                    for k in range(self.K):
                        for s in range(self.Gen):
                            gamma_numer1 += digamma(self.Gen_tks[t][k][s] + tmp_gamma)
                        # end for s
                        gamma_denom1 += digamma(self.Gen_tk[t][k] + tmp_gamma * self.Gen)
                    # end for k
                    gamma_numer = gamma_numer1 - self.K * self.Gen * digamma(tmp_gamma)
                    gamma_denom = self.Gen * gamma_denom1 - self.K * self.Gen * digamma(tmp_gamma * self.Gen)
                    try:
                        self.gamma_gen_t[t] = tmp_gamma * gamma_numer / gamma_denom
                    except ZeroDivisionError as err:
                        print("t=0の時のgamma_gen_t の更新でゼロ除算発生. gamma_gen_t が発散している可能性.", err)
                        return

                    # gamma_pre_t の更新
                    tmp_gamma = np.asarray([self.gamma_pre_t[t]]).copy()[0]
                    gamma_numer1 = 0.0
                    gamma_denom1 = 0.0
                    
                    for k in range(self.K):
                        for s in range(self.Pre):
                            gamma_numer1 += digamma(self.Pre_tks[t][k][s] + tmp_gamma)
                        # end for s
                        gamma_denom1 += digamma(self.Pre_tk[t][k] + tmp_gamma * self.Pre)
                    # end for k
                    gamma_numer = gamma_numer1 - self.K * self.Pre * digamma(tmp_gamma)
                    gamma_denom = self.Pre * gamma_denom1 - self.K * self.Pre * digamma(tmp_gamma * self.Pre)
                    try:
                        self.gamma_pre_t[t] = tmp_gamma * gamma_numer / gamma_denom
                    except ZeroDivisionError as err:
                        print("t=0 の時の gamma_pre_t の更新でゼロ除算発生. gamma_pre_t が発散している可能性.", err)
                        return                    
                    
                    # 補助情報分布の点推定: 子供の性別
                    if itr >= iteration - sample_size:
                        for k in range(self.K):
                            for s in range(self.S):
                                self.psi_sex_tks[t][k][s] = (self.Sex_tks[t][k][s] + self.gamma_sex_t[t]) / (self.Sex_tk[t][k] + self.gamma_sex_t[t] * self.S)
                                if itr == iteration - 1:
                                    self.psi_sex_tks[t][k][s] /= sample_size

                        # 補助情報分布の点推定: 子供数
                        for k in range(self.K):
                            for s in range(self.Chi):
                                self.psi_chi_tks[t][k][s] = (self.Chi_tks[t][k][s] + self.gamma_chi_t[t]) / (self.Chi_tk[t][k] + self.gamma_chi_t[t] * self.Chi)
                                if itr == iteration - 1:
                                    self.psi_chi_tks[t][k][s] /= sample_size

                        # 補助情報分布の点推定: 世代
                        for k in range(self.K):
                            for s in range(self.Gen):
                                self.psi_gen_tks[t][k][s] = (self.Gen_tks[t][k][s] + self.gamma_gen_t[t]) / (self.Gen_tk[t][k] + self.gamma_gen_t[t] * self.Gen)
                                if itr == iteration - 1:
                                    self.psi_gen_tks[t][k][s] /= sample_size
                        
                        # 補助情報分布の点推定: 都道府県
                        for k in range(self.K):
                            for s in range(self.Pre):
                                self.psi_pre_tks[t][k][s] = (self.Pre_tks[t][k][s] + self.gamma_pre_t[t]) / (self.Pre_tk[t][k] + self.gamma_pre_t[t] * self.Pre)
                                if itr == iteration - 1:
                                    self.psi_pre_tks[t][k][s] /= sample_size
                    # end if
                    
                    
                # 最初から数えて L 時刻目までの処理
                elif self.L > t > 0:

                    tmp_alpha = self.alpha_tdl.copy()
                    tmp_beta = self.beta_tkl.copy()
                    
                    for d in range(self.D):

                        # 時刻 t で投稿数 0 のユーザーの処理
                        if self.N_td[t][d] == 0:
                            continue

                        for l in range(t):

                            # nan 回避
                            if math.isclose(tmp_alpha[t][d][l], 0.0):
                                self.alpha_tdl[t][d][l] = tmp_alpha[t][d][l]
                                continue
                            # end if

                            # alpha の更新
                            alpha_numer = 0.0
                            for k in range(self.K):
                                tmp_alpha_numer = sum([tmp_alpha[t][d][_l] * self.theta_tdk[t-_l-1][d][k] for _l in range(t)])
                                if self.N_tdk[t][d][k] == 0: alpha_numer_digamma = 0.0
                                else: alpha_numer_digamma = digamma(self.N_tdk[t][d][k] + tmp_alpha_numer) - digamma(tmp_alpha_numer)
                                alpha_numer += self.theta_tdk[t-l-1][d][k] * alpha_numer_digamma
                            # end for k
                            tmp_alpha_denom = sum(np.asarray(tmp_alpha[t][d])[:t])
                            alpha_denom = digamma(self.N_td[t][d] * (itr + 1.0) + tmp_alpha_denom) - digamma(tmp_alpha_denom)
                            try:
                                self.alpha_tdl[t][d][l] = tmp_alpha[t][d][l] * alpha_numer / alpha_denom
                            except ZeroDivisionError as err:
                                print("L>t>0の時のalphaの更新でゼロ除算発生. alphaが発散している可能性.", err)
                                return
                        # end for l
                    # end for d

                    # トピック分布の点推定
                    if itr >= iteration - sample_size:
                        for d in range(self.D):
                            for k in range(self.K):
                                tmp_alpha_numer = sum([self.alpha_tdl[t][d][_l] * self.theta_tdk[t-_l-1][d][k] for _l in range(t)])
                                tmp_alpha_denom = sum(np.asarray(self.alpha_tdl[t][d])[:t])
                                self.theta_tdk[t][d][k] = (self.N_tdk[t][d][k] + tmp_alpha_numer) / (self.N_td[t][d] * (itr + 1.0) + tmp_alpha_denom)
                                # サンプルサイズの平均を求める
                                if itr == iteration - 1:
                                    self.theta_tdk[t][d][k] = sample_size
                            # end for k
                        # end for d
                    # end if
                    
                    for k in range(self.K):

                        # 時刻 t でトピック数 0 だと beta と phi は更新しない
                        if self.N_tk[t][k] == 0:
                            continue

                        for l in range(t):  
                            # beta の更新

                            beta_numer = 0.0
                            for v in range(self.V):
                                tmp_beta_numer = sum([tmp_beta[t][k][_l] * self.phi_tkv[t-_l-1][k][v] for _l in range(t)])
                                beta_numer_digamma = digamma(self.N_tkv[t][k][v] + tmp_beta_numer) - digamma(tmp_beta_numer)
                                beta_numer += self.phi_tkv[t-l-1][k][v] * beta_numer_digamma
                            # end for v
                            tmp_beta_denom = sum(np.asarray(tmp_beta[t][k])[:t])
                            beta_denom = digamma(self.N_tk[t][k] + tmp_beta_denom) - digamma(tmp_beta_denom)
                            try:
                                self.beta_tkl[t][k][l] = tmp_beta[t][k][l] * beta_numer / beta_denom
                            except ZeroDivisionError as err:
                                    print("L>t>0の時のbetaの更新でゼロ除算発生. betaが発散している可能性.", err)
                                    return            
                        # end for l
                    # end for k
                        
                    # 単語分布の点推定
                    if itr >= iteration - sample_size:
                        for k in range(self.K):
                            for v in range(self.V):
                                tmp_beta_numer = sum([self.beta_tkl[t][k][_l] * self.phi_tkv[t-_l-1][k][v] for _l in range(t)])
                                tmp_beta_denom = sum(np.asarray(self.beta_tkl[t][k])[:t])
                                self.phi_tkv[t][k][v] = (self.N_tkv[t][k][v] + tmp_beta_numer) / (self.N_tk[t][k] + tmp_beta_denom)
                                # サンプリングサイズの平均を求める
                                if itr == iteration - 1:
                                    self.phi_tkv[t][k][v] /= sample_size
                            # end for v
                        # end for k
                    # end if
                    

                # L 時刻目以降の処理
                else:

                    tmp_alpha = self.alpha_tdl.copy()
                    tmp_beta = self.beta_tkl.copy()

                    for d in range(self.D):

                        # 時刻 t で投稿数 0 のユーザーの処理
                        if self.N_td[t][d] == 0:
                            continue

                        for l in range(self.L):

                            # nan 回避
                            if math.isclose(tmp_alpha[t][d][l], 0.0):
                                self.alpha_tdl[t][d][l] = tmp_alpha[t][d][l]
                                continue
                            # end if

                            # alpha の更新
                            alpha_numer = 0.0
                            for k in range(self.K):
                                tmp_alpha_numer = sum([tmp_alpha[t][d][_l] * self.theta_tdk[t-_l-1][d][k] for _l in range(self.L)])
                                if math.isclose(tmp_alpha_numer, 0.0): alpha_numer += 0.0
                                else:
                                    alpha_numer_digamma = digamma(self.N_tdk[t][d][k] + tmp_alpha_numer) - digamma(tmp_alpha_numer)
                                    alpha_numer += self.theta_tdk[t-l-1][d][k] * alpha_numer_digamma
                            # end for k
                            tmp_alpha_denom = sum(tmp_alpha[t][d])
                            alpha_denom = digamma(self.N_td[t][d] * (itr + 1.0) + tmp_alpha_denom) - digamma(tmp_alpha_denom)
                            try:
                                self.alpha_tdl[t][d][l] = tmp_alpha[t][d][l] * alpha_numer / alpha_denom
                            except ZeroDivisionError as err:
                                print("t>Lの時のalphaの更新でゼロ除算発生. alphaが発散している可能性.", err)
                                return
                        # end for l
                    # end for d

                    # トピック分布の点推定
                    if itr >= iteration - sample_size:
                        for d in range(self.D):
                            for k in range(self.K):
                                tmp_alpha_numer = sum([self.alpha_tdl[t][d][_l] * self.theta_tdk[t-_l-1][d][k] for _l in range(self.L)])
                                tmp_alpha_denom = sum(self.alpha_tdl[t][d])
                                self.theta_tdk[t][d][k] = (self.N_tdk[t][d][k] + tmp_alpha_numer) / (self.N_td[t][d] * (itr + 1.0) + tmp_alpha_denom)
                                # サンプリングサイズの平均を求める
                                if itr == iteration - 1:
                                    self.theta_tdk[t][d][k] /= sample_size
                            # end for k
                        # end for d
                    # end if


                    for k in range(self.K):

                        # 時刻 t でトピック数 0 のユーザーの処理
                        if self.N_tk[t][k] == 0:
                            continue

                        for l in range(self.L):
                            # beta の更新

                            beta_numer = 0.0
                            for v in range(self.V):
                                if math.isclose(self.phi_tkv[t-l-1][k][v], 0.0): beta_numer += 0.0
                                else:
                                    tmp_beta_numer = sum([tmp_beta[t][k][_l] * self.phi_tkv[t-_l-1][k][v] for _l in range(self.L)])
                                    beta_numer_digamma = digamma(self.N_tkv[t][k][v] + tmp_beta_numer) - digamma(tmp_beta_numer)
                                    beta_numer += self.phi_tkv[t-l-1][k][v] * beta_numer_digamma                                
                            # end for v
                            tmp_beta_denom = sum(tmp_beta[t][k])
                            try:
                                beta_denom = digamma(self.N_tk[t][k] + tmp_beta_denom) - digamma(tmp_beta_denom)
                                self.beta_tkl[t][k][l] = tmp_beta[t][k][l] * beta_numer / beta_denom
                            except ZeroDivisionError as err:
                                print("t>Lの時のbetaの更新でゼロ除算発生. betaが発散している可能性.", err)
                                return
                        # end for l
                    # end for k

                    # 単語分布の点推定
                    if itr >= iteration - sample_size:
                        for k in range(self.K):
                            for v in range(self.V):
                                tmp_beta_numer = sum([self.beta_tkl[t][k][_l] * self.phi_tkv[t-_l-1][k][v] for _l in range(self.L)])
                                tmp_beta_denom = sum(self.beta_tkl[t][k])
                                self.phi_tkv[t][k][v] = (self.N_tkv[t][k][v] + tmp_beta_numer) / (self.N_tk[t][k] + tmp_beta_denom)
                                # サンプリングサイズの平均を求める
                                if itr == iteration - 1:
                                    self.phi_tkv[t][k][v] /= sample_size
                            # end for v
                        # end for k
                    #end if
                
                # end if

                # gamma_q_t の更新
                tmp_gamma = np.asarray([self.gamma_q_t[t]]).copy()[0]
                gamma_numer1 = 0.0
                gamma_denom1 = 0.0
                
                for k in range(self.K):
                    for s in range(self.Q):
                        gamma_numer1 += digamma(self.Q_tks[t][k][s] + tmp_gamma)
                    # end for s
                    gamma_denom1 += digamma(self.Q_tk[t][k] + tmp_gamma * self.Q)
                # end for k
                gamma_numer = gamma_numer1 - self.K * self.Q * digamma(tmp_gamma)
                gamma_denom = self.Q * gamma_denom1 - self.K * self.Q * digamma(tmp_gamma * self.Q)
                try:
                    self.gamma_q_t[t] = tmp_gamma * gamma_numer / gamma_denom
                except ZeroDivisionError as err:
                    print("t=0 の時の gamma_q_t の更新でゼロ除算発生. gamma_q_t が発散している可能性.", err)
                    return
            
                # 補助情報分布の点推定: 質問カテゴリ
                if itr >= iteration - sample_size:
                    for k in range(self.K):
                        for s in range(self.Q):
                            self.psi_q_tks[t][k][s] = (self.Q_tks[t][k][s] + self.gamma_q_t[t]) / (self.Q_tk[t][k] + self.gamma_q_t[t] * self.Q)
                            # サンプリングサイズの平均を求める
                            if itr == iteration - 1:
                                self.psi_q_tks[t][k][s] /= sample_size
                # end if

            # end iter
        # end for t
    # end def