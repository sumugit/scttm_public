"""Supervised Correspondence TTM with Cython"""

from libcpp.vector cimport vector
import numpy as np
import sys
import math
import cython
import random
cimport numpy as np
from scipy.special import digamma
from libc.math cimport pow as c_pow
from libc.math cimport exp as c_exp
from threading import Thread
from cython.parallel import prange
from cython.parallel import threadid
from tqdm import tqdm
import warnings

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval)

srand48(1234)
warnings.filterwarnings('error')

# 境界チェックを無視
@cython.boundscheck(False)
@cython.wraparound(False)

cdef class LDA():

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

    # 補助情報: ユーザーの子供の月齢
    cdef public vector[vector[double]] Age_td
    cdef public double[:,:] eta_tk
    cdef public double[:] sigma_t
    cdef public double[:,:,:] z_tdk_var

    # トピック
    cdef public int[:,:,:] z_tdn
    cdef public int[:,:,:] y_sex_tdm
    cdef public int[:,:,:] y_chi_tdm
    cdef public int[:,:,:] y_gen_tdm
    cdef public int[:,:,:] y_pre_tdm
    cdef public int[:,:,:] y_q_tdm

    # 推論するパラメータ
    cdef public double[:,:] first_alpha
    cdef public double[:] first_beta
    cdef public double[:] gamma_sex_t
    cdef public double[:] gamma_chi_t
    cdef public double[:] gamma_gen_t
    cdef public double[:] gamma_pre_t
    cdef public double[:] gamma_q_t
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
                vector[vector[int]] Q_td, vector[vector[double]] Age_td):
        """ 初期化 """
        # N : T×D×V

        cdef unsigned int t
        cdef unsigned int d

        self.K = K # トピック数
        self.L = L # 依存する過去の期間
        self.N_tdv = N_tdv # 単語のBOW行列
        self.V = N_tdv.at(0).at(0).size() # 単語数 + 上位n件に含まれなかった残りの語
        self.S = 3 # 子供の性別の要素数
        self.Chi = 7 # 子供数
        self.Gen = 8 # 世代数
        self.Pre = 48 # 都道府県数
        self.Q = 16 # 質問カテゴリ数
        self.Sex_ds = Sex_ds # ユーザー属性: Sex の BOW 行列  
        self.Con_ds = Con_ds # 他のユーザー属性
        self.Q_td = Q_td # ユーザーの質問カテゴリ
        self.Age_td = Age_td # ユーザーの子供の月齢
        self.T = self.N_tdv.size() # 期間
        self.D = self.N_tdv.at(0).size()  # 文書数
        self.N_td = np.zeros(shape=(self.T, self.D)).astype('uint32') # 時刻 t , ユーザー d の語彙数
        
        for t in range(self.T):
            for d in range(self.D):
                self.N_td[t][d] = sum(self.N_tdv[t][d])
        

        # ハイパーパラメータ alpha, betaの初期化
        # t = 0 の時のハイパラのみ別処理
        self.first_alpha = np.full((self.T, self.K), 2.0) # 非一様
        self.first_beta = np.full((self.T), 2.0) # 一様

        # トピック分布, 単語分布, 補助情報分布の初期化
        self.theta_tdk = np.full((self.T, self.D, self.K), 0.0)
        self.phi_tkv = np.full((self.T, self.K, self.V), 0.0)

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



        # 回帰の説明変数の初期化
        self.z_tdk_var = np.zeros(shape=(self.T, self.D, self.K))
    
    
    def lda_estimate(self, unsigned int iteration, unsigned int sample_size):
        """崩壊型ギブスサンプリングで推定"""

        cdef unsigned int t, d, k, l, _l, v, s, num_chi, gen_id, pre_id, q_id, index
        cdef unsigned int itr
        cdef int topic, new_topic
        cdef double tmp_sum, sigma_left
        cdef double[:] tmp_first_beta, pred_y, eta_right
        cdef double[:,:] tmp_first_alpha, eta_left
        cdef double tmp_alpha_numer, tmp_alpha_denom, tmp_alpha_sum, tmp_first_alpha_sum
        cdef double tmp_beta_numer, tmp_beta_denom, tmp_beta_sum
        cdef double p_alpha, p_beta_numer, p_beta_denom, p_gamma_numer, p_gamma_denom, p_aid
        cdef double alpha_theta, beta_phi
        cdef double alpha_numer1, alpha_numer, alpha_denom1, alpha_denom, alpha_numer_digamma
        cdef double beta_numer1, beta_numer, beta_denom1, beta_denom, beta_numer_digamma
        cdef double gamma_numer1, gamma_numer, gamma_denom1, gamma_denom, tmp_gamma, intercept
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

                            # 説明変数を構築
                            for k in range(self.K):
                                self.z_tdk_var[t][d][k] = self.N_tdk[t][d][k]/(self.N_td[t][d] * (itr+1.0))
                            # end for k

                            # 最初の時刻のサンプリング
                            if t == 0:
                                for k in range(self.K):
                                    # サンプリング確率を計算
                                    p_alpha = self.N_tdk[t][d][k] + self.first_alpha[t][k]
                                    p_beta_numer = self.N_tkv[t][k][v] + self.first_beta[t]
                                    p_beta_denom = self.N_tk[t][k] + self.first_beta[t] * self.V
                                    self.p_ztd_n[t][k] = p_alpha * (p_beta_numer / p_beta_denom)
                                # end for k
                                

                            # それ以外
                            else:
                                for k in range(self.K):
                                    # サンプリング確率を計算
                                    p_alpha = self.N_tdk[t][d][k] + self.first_alpha[t][k]
                                    p_beta_numer = self.N_tkv[t][k][v] + self.first_beta[t]
                                    p_beta_denom = self.N_tk[t][k] + self.first_beta[t] * self.V
                                    self.p_ztd_n[t][k] = p_alpha * (p_beta_numer / p_beta_denom)
                                # end for k
                            # end if

                            # 減らした分を元に戻す
                            if topic >= 0:
                                # カウントから topic の割当分を引く
                                self.N_tdk[t][d][topic] += 1
                                self.N_tkv[t][topic][v] += 1
                                self.N_tk[t][topic] += 1
                            # end if
                            
                            # 正規化してサンプリング確率に変換
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
                                return
                            # カウントに新たに割り当てたトピックの分を加える
                            new_topic = self.z_tdn[t][d][v]
                            self.N_tdk[t][d][new_topic] += 1
                            self.N_tkv[t][new_topic][v] += 1
                            self.N_tk[t][new_topic] += 1
                        # end for num
                    # end for v
                # end for d


                """ハイパーパラメータの更新"""
                # 最初の時刻の処理
                if t == 0:
                    
                    tmp_first_alpha = self.first_alpha.copy()
                    tmp_first_alpha_sum = sum(tmp_first_alpha[t])
                    tmp_first_beta = self.first_beta.copy()
                    beta_numer1 = 0.0
                    beta_denom1 = 0.0
                                            
                    # alpha の更新
                    for k in range(self.K):

                        # nan 回避
                        if math.isclose(tmp_first_alpha[t][k], 0.0):
                            self.first_alpha[t][k] = tmp_first_alpha[t][k]
                            continue
                        # end if

                        alpha_numer1 = 0.0
                        alpha_denom1 = 0.0
                        
                        for d in range(self.D):
                            alpha_numer1 += digamma(self.N_tdk[t][d][k] + tmp_first_alpha[t][k])
                            # 発散を防ぐため itr + 1 する
                            alpha_denom1 += digamma(self.N_td[t][d] * (itr + 1.0) + tmp_first_alpha_sum)
                        # end for d

                        alpha_numer = alpha_numer1 - self.D * digamma(tmp_first_alpha[t][k])
                        alpha_denom = alpha_denom1 - self.D * digamma(tmp_first_alpha_sum)
                        # nan の回避
                        try:
                            self.first_alpha[t][k] = tmp_first_alpha[t][k] * alpha_numer / alpha_denom
                        except ZeroDivisionError as err:
                            print("t=0の時のalphaの更新でゼロ除算発生. alphaが発散している可能性.", err)
                            return
                    # end for k

                    # beta の更新
                    for k in range(self.K):
                        for v in range(self.V):
                            beta_numer1 += digamma(self.N_tkv[t][k][v] + tmp_first_beta[t])
                        # end for v
                        beta_denom1 += digamma(self.N_tk[t][k] + tmp_first_beta[t] * self.V)
                    # end for k
                    beta_numer = beta_numer1 - self.K * self.V * digamma(tmp_first_beta[t])
                    beta_denom = self.V * beta_denom1 - self.K * self.V * digamma(tmp_first_beta[t] * self.V)
                    try:
                        self.first_beta[t] = tmp_first_beta[t] * beta_numer / beta_denom
                    except ZeroDivisionError as err:
                        print("t=0の時のbetaの更新でゼロ除算発生. betaが発散している可能性.", err)
                        return
                    
                    # トピック分布の点推定
                    if itr >= iteration - sample_size:
                        for d in range(self.D):
                            for k in range(self.K):
                                self.theta_tdk[t][d][k] += (self.N_tdk[t][d][k] + self.first_alpha[t][k]) / (self.N_td[t][d] * (itr + 1.0) + sum(self.first_alpha[t]))
                                # サンプリングサイズの平均を求める
                                if itr == iteration - 1:
                                    self.theta_tdk[t][d][k] /= sample_size                                
                            # end for k
                        # end for d
                        
                        # 単語分布の点推定
                        for k in range(self.K):
                            for v in range(self.V):
                                self.phi_tkv[t][k][v] += (self.N_tkv[t][k][v] + self.first_beta[t]) / (self.N_tk[t][k] + self.first_beta[t] * self.V)
                                # サンプリングサイズの平均を求める
                                if itr == iteration - 1:
                                    self.phi_tkv[t][k][v] /= sample_size                                
                            # end for v
                        # end for k
                    # end if

                    
                # L 時刻目以降の処理
                else:

                    tmp_first_alpha = self.first_alpha.copy()
                    tmp_first_alpha_sum = sum(tmp_first_alpha[t])
                    tmp_first_beta = self.first_beta.copy()
                    beta_numer1 = 0.0
                    beta_denom1 = 0.0
                    
                    # alpha の更新
                    for k in range(self.K):

                        # nan 回避
                        if math.isclose(tmp_first_alpha[t][k], 0.0):
                            self.first_alpha[t][k] = tmp_first_alpha[t][k]
                            continue
                        # end if

                        alpha_numer1 = 0.0
                        alpha_denom1 = 0.0
                        
                        for d in range(self.D):
                            alpha_numer1 += digamma(self.N_tdk[t][d][k] + tmp_first_alpha[t][k])
                            # 発散を防ぐため itr + 1 する
                            alpha_denom1 += digamma(self.N_td[t][d] * (itr + 1.0) + tmp_first_alpha_sum)
                        # end for d

                        alpha_numer = alpha_numer1 - self.D * digamma(tmp_first_alpha[t][k])
                        alpha_denom = alpha_denom1 - self.D * digamma(tmp_first_alpha_sum)
                        # nan の回避
                        try:
                            self.first_alpha[t][k] = tmp_first_alpha[t][k] * alpha_numer / alpha_denom
                        except ZeroDivisionError as err:
                            print("t=0の時のalphaの更新でゼロ除算発生. alphaが発散している可能性.", err)
                            return
                    # end for k

                    # beta の更新
                    for k in range(self.K):
                        for v in range(self.V):
                            beta_numer1 += digamma(self.N_tkv[t][k][v] + tmp_first_beta[t])
                        # end for v
                        beta_denom1 += digamma(self.N_tk[t][k] + tmp_first_beta[t] * self.V)
                    # end for k
                    beta_numer = beta_numer1 - self.K * self.V * digamma(tmp_first_beta[t])
                    beta_denom = self.V * beta_denom1 - self.K * self.V * digamma(tmp_first_beta[t] * self.V)
                    try:
                        self.first_beta[t] = tmp_first_beta[t] * beta_numer / beta_denom
                    except ZeroDivisionError as err:
                        print("t=0の時のbetaの更新でゼロ除算発生. betaが発散している可能性.", err)
                        return
                    
                    # トピック分布の点推定
                    if itr >= iteration - sample_size:
                        for d in range(self.D):
                            for k in range(self.K):
                                self.theta_tdk[t][d][k] += (self.N_tdk[t][d][k] + self.first_alpha[t][k]) / (self.N_td[t][d] * (itr + 1.0) + sum(self.first_alpha[t]))
                                # サンプリングサイズの平均を求める
                                if itr == iteration - 1:
                                    self.theta_tdk[t][d][k] /= sample_size
                            # end for k
                        # end for d
                        
                        # 単語分布の点推定
                        for k in range(self.K):
                            for v in range(self.V):
                                self.phi_tkv[t][k][v] += (self.N_tkv[t][k][v] + self.first_beta[t]) / (self.N_tk[t][k] + self.first_beta[t] * self.V)
                                # サンプリングサイズの平均を求める
                                if itr == iteration - 1:
                                    self.phi_tkv[t][k][v] /= sample_size                                
                            # end for v
                        # end for k
                    # end if
                # end if

            # end iter
        # end for t
    # end def