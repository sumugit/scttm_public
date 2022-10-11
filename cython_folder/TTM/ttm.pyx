"""TTM with Cython"""

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
from sklearn.metrics import r2_score, mean_squared_error
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

cdef class TTM():

    cdef public unsigned int K, D, V, T, L
    cdef public vector[vector[vector[int]]] N_tdv
    cdef public vector[vector[vector[int]]] N_tdk
    cdef public vector[vector[vector[int]]] N_tkv
    cdef public vector[vector[int]] N_tk
    cdef public vector[vector[int]] N_td
    cdef public int[:,:,:] z_tdn
    cdef public double[:] first_alpha
    cdef public double first_beta
    cdef public double[:,:,:] alpha_tdl
    cdef public double[:,:,:] beta_tkl
    cdef public double[:,:,:] theta_tdk
    cdef public double[:,:,:] phi_tkv
    cdef public double[:,:] p_ztd_n


    def __init__(self, unsigned int K, unsigned int L,
                vector[vector[vector[int]]] N_tdv):
        """ 初期化 """
        # N : T×D×V

        cdef unsigned int t
        cdef unsigned int d

        self.K = K # トピック数
        self.L = L # 依存する過去の期間
        self.N_tdv = N_tdv # BOW行列
        self.V = self.N_tdv.at(0).at(0).size()  # 単語数
        self.T = self.N_tdv.size() # 期間
        self.D = self.N_tdv.at(0).size()  # 文書数
        self.N_td = np.zeros(shape=(self.T, self.D)).astype('uint32') # 時刻 t , ユーザー d の語彙数
        
        for t in range(self.T):
            for d in range(self.D):
                self.N_td[t][d] = sum(self.N_tdv[t][d])
        

        # ハイパーパラメータ alpha, betaの初期化
        # t = 0 の時のハイパラのみ別処理
        self.first_alpha = np.full((self.K), 2.0) # 非一様
        self.first_beta = 2.0 # 一様
        # TODO: alphaをさらに k 次元に拡張して非一様なハイパラとする.
        self.alpha_tdl = np.full((self.T, self.D, self.L), 2.0) # 一様
        self.beta_tkl = np.full((self.T, self.K, self.L), 2.0) # 一様

        # トピック分布, 単語分布の初期化
        self.theta_tdk = np.full((self.T, self.D, self.K), 1.0/self.K)
        self.phi_tkv = np.full((self.T, self.K, self.V), 1.0/self.V)

        # 時刻 t, ユーザー d, 語彙 vに割り当てられたトピック (z_tdn) の集合
        self.z_tdn = np.full((self.T, self.D, self.V), -1).astype('int32')
        # 時刻 t の文書 d においてトピック k が割り当てられた単語数 N_dk の集合
        self.N_tdk = np.zeros(shape=(self.T, self.D, self.K)).astype('uint32')
        # 時刻 t の文書全体でトピック k が割り当てられた語彙 v の出現回数 N_kv の集合
        self.N_tkv = np.zeros(shape=(self.T, self.K, self.V)).astype('uint32')
        # 時刻 t の全文書でトピック k が割り当てられた単語数 N_k のベクトル
        self.N_tk = np.zeros(shape=(self.T, self.K)).astype('uint32')
        # 時刻 t のサンプリング確率
        self.p_ztd_n = np.zeros(shape=(self.T, self.K))

    
    def ttm_estimate(self, unsigned int iteration, unsigned int sample_size):
        """崩壊型ギブスサンプリングで推定"""

        cdef unsigned int t, d, n, k, l, _l, v, num
        cdef unsigned int itr
        cdef int topic, new_topic
        cdef double tmp_alpha_td, tmp_beta_tk, tmp_sum, mse
        cdef double[:] tmp_first_alpha, eta_right, eta_temp
        cdef double[:,:] eta_left
        cdef double[:,:,:] tmp_alpha, tmp_beta
        cdef double tmp_alpha_numer, tmp_alpha_denom, tmp_alpha_sum, tmp_first_alpha_sum
        cdef double tmp_beta_numer, tmp_beta_denom, tmp_beta_sum, tmp_first_beta
        cdef double p_alpha, p_beta_numer, p_beta_denom
        cdef double alpha_theta, beta_phi
        cdef double alpha_numer1, alpha_numer, alpha_denom1, alpha_denom, alpha_numer_digamma
        cdef double beta_numer1, beta_numer, beta_denom1, beta_denom, beta_numer_digamma
        cdef list del_list

        # 期間でループ
        for t in range(self.T):
            # repeat
            for itr in tqdm(range(iteration)):

                # ユーザー数でループ
                for d in range(self.D):
                    # ユーザー d の語彙数でループ
                    for v in range(self.V):
                        for num in range(self.N_tdv[t][d][v]):
                    
                            # 時刻 t におけるユーザー d の語彙 v のインデックスを取得
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
                                    self.p_ztd_n[t][k] = p_alpha * p_beta_numer / p_beta_denom
                                # end for k
                        
                            
                            # 最初から数えて L 時刻目までのサンプリング式
                            elif self.L > t > 0:
                                for k in range(self.K):
                                    alpha_theta = sum([self.alpha_tdl[t][d][l] * self.theta_tdk[t-l-1][d][k] for l in range(t)])
                                    beta_phi = sum([self.beta_tkl[t][k][l] * self.phi_tkv[t-l-1][k][v] for l in range(t)])

                                    p_alpha = self.N_tdk[t][d][k] + alpha_theta
                                    p_beta_numer = self.N_tkv[t][k][v] + beta_phi
                                    p_beta_denom = self.N_tk[t][k] + sum(self.beta_tkl[t][k][:t])
                                    self.p_ztd_n[t][k] = p_alpha * p_beta_numer / p_beta_denom
                                # end for k


                            # L 時刻目以降のサンプリング式
                            else:
                                for k in range(self.K):
                                    alpha_theta = sum([self.alpha_tdl[t][d][l] * self.theta_tdk[t-l-1][d][k] for l in range(self.L)])
                                    beta_phi = sum([self.beta_tkl[t][k][l] * self.phi_tkv[t-l-1][k][v] for l in range(self.L)])

                                    p_alpha = self.N_tdk[t][d][k] + alpha_theta
                                    p_beta_numer = self.N_tkv[t][k][v] + beta_phi
                                    p_beta_denom = self.N_tk[t][k] + sum(self.beta_tkl[t][k])
                                    self.p_ztd_n[t][k] = p_alpha * p_beta_numer / p_beta_denom
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
                            alpha_denom1 += digamma(self.N_td[t][d] * (itr + 1.0) + tmp_first_alpha_sum)
                        # end for d

                        alpha_numer = alpha_numer1 - self.D * digamma(tmp_first_alpha[k])
                        alpha_denom = alpha_denom1 - self.D * digamma(tmp_first_alpha_sum)
                        # nan の回避
                        try:
                            self.first_alpha[k] = tmp_first_alpha[k] * alpha_numer / alpha_denom
                            if self.first_alpha[k] <0:
                                print('a',tmp_first_alpha[k], alpha_numer1, self.D, digamma(tmp_first_alpha[k]), alpha_numer, alpha_denom)
                                return
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
                                # サンプリングサイズの平均を求める
                                if itr == iteration -1:
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
                                # サンプリングサイズの平均を求める
                                if itr == iteration - 1:
                                    self.theta_tdk[t][d][k] /= sample_size
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
                    # end if

                # end if
            # end iter
        # end for t
    # end def