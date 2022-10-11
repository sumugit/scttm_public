"""Evaluation"""

from libcpp.vector cimport vector
import numpy as np
import sys
import math
import cython
cimport numpy as np
from libc.math cimport pow as c_pow
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
from threading import Thread
from cython.parallel import threadid
from tqdm import tqdm
import warnings
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval)

srand48(1234)
warnings.filterwarnings('error')

# 境界チェックを無視
@cython.boundscheck(False)
@cython.wraparound(False)

cdef class Eval():

    cdef public unsigned int K, D, V, T, L
    cdef public vector[vector[vector[int]]] N_tdv
    cdef public vector[vector[int]] N_td
    cdef public vector[vector[vector[double]]] theta_tdk
    cdef public vector[vector[vector[double]]] phi_tkv


    # パープレキシティ
    cdef public double[:] perlx
    cdef public double[:] accuracy

    # N-best-accuracy
    cdef public double[:,:,:] score


    def __init__(self, 
                vector[vector[vector[double]]] theta_tdk,
                vector[vector[vector[double]]] phi_tkv, 
                vector[vector[vector[int]]] N_tdv):
        
        cdef unsigned int t
        cdef unsigned int d
        
        """ 初期化 """
        self.N_tdv = N_tdv                                               # 単語のBOW行列 (perplexity の計算ならテストデータ)
        self.V = N_tdv.at(0).at(0).size()                                # 単語数
        self.theta_tdk = theta_tdk                                       # トピック分布 (perplexity の計算なら訓練データ)
        self.phi_tkv = phi_tkv                                           # 単語分布 (perplexity の計算なら訓練データ)
        self.K = theta_tdk.at(0).at(0).size()                            # トピック数
        self.T = self.N_tdv.size()                                       # 期間
        self.D = self.N_tdv.at(0).size()                                 # 文書数
        self.N_td = np.zeros(shape=(self.T, self.D)).astype('uint32')    # 時刻 t , ユーザー d の語彙数

        self.perlx = np.zeros(shape=(self.T))
        self.score = np.zeros(shape=(self.T, self.D, self.V))
        self.accuracy = np.zeros(shape=(self.T))

        for t in range(self.T):
            for d in range(self.D):
                self.N_td[t][d] = sum(self.N_tdv[t][d])
            # end for
        # end for
    # end def


    """ perplexity の計算 (最初の時刻におけるモデル選択) """
    def model_select_perplexity(self):

        cdef double numer, denom, sum_theta_phi, sum_v
        cdef list del_list

        numer = 0.0
        del_list = []
        # ユーザー数でループ    
        for d in range(self.D):
            sum_v = 0.0
            if self.N_td[0][d] == 0: del_list.append(d)
            # ユーザー d の語彙数でループ
            for v in range(self.V):
                for num in range(self.N_tdv[0][d][v]):
                    sum_theta_phi = 0.0
                    # トピック数でループ
                    for k in range(self.K):
                        sum_theta_phi += self.theta_tdk[0][d][k] * self.phi_tkv[0][k][v]
                    # end for k
                    sum_v += np.log(sum_theta_phi)
                # end for num
            # end for v
            numer += sum_v
        # end for d
        denom = sum(np.delete(self.N_td[0], del_list))
        return np.exp(-1.0*(numer)/denom)
    # end def


    def top_N_accuracy(self, unsigned int n):
        cdef unsigned int t, d, v, count, item
        cdef list phi_list, top_array, del_list
        cdef double prob

        # 単語の生成行列作成
        for t in range(self.T):
            for d in range(self.D):
                for v in range(self.V):
                    prob = 0.0
                    for k in range(self.K):
                        prob += self.theta_tdk[t][d][k] * self.phi_tkv[t][k][v] 
                    # end for k
                    self.score[t][d][v] = prob
                # end for v
            # end for d
        # end for t

        # N-best-Accuracy の評価
        for t in tqdm(range(1, self.T)):

            del_list = []
            count = 0
            
            for d in range(self.D):
                top_array = list(np.argsort(-np.array(self.score)[t-1][d])[:n])
                
                if self.N_td[t][d] == 0:
                    del_list.append(d)
                    continue
                
                for v in range(self.V):

                    if self.N_tdv[t][d][v] > 0:
                        if v in top_array:
                            count += 1
                            break
                        # end if
                    # end if
                # end for v
            # end for d
            self.accuracy[t] = 100 * count/(self.D - len(del_list))
        # end for t
    # end def