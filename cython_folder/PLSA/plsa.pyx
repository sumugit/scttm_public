"""PLSA with Cython"""

from libcpp.vector cimport vector
import numpy as np
import cython
cimport numpy as np
from libc.math cimport pow
from tqdm import tqdm

# 境界チェックを無視
@cython.boundscheck(False)
@cython.wraparound(False)

cdef class PLSA():

    cdef public int K, D, V
    cdef public vector[vector[int]] N
    cdef public double[:,:] theta_dk
    cdef public double[:,:,:] q_dvk
    cdef public double[:,:] phi_kv

    def __init__(self, unsigned int K, vector[vector[int]] N):
        cdef double theta_dk_denom
        cdef double phi_kv_denom
        cdef unsigned int d, k, v

        self.K = K # トピック数
        self.N = N # 単語頻度行列
        self.D = self.N.size()  # 文書数
        self.V = self.N.at(0).size()  # 単語数
        # 負担率 q_dvk の集合
        self.q_dvk = np.zeros(shape=(self.D, self.V, self.K))
        # トピック分布 theta の初期化
        self.theta_dk = np.random.random_sample((self.D, self.K))
        for d in np.arange(self.D):
            theta_dk_denom = np.sum(self.theta_dk[d, :])
            for k in np.arange(self.K):
                self.theta_dk[d, k] = self.theta_dk[d, k] / theta_dk_denom
        # 単語分布 phi の初期化
        self.phi_kv = np.random.random_sample((self.K, self.V))
        for k in np.arange(self.K):
            phi_kv_denom = np.sum(self.phi_kv[k, :])
            for v in np.arange(self.V):
                self.phi_kv[k, v] = self.phi_kv[k, v] / phi_kv_denom
        pass
    
    # EMアルゴリズムで推定
    def em_estimate(self, unsigned int iter):
        cdef double[:,:] theta_dk_next
        cdef double[:,:] phi_kv_next
        cdef double q_numer
        cdef double q_denom
        cdef double theta_dk_denom
        cdef double phi_kv_denom
        cdef unsigned int i, d, v, v2, k, k2, iteration = iter

        # 適当な反復回数
        for i in range(iteration):
            # 次ステップのパラメータを0に初期化
            theta_dk_next = np.zeros((self.D, self.K))
            phi_kv_next = np.zeros((self.K, self.V))
            # 文書数でループ
            for d in tqdm(range(self.D)):
                # 単語の種類数でループ
                for v in range(self.V):
                    # トピック数でループ
                    for k in range(self.K):
                        # 負担率を計算
                        q_numer = self.theta_dk[d, k] * pow(self.phi_kv[k, v], self.N[d][v]) # 分子
                        q_denom = 0
                        for k2 in range(self.K):
                            q_denom += self.theta_dk[d, k2] * pow(self.phi_kv[k2, v], self.N[d][v]) # 分母
                        try:
                            self.q_dvk[d, v, k] = q_numer / q_denom
                        except:
                            self.q_dvk[d, v, k] = q_numer / (0.001 + q_denom)
                        # theta_dk の更新
                        theta_dk_next[d, k] = theta_dk_next[d, k] + self.q_dvk[d, v, k] * self.N[d][v]
                        # 単語の種類数でループ
                        for v2 in range(self.V):
                            # phi_kv の更新
                            phi_kv_next[k, v2] = phi_kv_next[k, v2] + self.q_dvk[d, v2, k] * self.N[d][v2]
            # パラメータ更新
            self.theta_dk = theta_dk_next
            self.phi_kv = phi_kv_next
            # パラメータ正規化
            for d in range(self.D):
                theta_dk_denom = np.sum(self.theta_dk[d, :])
                for k in range(self.K):
                    try:
                        self.theta_dk[d, k] = self.theta_dk[d, k] / theta_dk_denom
                    except:
                        self.theta_dk[d, k] = self.theta_dk[d, k] / (0.001 + theta_dk_denom)
            for k in range(self.K):
                phi_kv_denom = np.sum(self.phi_kv[k, :])
                for v in range(self.V):
                    try:
                        self.phi_kv[k, v] = self.phi_kv[k, v] / phi_kv_denom
                    except:
                        self.phi_kv[k, v] = self.phi_kv[k, v] / (0.001 + phi_kv_denom)