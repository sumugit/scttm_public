"""トピックモデルの最尤推定"""

import numpy as np
from tqdm import tqdm

class PLSA():

    def __init__(self, K, N) -> None:
        self.K = K # トピック数
        self.N = N # 単語頻度行列
        self.D = self.N.shape[0]  # 文書数
        self.V = self.N.shape[1]  # 単語数

        # 負担率 q_dvk の集合
        self.q = np.zeros((self.D, self.V, self.K))
        # トピック分布 theta の初期化
        self.theta = np.random.random_sample((self.D, self.K))
        for d in np.arange(self.D):
            self.theta[d, :] = self.theta[d, :] / np.sum(self.theta[d, :])
        # 単語分布 phi の初期化
        self.phi = np.random.random_sample((self.K, self.V))
        for k in np.arange(self.K):
            self.phi[k, :] = self.phi[k, :] / np.sum(self.phi[k, :])
        pass

    # EMアルゴリズムで推定
    def em_estimate(self, iter):
        # 適当な回数反復
        for i in np.arange(iter):
            # 次ステップのパラメータを0に初期化
            theta_next = np.zeros((self.D, self.K))
            phi_next = np.zeros((self.K, self.V))
            # 文書数でループ
            for d in tqdm(np.arange(self.D)):
                # 単語の種類数でループ
                for v in np.arange(self.V):
                    # トピック数でループ
                    for k in np.arange(self.K):
                        # 負担率を計算
                        q_numer = self.theta[d, k] * self.phi[k, v]**self.N[d, v] # 分子
                        q_denom = np.sum(self.theta[d,:] * self.phi[:, v]**self.N[d, v]) # 分母
                        self.q[d, v, k] = q_numer / q_denom
                        # theta_dk の更新
                        theta_next[d, k] = theta_next[d, k] + self.q[d, v, k]*self.N[d, v]
                        # 単語の種類数でループ
                        for v2 in np.arange(self.V):
                            # phi_kv の更新
                            phi_next[k, v2] = phi_next[k, v2] + self.q[d, v2, k]*self.N[d, v2]
            # パラメータ更新
            self.theta = theta_next
            self.phi = phi_next
            # パラメータ正規化
            for d in np.arange(self.D):
                self.theta[d, :] = self.theta[d, :] / np.sum(self.theta[d, :])
            for k in np.arange(self.K):
                self.phi[k, :] = self.phi[k, :] / np.sum(self.phi[k, :])
