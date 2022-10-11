"""変分ベイズ推定による LDA"""

import numpy as np
from scipy.special import psi


class VLDA():

    def __init__(self, K, N) -> None:
        self.K = K  # トピック数
        self.N = N  # 単語頻度行列
        self.D = self.N.shape[0]  # 文書数
        self.V = self.N.shape[1]  # 単語数

        # トピックの変分事後分布 q_dvk の集合
        self.q = np.zeros((self.D, self.V, self.K))
        # 変分事後分布パラメータ alpha, beta をランダムな正の値で初期化
        self.alpha = np.random.random_sample((self.D, self.K))
        self.beta = np.random.random_sample((self.K, self.V))
        pass

    # 変分事後分布の推定
    def variational_bayesian_inference(self, iter):
        # 適当な回数反復
        for i in range(iter):
            # 次ステップのパラメータを初期化
            alpha_next = np.zeros((self.D, self.K))
            beta_next = np.zeros((self.K, self.V))
            # 文書数でループ
            for d in range(self.D):
                # トピックの変分事後分布を計算 (全ての v, k で計算)
                tmp_q_alpha_dk = psi(self.alpha[d, :]) - psi(np.sum(self.alpha[d, :]))
                tmp_q_beta_kv = self.N[d, :] * (psi(self.beta) - psi(np.sum(self.beta, axis=1)))
                self.q[d, :, :] = np.exp(tmp_q_alpha_dk + tmp_q_beta_kv)
                self.q[d, :, :] = self.q[d, :, :] / np.sum(self.q[d, :, :], axis=1) # 正規化
                # alpha を更新
                alpha_next[d, :] = alpha_next[d, :] + np.sum(self.q[d, :, :], axis=0)
                # beta を更新
                beta_next = beta_next + self.N[d, :] * self.q[d, :, :]
            # パラメータ更新
            self.alpha = alpha_next
            self.beta = beta_next
