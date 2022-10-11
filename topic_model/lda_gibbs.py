"""崩壊型ギブスサンプリングによる LDA"""

import numpy as np
from scipy.special import psi

class GLDA():

    def __init__(self, K, N) -> None:
        self.K = K  # トピック数
        self.N_dv = N  # 単語頻度行列
        self.N_d = np.sum(self.N_dv, axis=1) # 行毎の和
        self.D = self.N_dv.shape[0] # 文書数
        self.V = self.N_dv.shape[1] # 単語数

        # ハイパーパラメータ alpha, betaの初期化
        self.alpha = np.repeat(2, self.K) # 一様でない
        self.beta = 2 # 一様

        # 文書 d の語彙 v に割り当てられたトピック z_dn の集合 (0で初期化)
        self.z = np.zeros(self.D, self.V, np.max(self.N_dv))
        # 文書 d においてトピック k が割り当てられた単語数 N_dk の集合 (0で初期化)
        self.N_dk = np.zeros(self.D, self.K)
        # 文書全体でトピック k が割り当てられた語彙 v の出現回数 N_kv の集合 (0で初期化)
        self.N_kv = np.zeros(self.K, self.V)
        # 全文書でトピック k が割り当てられた単語数 N_k のベクトル (0で初期化)
        self.N_k = np.repeat(0, self.K)
        pass

    # 崩壊型ギブスサンプリング
    def collapsed_gibbs_sampling(self, iter):
        # 受け皿の用意
        p = [None for _ in range(len(self.K))]
        for i in iter:
            # 新たに割り当てられたトピックに関するカウントを初期化
            new_N_dk = np.zeros(self.D, self.K)
            new_N_kv = np.zeros(self.K, self.V)
            new_N_k = np.repeat(0, self.K)
            # 文書数でループ
            for d in range(self.D):
                # 単語数でループ
                for v in range(self.V):
                    # 出現回数 : N_dv > 0 のときのみ
                    if self.N_dv[d, v] > 0:
                        # 各単語の出現回数 : 1,...,N_dv
                        for n_dv in self.N_dv[d, v]:
                            # 現ステップの計算のためにカウントを移す
                            tmp_N_dk = self.N_dk
                            tmp_N_kv = self.N_kv
                            tmp_N_k = self.N_k
                            # 初回を飛ばす処理
                            if self.z[d, v, n_dv] > 0:
                                # 前ステップで文書 d の語彙 v に割り当てられたトピックを k に代入
                                k = self.z[d, v, n_dv]
                                # 文書 d の語彙 v の分のカウントを引く
                                tmp_N_dk[d, k] = self.N_dk[d, k] - 1
                                tmp_N_kv[k, v] = self.N_kv[k, v] - 1
                                tmp_N_k[k] = self.N_k[k] - 1
                            # 各トピック
                            for k in range(self.K):
                                # サンプリング確率を計算
                                tmp_p_alpha = tmp_N_dk[d, k] + self.alpha[k]
                                tmp_p_beta_numer = tmp_N_kv[k, v] + self.beta
                                tmp_p_beta_denom = tmp_N_k[k] + self.beta*self.V
                                p[k] = tmp_p_alpha * tmp_p_beta_numer / tmp_p_beta_denom
                            # トピックをサンプリング
                            self.z[d, v, n_dv] = np.argmax(np.random.multinomial(n=1, pvals=p, size=self.K))
                            # 新たに割り当てられたトピックを k に代入
                            k = self.z[d, v, n_dv]
                            # 文書 d の語彙 v の分のカウントを加える
                            new_N_dk[d, k] = new_N_dk[d, k] + 1
                            new_N_kv[k, v] = new_N_kv[k, v] + 1
                            new_N_k[k] = new_N_k[k] + 1
            
            # トピック集合とカウントを更新
            self.N_dk = new_N_dk
            self.N_kv = new_N_kv
            self.N_k = new_N_k

            # ハイパーパラメータの更新 (alpha)
            tmp_alpha_numer1 = np.sum(psi(self.N_dk + self.alpha), axis=1) # 分子第一項
            tmp_alpha_numer2 = self.D * psi(self.alpha) # 分子第二項
            tmp_alpha_denom1 = np.sum(psi(self.N_d + np.sum(self.alpha))) # 分母第一項
            tmp_alpha_denom2 = self.D * psi(np.sum(self.alpha)) # 分母第二項
            self.alpha = self.alpha * (tmp_alpha_numer1 - tmp_alpha_numer2) / (tmp_alpha_denom1 - tmp_alpha_denom2) # ハイパラ更新

            # ハイパーパラメータの更新 (beta)
            tmp_beta_numer1 =  np.sum(psi(self.N_kv + self.beta)) # 分子第一項
            tmp_beta_numer2 = self.K * self.V * psi(self.beta) # 分子第二項
            tmp_beta_denom1 = self.V * np.sum(psi(self.N_k + self.beta * self.V)) # 分母第一項
            tmp_beta_denom2 = self.K * self.V * psi(self.beta * self.V) # 分母第二項
            self.beta = self.beta * (tmp_beta_numer1 - tmp_beta_numer2) / (tmp_beta_denom1 - tmp_beta_denom2) # ハイパラ更新
            

                            
