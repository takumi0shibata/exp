import numpy as np
from dataclasses import dataclass

@dataclass
class SemiDiscreteOT1D:
    """連続サンプル x と整数ラベル y_int から
    1次元の半離散OT写像 T: R -> Z を学習する。

    - 学習: y_int の頻度から離散分布の質量 b_i を推定し、
            x の ECDF を構築
    - 予測: u = F_emp(x_new) を計算し、累積質量に基づいて
            対応する整数値を返す（決定写像）
    """
    eps: float = 1e-12  # ECDF端の数値安定化

    def fit(self, x: np.ndarray, y_int: np.ndarray):
        x = np.asarray(x).ravel()
        y_int = np.asarray(y_int).ravel()
        if x.shape[0] != y_int.shape[0]:
            raise ValueError("x と y_int は同じ長さである必要があります。")

        # ---- 離散側の質点と質量（確率）を推定 ----
        labels, counts = np.unique(y_int.astype(int), return_counts=True)
        b = counts / counts.sum()              # 質量（確率）
        cum_b = np.cumsum(b)                   # 累積確率 B_i

        # ---- 連続側のECDFを構築 ----
        x_sorted = np.sort(x)
        self._x_sorted = x_sorted
        self._n = x_sorted.size
        self.labels_ = labels                  # 離散の値（整数）
        self.b_ = b
        self.cum_b_ = cum_b
        return self

    def _ecdf(self, t):
        """経験CDF F_emp(t) = P_hat(X <= t)"""
        xs = self._x_sorted
        # right側で数えることで P(X<=t)
        idx = np.searchsorted(xs, np.asarray(t), side="right")
        u = idx / self._n
        return np.clip(u, self.eps, 1.0 - self.eps)

    def predict(self, x_new):
        """単発 or ベクトル入力どちらも可。対応する整数を返す。"""
        u = self._ecdf(x_new)
        # u が入る累積区間 (B_{i-1}, B_i] を二分探索で取得
        i = np.searchsorted(self.cum_b_, u, side="right")
        return self.labels_[i]

    __call__ = predict  # T(x) として関数的にも使えるように

    # おまけ: 返すインデックスではなくカテゴリ確率（one-hot）
    def predict_proba(self, x_new):
        i = np.searchsorted(self.cum_b_, self._ecdf(x_new), side="right")
        k = len(self.labels_)
        if np.isscalar(i):
            p = np.zeros(k, dtype=float)
            p[i] = 1.0
            return p
        P = np.zeros((len(i), k), dtype=float)
        P[np.arange(len(i)), i] = 1.0
        return P