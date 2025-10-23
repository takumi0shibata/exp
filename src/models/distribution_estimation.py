import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union

# ========= utilities (no weights) =========

def _make_x_grid(values: np.ndarray, n_points: int = 200) -> np.ndarray:
    vmin, vmax = float(np.min(values)), float(np.max(values))
    pad = max(1e-6, 0.02 * (vmax - vmin + 1e-6))
    return np.linspace(vmin - pad, vmax + pad, n_points)

def _ecdf_on_grid(y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(y)
    y_sorted = y[order]
    idx = np.searchsorted(y_sorted, x_grid, side="right") - 1
    idx = np.clip(idx, -1, len(y_sorted) - 1)
    return np.where(idx >= 0, (idx + 1) / len(y_sorted), 0.0)

def _silverman_bandwidth(y: np.ndarray) -> float:
    n = len(y)
    if n <= 1:
        return 1.0
    std = np.std(y, ddof=1)
    iqr = np.subtract(*np.percentile(y, [75, 25]))
    sigma = min(std, iqr / 1.349) if (std > 0 or iqr > 0) else max(std, 1.0)
    return 0.9 * sigma * n ** (-1/5)

def _kde_gaussian(y: np.ndarray, x_grid: np.ndarray, bw: Optional[float] = None) -> Tuple[np.ndarray, float]:
    if bw is None:
        bw = max(_silverman_bandwidth(y), 1e-8)
    x = x_grid[:, None]
    yi = y[None, :]
    z = (x - yi) / bw
    dens = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    f = dens.mean(axis=1) / bw
    return f, bw

# ========= config / result =========

@dataclass
class UnifiedConfig:
    max_samples: int = 100
    batch_size: int = 20
    ks_tol: Union[float, List[float]] = 0.03
    band_supwidth_tol: Union[float, List[float]] = 0.15
    B_boot: int = 400
    seed: Optional[int] = 0
    stop_aggregator: str = "all"  # "all" か "any"

@dataclass
class MetricResult:
    x_grid: np.ndarray
    ecdf: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    kde: np.ndarray
    bandwidth: float
    history: List[Dict[str, float]]

@dataclass
class UnifiedResult:
    stopped_n: int
    reason_per_metric: List[str]   # "stabilized" / "band_narrow_enough" / "pending"
    overall_reason: str            # "all_metrics_stabilized" / "all_bands_narrow_enough" / "mixed_ok" / "max_samples_reached"
    sample_indices: np.ndarray
    sample_scores: np.ndarray      # (n, K)
    metrics: List[MetricResult]

# ========= core =========

def estimate_distribution(
    pop_scores: np.ndarray,
    config: Optional[UnifiedConfig] = None
) -> UnifiedResult:
    """
    pop_scores: shape (N,) or (N, K) を受け付ける（重みなし）。
    ランダムサンプリングで各分布の ECDF/KDE を推定し、
    連続バッチ間のKS距離またはECDF 95%同時信頼帯の最大幅で停止判定。
    K=1 でも K>1 でも同一処理。停止の集約は stop_aggregator ("all"/"any")。
    """
    if config is None:
        config = UnifiedConfig()
    rng = np.random.default_rng(config.seed)

    Ypop = np.asarray(pop_scores)
    if Ypop.ndim == 1:
        Ypop = Ypop[:, None]
    N, K = Ypop.shape
    assert config.max_samples % config.batch_size == 0, "max_samples must be multiple of batch_size"

    # tol を列数 K にブロードキャスト
    ks_tols = np.array(config.ks_tol if isinstance(config.ks_tol, (list, tuple, np.ndarray)) else [config.ks_tol] * K, dtype=float)
    band_tols = np.array(config.band_supwidth_tol if isinstance(config.band_supwidth_tol, (list, tuple, np.ndarray)) else [config.band_supwidth_tol] * K, dtype=float)

    # 母集団レンジからメトリクスごとのグリッドを作成
    xgrids = [_make_x_grid(Ypop[:, k]) for k in range(K)]

    sampled = np.zeros(N, dtype=bool)
    chosen: List[int] = []
    prev_Fs: List[Optional[np.ndarray]] = [None] * K
    histories: List[List[Dict[str, float]]] = [[] for _ in range(K)]

    def agg_stop(flags: List[bool]) -> bool:
        return all(flags) if config.stop_aggregator == "all" else any(flags)

    # 1回のブートストラップで「全メトリクス同じブートストラップindex」を共有
    # （再現性と速度のため）
    def _bootstrap_bands_all(y_mat: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        B = config.B_boot
        n = y_mat.shape[0]
        # 事前に全 replicate のインデックスをまとめて作り、全メトリクスで使い回す
        boot_indices = rng.integers(0, n, size=(B, n))
        ql_list, qh_list = [None] * K, [None] * K
        for k in range(K):
            xg = xgrids[k]
            boot_vals = np.empty((B, len(xg)))
            col = y_mat[:, k]
            for b in range(B):
                idxb = boot_indices[b]
                Fb = _ecdf_on_grid(col[idxb], xg)
                boot_vals[b, :] = Fb
            ql_list[k] = np.percentile(boot_vals, 2.5, axis=0)
            qh_list[k] = np.percentile(boot_vals, 97.5, axis=0)
        return ql_list, qh_list

    overall_reason = "max_samples_reached"

    for _ in range(config.batch_size, config.max_samples + 1, config.batch_size):
        pool = np.where(~sampled)[0]
        draw = rng.choice(pool, size=config.batch_size, replace=False)
        sampled[draw] = True
        chosen.extend(draw.tolist())
        idx = np.array(chosen, dtype=int)
        Y = Ypop[idx, :]  # (n, K)

        # 現在の ECDF/KDE
        Fs, KDES, BWs = [], [], []
        for k in range(K):
            Fk = _ecdf_on_grid(Y[:, k], xgrids[k])
            kde_k, bw_k = _kde_gaussian(Y[:, k], xgrids[k])
            Fs.append(Fk); KDES.append(kde_k); BWs.append(bw_k)

        # ECDF 同時信頼帯
        ql_list, qh_list = _bootstrap_bands_all(Y)

        # 停止判定（メトリクスごと）
        ks_changes, supwidths, reasons, ok_flags = [], [], [], []
        for k in range(K):
            ks_change = float("inf") if prev_Fs[k] is None else float(np.max(np.abs(Fs[k] - prev_Fs[k])))
            supw = float(np.max(qh_list[k] - ql_list[k]))
            reasons.append("stabilized" if ks_change <= ks_tols[k] else ("band_narrow_enough" if supw <= band_tols[k] else "pending"))
            ok_flags.append((ks_change <= ks_tols[k]) or (supw <= band_tols[k]))
            ks_changes.append(ks_change); supwidths.append(supw)
            histories[k].append({"n": float(Y.shape[0]), "ks_change": ks_change, "band_supwidth": supw})

        if agg_stop(ok_flags):
            if all(ks_changes[k] <= ks_tols[k] for k in range(K)):
                overall_reason = "all_metrics_stabilized" if config.stop_aggregator == "all" else "some_metrics_stabilized"
            elif all(supwidths[k] <= band_tols[k] for k in range(K)):
                overall_reason = "all_bands_narrow_enough" if config.stop_aggregator == "all" else "some_bands_narrow_enough"
            else:
                overall_reason = "mixed_ok"

            metrics_out = []
            for k in range(K):
                metrics_out.append(MetricResult(
                    x_grid=xgrids[k],
                    ecdf=Fs[k],
                    ci_low=ql_list[k],
                    ci_high=qh_list[k],
                    kde=KDES[k],
                    bandwidth=float(BWs[k]),
                    history=histories[k]
                ))
            return UnifiedResult(
                stopped_n=Y.shape[0],
                reason_per_metric=reasons,
                overall_reason=overall_reason,
                sample_indices=idx,
                sample_scores=Y,
                metrics=metrics_out
            )

        prev_Fs = [Fs[k].copy() for k in range(K)]

    # max_samples 到達時
    idx = np.array(chosen, dtype=int)
    Y = Ypop[idx, :]
    Fs, KDES, BWs = [], [], []
    ql_list, qh_list = _bootstrap_bands_all(Y)
    reasons_final = []
    for k in range(K):
        Fk = _ecdf_on_grid(Y[:, k], xgrids[k])
        kde_k, bw_k = _kde_gaussian(Y[:, k], xgrids[k])
        Fs.append(Fk); KDES.append(kde_k); BWs.append(bw_k)
        last = histories[k][-1]
        if last["ks_change"] <= ks_tols[k]:
            reasons_final.append("stabilized")
        elif last["band_supwidth"] <= band_tols[k]:
            reasons_final.append("band_narrow_enough")
        else:
            reasons_final.append("pending")

    metrics_out = []
    for k in range(K):
        metrics_out.append(
            MetricResult(
                x_grid=xgrids[k],
                ecdf=Fs[k],
                ci_low=ql_list[k],
                ci_high=qh_list[k],
                kde=KDES[k],
                bandwidth=float(BWs[k]),
                history=histories[k]
            )
        )

    return UnifiedResult(
        stopped_n=Y.shape[0],
        reason_per_metric=reasons_final,
        overall_reason="max_samples_reached",
        sample_indices=idx,
        sample_scores=Y,
        metrics=metrics_out
    )
