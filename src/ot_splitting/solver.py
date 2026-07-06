"""PPXA(並列 Douglas–Rachford)による BB エネルギー最小化。

MATLAB 版 ``test_bb_dr.m`` の反復ループの移植。3 つの近接作用素

* prox_J   : BB エネルギー(運動量の縮小 + 密度の 3 次方程式)
* prox_I   : 発散ゼロ制約への射影(div_proj)
* prox_S   : 補間整合制約への射影(interp_proj)

を並列に適用し、重み付き平均への反射で更新する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Staggered
from .operators import div, interp
from .projections import div_proj, interp_proj
from .prox import prox_j


def linear_init(f0: np.ndarray, f1: np.ndarray, Q: int) -> np.ndarray:
    """密度の線形補間による初期化(形状 (N, P, Q+1))。"""
    t = np.linspace(0.0, 1.0, Q + 1).reshape(1, 1, Q + 1)
    return (1.0 - t) * f0[:, :, None] + t * f1[:, :, None]


def bb_energy(V: np.ndarray, epsilon: float) -> float:
    """BB エネルギー J(V) = Σ |m|² / max(f, ε)(MATLAB 版と同じ規約)。"""
    m_sq = np.sum(V[..., :-1] ** 2, axis=-1)
    f = np.maximum(V[..., -1], max(epsilon, 1e-10))
    return float(np.sum(m_sq / f))


@dataclass
class PPXAResult:
    """ソルバの出力。U はスタガード格子解、V = interp(U) 相当の中心格子。"""

    U: Staggered
    V: np.ndarray
    J: np.ndarray = field(repr=False)
    constr: np.ndarray = field(repr=False)
    min_density: np.ndarray = field(repr=False)


def solve_ppxa(
    f0: np.ndarray,
    f1: np.ndarray,
    Q: int,
    *,
    gamma: float = 1.0 / 230.0,
    mu: float = 1.98,
    niter: int = 1000,
    epsilon: float = 1e-8,
    obstacle: np.ndarray | None = None,
    verbose: bool = False,
) -> PPXAResult:
    """f0 から f1 への L2 最適輸送測地線を PPXA で計算する。

    f0, f1 は (N, P) の非負密度(総和が等しいこと)、Q は時間離散化数。
    gamma > 0 は prox のステップ、mu ∈ ]0, 2[ は緩和パラメータ
    (MATLAB 版の既定値 gamma = 1/230, mu = 1.98)。
    obstacle は (N, P, Q) のマスク(正の点は質量が通れない)。
    """
    if f0.shape != f1.shape:
        raise ValueError(f"shape mismatch: {f0.shape} vs {f1.shape}")
    N, P = f0.shape
    d = (N, P, Q)

    proxes = (
        # k=0: J(V) — U はそのまま、V に prox_J
        lambda U, V, g: (U, prox_j(V, g, epsilon, obstacle)),
        # k=1: div = 0 — U を射影、V はそのまま
        lambda U, V, g: (div_proj(U), V),
        # k=2: V = interp(U)
        lambda U, V, g: interp_proj(U, V),
    )
    K = len(proxes)
    omega = np.full(K, 1.0 / K)

    # 線形補間による初期化
    Xu = Staggered(d)
    Xu.M[2] = linear_init(f0, f1, Q)
    Xv = interp(Xu)
    Yu = [Xu.copy() for _ in range(K)]
    Yv = [Xv.copy() for _ in range(K)]

    J_hist = np.empty(niter)
    constr_hist = np.empty(niter)
    min_hist = np.empty(niter)

    for it in range(niter):
        Zu = [None] * K
        Zv = [None] * K
        for k in range(K):
            Zu[k], Zv[k] = proxes[k](Yu[k], Yv[k], gamma / omega[k])

        # 重み付き平均 Z = Σ_k ω_k Z_k
        Zu_bar = Staggered(d)
        Zv_bar = np.zeros_like(Xv)
        for k in range(K):
            Zu_bar = Zu_bar + omega[k] * Zu[k]
            Zv_bar = Zv_bar + omega[k] * Zv[k]

        # 反射更新 Y_k += μ (2Z - X - Z_k)、X += μ (Z - X)
        for k in range(K):
            Yu[k] = Yu[k] + mu * (2.0 * Zu_bar - Xu - Zu[k])
            Yv[k] = Yv[k] + mu * (2.0 * Zv_bar - Xv - Zv[k])
        Xu = Xu + mu * (Zu_bar - Xu)
        Xv = Xv + mu * (Zv_bar - Xv)

        # 収束診断(MATLAB 版と同じ量)
        Vc = interp(div_proj(Xu))
        J_hist[it] = bb_energy(Vc, epsilon)
        constr_hist[it] = float(np.linalg.norm(div(Xu)))
        min_hist[it] = float(interp(Xu)[..., -1].min())
        if verbose and (it + 1) % max(1, niter // 10) == 0:
            print(
                f"iter {it + 1:5d}/{niter}: J = {J_hist[it]:.6e}, "
                f"|div| = {constr_hist[it]:.2e}, "
                f"min f = {min_hist[it]:.2e}"
            )

    return PPXAResult(
        U=Xu, V=interp(Xu), J=J_hist, constr=constr_hist,
        min_density=min_hist,
    )
