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
    """ソルバの出力。U はスタガード格子解、V = interp(U) 相当の中心格子。

    J / constr / min_density は収束診断の履歴で、diag_iters が記録した
    反復番号(0 始まり)。既定(diag_every=1)では全反復分が入る。
    """

    U: Staggered
    V: np.ndarray
    J: np.ndarray = field(repr=False)
    constr: np.ndarray = field(repr=False)
    min_density: np.ndarray = field(repr=False)
    diag_iters: np.ndarray = field(repr=False, default=None)


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
    diag_every: int = 1,
    verbose: bool = False,
) -> PPXAResult:
    """f0 から f1 への L2 最適輸送測地線を PPXA で計算する。

    f0, f1 は (N, P) の非負密度(総和が等しいこと)、Q は時間離散化数。
    gamma > 0 は prox のステップ、mu ∈ ]0, 2[ は緩和パラメータ
    (MATLAB 版の既定値 gamma = 1/230, mu = 1.98)。
    obstacle は (N, P, Q) のマスク(正の点は質量が通れない)。

    diag_every は収束診断(J, |div|, min f の記録)の間隔。診断は
    毎反復あたり全体の 1 割超のコストがあるため、間引くと速くなる
    (更新自体には影響しない)。最終反復は必ず記録される。
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
    omega = 1.0 / K  # 均等重み

    # 線形補間による初期化
    Xu = Staggered(d)
    Xu.M[2] = linear_init(f0, f1, Q)
    Xv = interp(Xu)
    Yu = [Xu.copy() for _ in range(K)]
    Yv = [Xv.copy() for _ in range(K)]

    # 更新はすべて in-place で行うため、状態を成分配列のフラットな
    # リストとして扱う(U の 3 成分 + 中心格子 V の計 4 本)。
    # X / Y[k] の配列オブジェクトは反復を通じて同一に保たれる。
    X = [*Xu.M, Xv]
    Y = [[*Yu[k].M, Yv[k]] for k in range(K)]
    nc = len(X)
    Z_bar = [np.empty_like(x) for x in X]  # Σ_k ω Z_k
    W = [np.empty_like(x) for x in X]  # 2 Z̄ − X
    T = [np.empty_like(x) for x in X]  # 汎用スクラッチ

    diag_iters: list[int] = []
    J_hist: list[float] = []
    constr_hist: list[float] = []
    min_hist: list[float] = []
    print_every = max(1, niter // 10)

    for it in range(niter):
        Z = []
        for k in range(K):
            zu, zv = proxes[k](Yu[k], Yv[k], gamma / omega)
            Z.append([*zu.M, zv])

        for c in range(nc):
            # 重み付き平均 Z̄ = Σ_k ω Z_k
            np.multiply(Z[0][c], omega, out=Z_bar[c])
            for k in range(1, K):
                np.multiply(Z[k][c], omega, out=T[c])
                Z_bar[c] += T[c]
            # W = 2 Z̄ − X
            np.multiply(Z_bar[c], 2.0, out=W[c])
            W[c] -= X[c]
            # 反射更新 Y_k += μ (W − Z_k)。Z[0] の U 成分は Y[0] と
            # 同一配列(prox_J は U を素通しする)だが、読み取りは
            # 書き込みより前に完了するため問題ない
            for k in range(K):
                np.subtract(W[c], Z[k][c], out=T[c])
                T[c] *= mu
                Y[k][c] += T[c]
            # X += μ (Z̄ − X)
            np.subtract(Z_bar[c], X[c], out=T[c])
            T[c] *= mu
            X[c] += T[c]

        # 収束診断(MATLAB 版と同じ量)。純粋なロギングなので
        # diag_every 反復ごとに間引ける
        if it % diag_every == 0 or it == niter - 1:
            Vc = interp(div_proj(Xu))
            diag_iters.append(it)
            J_hist.append(bb_energy(Vc, epsilon))
            constr_hist.append(float(np.linalg.norm(div(Xu))))
            min_hist.append(float(interp(Xu)[..., -1].min()))
            if verbose and (it + 1) % print_every < diag_every:
                print(
                    f"iter {it + 1:5d}/{niter}: J = {J_hist[-1]:.6e}, "
                    f"|div| = {constr_hist[-1]:.2e}, "
                    f"min f = {min_hist[-1]:.2e}"
                )

    return PPXAResult(
        U=Xu, V=interp(Xu), J=np.asarray(J_hist),
        constr=np.asarray(constr_hist),
        min_density=np.asarray(min_hist),
        diag_iters=np.asarray(diag_iters),
    )
