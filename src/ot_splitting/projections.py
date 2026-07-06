"""制約集合への射影作用素。

MATLAB 版 ``@staggered/div_proj.m`` と ``@staggered/interp_proj.m`` の
移植。
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from .grid import Staggered
from .operators import div
from .poisson import poisson_neumann


def div_proj(
    u: Staggered, lengths: Sequence[float] | None = None
) -> Staggered:
    """発散ゼロ制約 ``{div v = 0}`` への直交射影。

    Poisson 方程式 ``Δp = div u`` (Neumann 境界条件)を解き、
    圧力勾配を各成分の**内部点のみ**から引く。境界スライス
    (各成分の軸方向両端。時間成分 ``M[-1]`` では初期・最終密度
    f0, f1 に相当)は変更しない。

    注意: 発散の定数モード(空間平均)は境界成分の総フラックスで
    決まるため射影では消えない(MATLAB 版と同じ)。BB 問題では
    f0 と f1 の総質量が等しいため常にゼロであり、問題にならない。
    """
    if lengths is None:
        lengths = (1.0,) * len(u.dim)
    p = poisson_neumann(-div(u, lengths), lengths)
    v = u.copy()
    for k, (n, length) in enumerate(zip(u.dim, lengths)):
        interior = [slice(None)] * len(u.dim)
        interior[k] = slice(1, -1)
        v.M[k][tuple(interior)] -= np.diff(p, axis=k) * (n / length)
    return v


def _interp_matrix(n: int) -> np.ndarray:
    """中点補間行列 S (形状 (n, n+1)):(S u)_j = (u_j + u_{j+1}) / 2。"""
    S = np.zeros((n, n + 1))
    idx = np.arange(n)
    S[idx, idx] = 0.5
    S[idx, idx + 1] = 0.5
    return S


@lru_cache(maxsize=None)
def _projection_operator(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """制約 ``{S u = v, u_0 = a, u_n = b}`` への射影行列を返す。

    MATLAB 版の persistent キャッシュに相当(格子サイズ n をキーに
    lru_cache で保持)。x = [u; v] (長さ 2n+1)に対し射影は
    ``B x + pA g`` (g = [0…0; a; b])。行列積を効率よく適用できる
    よう、あらかじめブロックに分割した連続配列
    ``(Bu = B[:, :n+1], Bv = B[:, n+1:], pA_bc = pA[:, n:])``
    を返す(g の先頭 n 行はゼロなので pA は末尾 2 列のみ使う)。
    """
    S = _interp_matrix(n)
    A = np.zeros((n + 2, 2 * n + 1))
    A[:n, : n + 1] = S
    A[:n, n + 1 :] = -np.eye(n)
    A[n, 0] = 1.0
    A[n + 1, n] = 1.0
    pA = np.linalg.pinv(A)
    B = np.eye(2 * n + 1) - pA @ A
    return (
        np.ascontiguousarray(B[:, : n + 1]),
        np.ascontiguousarray(B[:, n + 1 :]),
        np.ascontiguousarray(pA[:, n:]),
    )


def interp_proj(
    u0: Staggered, v0: np.ndarray
) -> tuple[Staggered, np.ndarray]:
    """補間整合制約への直交射影。

    制約集合 ``{(u, v) : v_k = S_k u_k, u_k の軸 k 方向両端 = u0 の値}``
    に ``(u0, v0)`` を射影する。成分 k は軸 k の制約にのみ現れるため、
    軸ごとに独立な小規模射影(行列サイズ 2n+1)に分解できる。

    u0 はスタガード格子、v0 は中心格子(形状 ``dim + (d,)``)。
    戻り値も同じ形式のペア。
    """
    dim = u0.dim
    u = u0.copy()
    v = np.array(v0, dtype=float)
    for k, n in enumerate(dim):
        Bu, Bv, pA_bc = _projection_operator(n)
        # 軸 k を先頭に移して (格子点数, 残り) の 2 次元に畳む
        uc = np.moveaxis(u0.M[k], k, 0)
        vc = np.moveaxis(v0[..., k], k, 0)
        moved_u_shape = uc.shape
        moved_v_shape = vc.shape
        uc = np.ascontiguousarray(uc).reshape(n + 1, -1)
        vc = np.ascontiguousarray(vc).reshape(n, -1)

        # y = B @ [u; v] + pA @ g。x の連結を避けて B をブロック別に
        # 適用する。g は境界 2 行(u の両端値)以外ゼロなので、
        # pA @ g は pA の末尾 2 列との積に縮約できる
        y = Bu @ uc
        y += Bv @ vc
        y += pA_bc @ np.stack([uc[0], uc[-1]])

        u.M[k] = np.moveaxis(
            y[: n + 1].reshape(moved_u_shape), 0, k
        )
        v[..., k] = np.moveaxis(
            y[n + 1 :].reshape(moved_v_shape), 0, k
        )
    return u, v
