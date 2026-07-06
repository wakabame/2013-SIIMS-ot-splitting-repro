"""Benamou-Brenier 離散化のためのスタガード格子表現。

MATLAB 版の ``code/toolbox/@staggered`` クラスのうち、DR/PPXA ソルバで
使用する部分(ゼロ初期化と線形演算)のみを移植したもの。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Staggered:
    """d 次元スタガード格子上のベクトル場。

    第 k 成分 ``M[k]`` は軸 k 方向に半セルずれた格子上に定義され、
    形状は ``dim`` の第 k 成分だけを ``dim[k] + 1`` に置き換えたものになる。
    例えば ``dim = (N, P, Q)`` なら::

        M[0].shape == (N + 1, P, Q)
        M[1].shape == (N, P + 1, Q)
        M[2].shape == (N, P, Q + 1)
    """

    def __init__(
        self,
        dim: Sequence[int],
        M: Sequence[np.ndarray] | None = None,
    ) -> None:
        self.dim = tuple(int(n) for n in dim)
        if M is None:
            self.M = [
                np.zeros(self.component_shape(k)) for k in range(len(self.dim))
            ]
        else:
            if len(M) != len(self.dim):
                raise ValueError(
                    f"expected {len(self.dim)} components, got {len(M)}"
                )
            self.M = []
            for k, m in enumerate(M):
                m = np.asarray(m, dtype=float)
                if m.shape != self.component_shape(k):
                    raise ValueError(
                        f"component {k}: expected shape "
                        f"{self.component_shape(k)}, got {m.shape}"
                    )
                self.M.append(m)

    def component_shape(self, k: int) -> tuple[int, ...]:
        """第 k 成分の配列形状(軸 k 方向に 1 点多い)。"""
        return tuple(
            n + 1 if i == k else n for i, n in enumerate(self.dim)
        )

    def copy(self) -> Staggered:
        return Staggered(self.dim, [m.copy() for m in self.M])

    def _check_same_grid(self, other: Staggered) -> None:
        if self.dim != other.dim:
            raise ValueError(f"grid mismatch: {self.dim} vs {other.dim}")

    def __add__(self, other: Staggered) -> Staggered:
        self._check_same_grid(other)
        return Staggered(
            self.dim, [a + b for a, b in zip(self.M, other.M)]
        )

    def __sub__(self, other: Staggered) -> Staggered:
        self._check_same_grid(other)
        return Staggered(
            self.dim, [a - b for a, b in zip(self.M, other.M)]
        )

    def __mul__(self, scalar: float) -> Staggered:
        return Staggered(self.dim, [m * scalar for m in self.M])

    __rmul__ = __mul__

    def __neg__(self) -> Staggered:
        return self * (-1.0)

    def norm(self) -> float:
        """全成分をまとめた L2 ノルム(MATLAB 版 ``norm.m`` 相当)。"""
        return float(
            np.sqrt(sum(np.sum(m**2) for m in self.M))
        )

    def __repr__(self) -> str:
        return f"Staggered(dim={self.dim})"
