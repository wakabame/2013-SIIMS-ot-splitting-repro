"""制約集合への射影作用素。

MATLAB 版 ``@staggered/div_proj.m`` の移植(interp_proj はステップ 5-5
で追加予定)。
"""

from __future__ import annotations

from collections.abc import Sequence

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
