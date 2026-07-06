"""スタガード格子上の基本作用素(div, interp)。

MATLAB 版 ``@staggered/div.m`` と ``@staggered/interp.m`` の移植。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .grid import Staggered


def div(u: Staggered, lengths: Sequence[float] | None = None) -> np.ndarray:
    """離散発散作用素。

    第 k 成分の軸 k 方向の前進差分に格子数 ``dim[k] / lengths[k]`` を
    掛けた和。戻り値は中心格子上の配列(形状 ``u.dim``)。
    """
    if lengths is None:
        lengths = (1.0,) * len(u.dim)
    v = np.zeros(u.dim)
    for k, n in enumerate(u.dim):
        v += np.diff(u.M[k], axis=k) * (n / lengths[k])
    return v


def interp(u: Staggered) -> np.ndarray:
    """スタガード格子から中心格子への中点補間。

    各成分を軸方向に隣接 2 点平均し、最終軸に積む。
    戻り値の形状は ``u.dim + (d,)``(d は次元数)。
    """
    components = []
    for k, m in enumerate(u.M):
        lo = [slice(None)] * m.ndim
        hi = [slice(None)] * m.ndim
        lo[k] = slice(None, -1)
        hi[k] = slice(1, None)
        components.append((m[tuple(lo)] + m[tuple(hi)]) / 2)
    return np.stack(components, axis=-1)
