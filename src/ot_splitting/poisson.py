"""DCT による Neumann 境界条件付き Poisson ソルバ。

MATLAB 版 ``poisson2d_Neumann.m`` / ``poisson3d_Neumann.m`` の移植
(次元数によらない一般化)。``mirt_dctn`` は直交正規化 DCT-II と等価
なので、``scipy.fft.dctn(type=2, norm='ortho')`` で置き換える。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.fft import dctn, idctn


def poisson_neumann(
    f: np.ndarray, lengths: Sequence[float] | None = None
) -> np.ndarray:
    """一様格子上で ``Δu = -f`` (Neumann 境界条件)を解く。

    MATLAB 版と同じ符号規約:戻り値 u は離散 Neumann Laplacian Δ に
    対して ``Δu = -f`` を満たす(``div_proj`` では ``p = poisson(-div u)``
    として ``Δp = div u`` を得るのに使う)。

    Laplacian の DCT-II 基底での固有値は ``(2cos(πk/n) - 2) / h²``
    (h = lengths[axis] / n)。ゼロ固有値(定数モード)は 1 に
    置き換えるため、解の定数成分は意味を持たない。
    """
    f = np.asarray(f, dtype=float)
    if lengths is None:
        lengths = (1.0,) * f.ndim

    denom = np.zeros(f.shape)
    for axis, (n, length) in enumerate(zip(f.shape, lengths)):
        h = length / n
        eig = (2.0 * np.cos(np.pi * np.arange(n) / n) - 2.0) / h**2
        shape = [1] * f.ndim
        shape[axis] = n
        denom = denom + eig.reshape(shape)
    denom[denom == 0] = 1.0

    fhat = dctn(f, type=2, norm="ortho")
    uhat = -fhat / denom
    return idctn(uhat, type=2, norm="ortho")
