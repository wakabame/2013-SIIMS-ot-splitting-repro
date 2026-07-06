"""BB エネルギー J の近接作用素。

MATLAB 版 ``proxJ.m`` の alpha = 1(L2-Wasserstein)分岐の移植。
3 次方程式は ``poly_root_new.m`` の逐点計算の代わりに Cardano の
公式で全格子点を一括処理する。
"""

from __future__ import annotations

import numpy as np


def solve_cubic_largest_root(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """``x³ + a x² + b x + c = 0`` の最大実根を要素ごとに求める。

    Cardano の公式によるベクトル化実装。判別式が正なら唯一の実根、
    非正なら 3 実根のうち最大のもの ``2r·cos(θ/3) − a/3`` を返す。
    prox_J の 3 次方程式(c ≤ 0)では最大実根が唯一の非負根になる。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    # 標準形 t³ + p t + q = 0 (x = t - a/3)
    p = b - a**2 / 3.0
    q = 2.0 * a**3 / 27.0 - a * b / 3.0 + c
    delta = (q / 2.0) ** 2 + (p / 3.0) ** 3

    # delta > 0: 実根 1 つ(Cardano)
    sqrt_delta = np.sqrt(np.maximum(delta, 0.0))
    t_single = np.cbrt(-q / 2.0 + sqrt_delta) + np.cbrt(
        -q / 2.0 - sqrt_delta
    )

    # delta <= 0: 実根 3 つ(三角関数解)。最大根は 2r·cos(θ/3)
    r = np.sqrt(np.maximum(-p / 3.0, 0.0))
    r3 = r**3
    safe_r3 = np.where(r3 > 0.0, r3, 1.0)
    cos_theta = np.clip(np.where(r3 > 0.0, -q / 2.0 / safe_r3, 1.0), -1.0, 1.0)
    t_triple = 2.0 * r * np.cos(np.arccos(cos_theta) / 3.0)

    t = np.where(delta > 0.0, t_single, t_triple)
    return t - a / 3.0


def prox_j(
    V0: np.ndarray,
    gamma: float,
    epsilon: float,
    obstacle: np.ndarray | None = None,
) -> np.ndarray:
    """``prox_{γJ}(V0)``:BB エネルギーの近接作用素(alpha = 1)。

    J(m, f) = Σ |m|²/f + χ_{f ≥ ε}。V0 の最終軸は
    ``(m_1, …, m_{d-1}, f)`` の順(MATLAB 版と同じレイアウト)。

    密度 f は 3 次方程式
    ``f³ + (4γ − f0) f² + (4γ² − 4γ f0) f − γ|m0|² − 4γ² f0 = 0``
    の正根、運動量は ``m = m0 / (1 + 2γ/f)``。
    obstacle が正の点では f = ε に固定する(質量を通さない)。
    """
    V0 = np.asarray(V0, dtype=float)
    m0 = V0[..., :-1]
    f0 = V0[..., -1]
    msq = np.sum(m0**2, axis=-1)

    f = solve_cubic_largest_root(
        4.0 * gamma - f0,
        4.0 * gamma**2 - 4.0 * gamma * f0,
        -gamma * msq - 4.0 * gamma**2 * f0,
    )
    f = np.maximum(f, epsilon)
    if obstacle is not None:
        f = np.where(obstacle > 0, epsilon, f)

    m = m0 / (1.0 + 2.0 * gamma / f)[..., None]
    return np.concatenate([m, f[..., None]], axis=-1)
