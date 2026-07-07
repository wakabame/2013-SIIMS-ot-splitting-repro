"""BB エネルギー J の近接作用素。

MATLAB 版 ``proxJ.m`` の alpha = 1(L2-Wasserstein)分岐の移植。
3 次方程式は ``poly_root_new.m`` の逐点計算の代わりに Cardano の
公式で全格子点を一括処理する。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


def solve_cubic_largest_root(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """``x³ + a x² + b x + c = 0`` の最大実根を要素ごとに求める。

    Cardano の公式によるベクトル化実装。判別式が正なら唯一の実根、
    非正なら 3 実根のうち最大のもの ``2r·cos(θ/3) − a/3`` を返す。
    prox_J の 3 次方程式(c ≤ 0)では最大実根が唯一の非負根になる。
    """
    out_shape = np.broadcast_shapes(
        np.shape(a), np.shape(b), np.shape(c)
    )
    a = np.broadcast_to(np.asarray(a, dtype=float), out_shape).reshape(-1)
    b = np.broadcast_to(np.asarray(b, dtype=float), out_shape).reshape(-1)
    c = np.broadcast_to(np.asarray(c, dtype=float), out_shape).reshape(-1)

    # 標準形 t³ + p t + q = 0 (x = t - a/3)
    # 注意: 配列の整数べき(**2, **3)は値域によって libm pow の
    # 低速経路に落ちる(実測で 50 倍超の差)ため、明示的な乗算で書く
    a2 = a * a
    p = b - a2 / 3.0
    q = 2.0 * (a2 * a) / 27.0 - a * b / 3.0 + c
    half_q = q / 2.0
    third_p = p / 3.0
    delta = half_q * half_q + third_p * third_p * third_p

    # delta > 0: 実根 1 つ(Cardano)。prox_J の係数では実運用上
    # ほぼ全点がこちらに落ちるため、まず全点を Cardano で計算する。
    # 2 つの立方根 u1, u2 は u1·u2 = -p/3 を満たすので、片方 u から
    # t = u - p/(3u) が得られ、高コストな cbrt を 1 回に減らせる。
    # 相殺誤差を避けるため -q/2 と同符号側(絶対値が大きい方)の
    # 被開立数を選ぶ。u = 0 となるのは q = 0 かつ delta = 0 のとき
    # だけで、その点は下の delta <= 0 分岐が必ず上書きする
    sqrt_delta = np.sqrt(np.maximum(delta, 0.0))
    u_root = np.cbrt(-half_q + np.copysign(sqrt_delta, -q))
    safe_u = np.where(u_root != 0.0, u_root, 1.0)
    t = u_root - third_p / safe_u

    # delta <= 0: 実根 3 つ(三角関数解)。最大根は 2r·cos(θ/3)。
    # 該当点がある場合のみ、その部分集合に対して計算する
    triple = delta <= 0.0
    if np.any(triple):
        pm = p[triple]
        qm = q[triple]
        r = np.sqrt(np.maximum(-pm / 3.0, 0.0))
        r3 = r * r * r
        safe_r3 = np.where(r3 > 0.0, r3, 1.0)
        cos_theta = np.clip(
            np.where(r3 > 0.0, -qm / 2.0 / safe_r3, 1.0), -1.0, 1.0
        )
        t[triple] = 2.0 * r * np.cos(np.arccos(cos_theta) / 3.0)

    return (t - a / 3.0).reshape(out_shape)


@njit(cache=True, fastmath=True, parallel=True)
def _prox_j_kernel(V0, out, gamma, epsilon, obstacle):
    """prox_J を 1 パスで計算する融合カーネル(一時配列なし)。

    V0 / out は (npts, d) の 2 次元ビュー、obstacle は長さ npts
    (なければ長さ 0)。3 次方程式の分岐はスカラー if で片側だけ
    実行される。数式は numpy 版(_prox_j_numpy)と同一。
    """
    npts = V0.shape[0]
    nm = V0.shape[1] - 1
    has_obs = obstacle.size == npts
    g2 = gamma * gamma
    for i in prange(npts):
        if has_obs and obstacle[i] > 0.0:
            f = epsilon
        else:
            f0 = V0[i, nm]
            msq = 0.0
            for j in range(nm):
                msq += V0[i, j] * V0[i, j]
            a = 4.0 * gamma - f0
            b = 4.0 * g2 - 4.0 * gamma * f0
            c = -gamma * msq - 4.0 * g2 * f0

            # 標準形 t³ + p t + q = 0 (f = t - a/3) の最大実根
            a2 = a * a
            p = b - a2 / 3.0
            q = 2.0 * (a2 * a) / 27.0 - a * b / 3.0 + c
            half_q = 0.5 * q
            third_p = p / 3.0
            delta = half_q * half_q + third_p * third_p * third_p
            if delta > 0.0:
                # Cardano:-q/2 と同符号側の被開立数を選び相殺誤差を回避
                u = np.cbrt(-half_q + np.copysign(np.sqrt(delta), -q))
                t = u - third_p / u
            else:
                # 3 実根(三角関数解)。最大根は 2r·cos(θ/3)
                r = np.sqrt(max(-third_p, 0.0))
                r3 = r * r * r
                if r3 > 0.0:
                    cos_theta = min(max(-half_q / r3, -1.0), 1.0)
                else:
                    cos_theta = 1.0
                t = 2.0 * r * np.cos(np.arccos(cos_theta) / 3.0)
            f = max(t - a / 3.0, epsilon)

        scale = f / (f + 2.0 * gamma)
        for j in range(nm):
            out[i, j] = V0[i, j] * scale
        out[i, nm] = f


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
    V0 = np.ascontiguousarray(V0, dtype=float)
    flat = V0.reshape(-1, V0.shape[-1])
    if obstacle is None:
        obs = np.empty(0)
    else:
        obs = np.ascontiguousarray(
            np.broadcast_to(np.asarray(obstacle, dtype=float), V0.shape[:-1])
        ).reshape(-1)
    out = np.empty_like(flat)
    _prox_j_kernel(flat, out, float(gamma), float(epsilon), obs)
    return out.reshape(V0.shape)


def _prox_j_numpy(
    V0: np.ndarray,
    gamma: float,
    epsilon: float,
    obstacle: np.ndarray | None = None,
) -> np.ndarray:
    """prox_j の numpy 参照実装(融合カーネルとの数値同等性テスト用)。"""
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

    # m0 / (1 + 2γ/f) と等価だが f = 0(obstacle ケースの ε = 0)でも
    # ゼロ除算にならない形
    m = m0 * (f / (f + 2.0 * gamma))[..., None]
    return np.concatenate([m, f[..., None]], axis=-1)
