"""等高線モンタージュの描画(img/evolution.png の再現)。

MATLAB 版 ``animation_matlab.m`` の等高線表示(jet カラーマップ、
32 レベル)を踏襲し、時刻 t = 0 … 1 のスナップショットを格子状に
並べる。
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def select_frames(n_times: int, n_frames: int) -> np.ndarray:
    """0 … n_times-1 から等間隔に n_frames 個のインデックスを選ぶ。"""
    return np.round(np.linspace(0, n_times - 1, n_frames)).astype(int)


def frame_labels(n_frames: int) -> list[str]:
    """時刻ラベル ``t = k/(n_frames-1)`` (既約分数)のリスト。"""
    labels = []
    for k in range(n_frames):
        frac = Fraction(k, n_frames - 1)
        if frac.denominator == 1:
            labels.append(f"$t = {frac.numerator}$")
        else:
            labels.append(f"$t = {frac.numerator}/{frac.denominator}$")
    return labels


def plot_evolution(
    F: np.ndarray,
    wall: np.ndarray | None = None,
    n_frames: int = 10,
    ncols: int = 5,
    n_levels: int = 32,
    path: str | Path | None = None,
    dpi: int = 150,
):
    """密度スナップショット F (N, P, T) の等高線モンタージュを描く。

    wall は (N, P) の壁マスク(黒で重ね描きし、壁内の密度は無視)。
    path を与えると PNG として保存する。matplotlib の Figure を返す。
    """
    N, P, T = F.shape
    indices = select_frames(T, n_frames)
    labels = frame_labels(n_frames)
    nrows = ceil(n_frames / ncols)

    frames = []
    for i in indices:
        data = F[:, :, i]
        if wall is not None:
            data = np.where(wall, 0.0, data)
        frames.append(data)
    vmax = max(f.max() for f in frames)
    levels = np.linspace(vmax / n_levels, vmax, n_levels)

    if wall is not None:
        overlay = np.zeros((N, P, 4))
        overlay[wall] = (0.0, 0.0, 0.0, 1.0)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.3, nrows * 2.55)
    )
    for ax, data, label in zip(np.ravel(axes), frames, labels):
        # MATLAB の caxis と同様に上限を少し超えて設定し、
        # 最上位レベルが彩度の高い赤になるようにする
        ax.contour(
            data, levels=levels, cmap="jet",
            vmin=0.0, vmax=vmax * 1.08, linewidths=0.8,
        )
        if wall is not None:
            ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.invert_yaxis()  # 行 0 を上に(画像と同じ向き)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.7")
        ax.set_xlabel(label, fontsize=13)
    for ax in np.ravel(axes)[n_frames:]:
        ax.set_visible(False)

    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig
