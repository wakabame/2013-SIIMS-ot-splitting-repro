"""テストケース定義(f0, f1, obstacle の生成)。

MATLAB 版 ``test_bb_dr.m`` のデータ生成部('gaussian' と 'obstacle')
の移植。座標規約は MATLAB と同じ:第 1 軸(行)が x、第 2 軸(列)が
y で、いずれも [0, 1] に正規化。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAZE_PATH = _REPO_ROOT / "code" / "Labyrinthe.png"


@dataclass
class Problem:
    """ソルバに渡す問題設定一式。"""

    f0: np.ndarray  # 初期密度 (N, P)、総和 1
    f1: np.ndarray  # 最終密度 (N, P)、総和 1
    Q: int  # 時間離散化数
    epsilon: float  # 密度の下限(prox_J のクランプ値)
    obstacle: np.ndarray | None = None  # (N, P, Q)、正の点は通行不可


def gaussian_field(
    N: int, P: int, a: float, b: float, sigma: float
) -> np.ndarray:
    """中心 (a, b)(行方向, 列方向)の等方ガウシアン (N, P)。"""
    x = np.linspace(0.0, 1.0, N)[:, None]
    y = np.linspace(0.0, 1.0, P)[None, :]
    return np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2.0 * sigma**2))


def gaussian_problem(
    N: int = 32, P: int = 32, Q: int = 32,
    sigma: float = 0.1, rho: float = 1e-6,
) -> Problem:
    """'gaussian' ケース:(.2, .2) から (.8, .8) へのガウシアン輸送。"""
    f0 = rho + gaussian_field(N, P, 0.2, 0.2, sigma)
    f0 /= f0.sum()
    f1 = rho + gaussian_field(N, P, 0.8, 0.8, sigma)
    f1 /= f1.sum()
    return Problem(f0=f0, f1=f1, Q=Q, epsilon=float(f0.min()))


def obstacle_problem(
    maze_path: str | Path = DEFAULT_MAZE_PATH,
) -> Problem:
    """'obstacle' ケース:迷路内を (.08, .08) から (.92, .92) へ輸送。

    迷路画像の赤チャネルが 0(黒)の画素を壁とする。Q = 2N
    (MATLAB 版と同じ)。密度は壁の内側でゼロにしてから正規化する
    ため、epsilon = min(f0) = 0 になる(MATLAB 版と同じ挙動)。
    """
    im = np.asarray(Image.open(maze_path))
    N, P = im.shape[:2]
    Q = 2 * N
    wall = im[:, :, 0] == 0

    sigma = 0.04
    f0 = gaussian_field(N, P, 0.08, 0.08, sigma)
    f1 = gaussian_field(N, P, 0.92, 0.92, sigma)
    f0[wall] = 0.0
    f1[wall] = 0.0
    f0 /= f0.sum()
    f1 /= f1.sum()

    obstacle = np.broadcast_to(
        wall[:, :, None], (N, P, Q)
    ).astype(float)
    return Problem(
        f0=f0, f1=f1, Q=Q, epsilon=float(f0.min()), obstacle=obstacle
    )
