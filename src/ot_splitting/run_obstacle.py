"""エントリポイント:'obstacle' ケースを解いて evolution.png を出力する。

使い方::

    uv run python -m ot_splitting.run_obstacle [--niter 2000] [--output out/evolution.png]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt

from .problems import DEFAULT_MAZE_PATH, obstacle_problem
from .solver import solve_ppxa
from .viz import plot_evolution


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="迷路内の最適輸送を PPXA で解き、等高線モンタージュを出力する"
    )
    parser.add_argument("--niter", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=1.0 / 230.0)
    parser.add_argument("--mu", type=float, default=1.98)
    parser.add_argument(
        "--diag-every", type=int, default=25,
        help="収束診断(J, |div| の記録)の間隔。1 で毎反復(低速)",
    )
    parser.add_argument(
        "--maze", type=Path, default=DEFAULT_MAZE_PATH,
        help="迷路画像(赤チャネル 0 の画素を壁とみなす)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("out/evolution.png")
    )
    args = parser.parse_args(argv)

    prob = obstacle_problem(args.maze)
    N, P = prob.f0.shape
    print(f"problem: {N}x{P} spatial, Q={prob.Q}, niter={args.niter}")

    t0 = time.perf_counter()
    res = solve_ppxa(
        prob.f0, prob.f1, prob.Q,
        gamma=args.gamma, mu=args.mu, niter=args.niter,
        epsilon=prob.epsilon, obstacle=prob.obstacle,
        diag_every=args.diag_every, verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"done in {elapsed:.1f}s: J = {res.J[-1]:.6e}, "
        f"|div| = {res.constr[-1]:.3e} "
        f"(initial |div| = {res.constr[0]:.3e})"
    )

    wall = prob.obstacle[:, :, 0] > 0
    fig = plot_evolution(res.U.M[2], wall=wall, path=args.output)
    plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
