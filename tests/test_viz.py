"""可視化(viz.py)のテスト(実装ステップ 5-9)。"""

import matplotlib.pyplot as plt
import numpy as np

from ot_splitting.viz import frame_labels, plot_evolution, select_frames


class TestSelectFrames:
    def test_endpoints_included(self):
        idx = select_frames(101, 10)
        assert idx[0] == 0
        assert idx[-1] == 100
        assert len(idx) == 10
        assert np.all(np.diff(idx) > 0)

    def test_exact_when_counts_match(self):
        np.testing.assert_array_equal(
            select_frames(10, 10), np.arange(10)
        )


class TestFrameLabels:
    def test_ten_frames(self):
        labels = frame_labels(10)
        # 分数は既約(3/9 → 1/3 など)。目標画像と同じ表記
        assert labels[0] == "$t = 0$"
        assert labels[1] == "$t = 1/9$"
        assert labels[3] == "$t = 1/3$"
        assert labels[6] == "$t = 2/3$"
        assert labels[-1] == "$t = 1$"


class TestPlotEvolution:
    def make_moving_gaussian(self, N=20, T=11):
        x = np.linspace(0, 1, N)
        F = np.empty((N, N, T))
        for k, t in enumerate(np.linspace(0, 1, T)):
            c = 0.2 + 0.6 * t
            F[:, :, k] = np.exp(
                -((x[:, None] - c) ** 2 + (x[None, :] - c) ** 2) / 0.01
            )
        return F

    def test_saves_png(self, tmp_path):
        F = self.make_moving_gaussian()
        out = tmp_path / "sub" / "evolution.png"
        fig = plot_evolution(F, path=out)
        plt.close(fig)
        assert out.exists()
        assert out.stat().st_size > 10_000  # 中身のある PNG

    def test_with_wall_overlay(self, tmp_path):
        F = self.make_moving_gaussian()
        wall = np.zeros(F.shape[:2], dtype=bool)
        wall[8:12, :10] = True
        out = tmp_path / "walled.png"
        fig = plot_evolution(F, wall=wall, path=out)
        plt.close(fig)
        assert out.exists()

    def test_grid_layout(self):
        F = self.make_moving_gaussian()
        fig = plot_evolution(F, n_frames=10, ncols=5)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 10
        plt.close(fig)
