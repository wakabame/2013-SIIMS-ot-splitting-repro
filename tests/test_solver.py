"""PPXA ソルバの結合テスト(実装ステップ 5-7)。

test_bb_dr.m の 'gaussian' ケースを 16³ に縮小して実行し、
収束・保存則・対称性を確認する。
"""

import numpy as np
import pytest

from ot_splitting.grid import Staggered
from ot_splitting.operators import interp
from ot_splitting.solver import PPXAResult, bb_energy, linear_init, solve_ppxa

N = P = Q = 16
NITER = 300


def gaussian_pair():
    x, y = np.meshgrid(np.linspace(0, 1, P), np.linspace(0, 1, N))
    gauss = lambda a, b, s: np.exp(
        -((y - a) ** 2 + (x - b) ** 2) / (2 * s**2)
    )
    f0 = 1e-6 + gauss(0.2, 0.2, 0.1)
    f0 /= f0.sum()
    f1 = 1e-6 + gauss(0.8, 0.8, 0.1)
    f1 /= f1.sum()
    return f0, f1


@pytest.fixture(scope="module")
def gaussian_result() -> tuple[np.ndarray, np.ndarray, PPXAResult]:
    f0, f1 = gaussian_pair()
    res = solve_ppxa(f0, f1, Q, niter=NITER, epsilon=f0.min())
    return f0, f1, res


class TestHelpers:
    def test_linear_init(self):
        f0, f1 = gaussian_pair()
        F = linear_init(f0, f1, Q)
        assert F.shape == (N, P, Q + 1)
        np.testing.assert_array_equal(F[:, :, 0], f0)
        np.testing.assert_array_equal(F[:, :, -1], f1)
        np.testing.assert_allclose(F[:, :, Q // 2], (f0 + f1) / 2)

    def test_bb_energy(self):
        V = np.zeros((2, 2, 2, 3))
        V[..., 0] = 2.0  # |m|² = 4
        V[..., 2] = 0.5  # f
        assert np.isclose(bb_energy(V, 1e-8), 8 * 4 / 0.5)

    def test_bb_energy_clamps_small_density(self):
        V = np.zeros((1, 1, 1, 3))
        V[..., 0] = 1.0
        V[..., 2] = 0.0  # ゼロ密度は ε でクランプされ発散しない
        assert np.isfinite(bb_energy(V, 1e-6))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            solve_ppxa(np.zeros((4, 4)), np.zeros((4, 5)), 4)


class TestGaussianCase:
    def test_result_shapes(self, gaussian_result):
        _, _, res = gaussian_result
        assert isinstance(res.U, Staggered)
        assert res.U.dim == (N, P, Q)
        assert res.U.M[2].shape == (N, P, Q + 1)
        assert res.V.shape == (N, P, Q, 3)
        assert len(res.J) == len(res.constr) == NITER

    def test_energy_decreases(self, gaussian_result):
        _, _, res = gaussian_result
        # 線形補間初期値から大きく減少する(観測値: 55627 → 969)
        assert res.J[-1] < 0.05 * res.J[0]
        # 後半 100 反復の平均は前半 100 反復の平均より小さい
        assert res.J[-100:].mean() < res.J[:100].mean()

    def test_divergence_violation_decreases(self, gaussian_result):
        _, _, res = gaussian_result
        assert res.constr[-1] < 0.3 * res.constr[0]

    def test_endpoint_densities_preserved(self, gaussian_result):
        f0, f1, res = gaussian_result
        np.testing.assert_allclose(res.U.M[2][:, :, 0], f0, atol=1e-9)
        np.testing.assert_allclose(res.U.M[2][:, :, -1], f1, atol=1e-9)

    def test_mass_conserved_at_each_time(self, gaussian_result):
        _, _, res = gaussian_result
        mass = res.U.M[2].sum(axis=(0, 1))
        np.testing.assert_allclose(mass, 1.0, atol=0.01)

    def test_density_essentially_nonnegative(self, gaussian_result):
        _, _, res = gaussian_result
        assert res.min_density[-1] > -0.01

    def test_mass_moves_along_diagonal(self, gaussian_result):
        # 中間時刻の重心が輸送経路の中点 (0.5, 0.5) にあること
        _, _, res = gaussian_result
        x, y = np.meshgrid(np.linspace(0, 1, P), np.linspace(0, 1, N))
        ft = res.U.M[2][:, :, Q // 2]
        com = ((y * ft).sum() / ft.sum(), (x * ft).sum() / ft.sum())
        assert abs(com[0] - 0.5) < 0.05
        assert abs(com[1] - 0.5) < 0.05
        # 中間密度は端点の単なる平均(線形補間)ではなく移動している:
        # 線形補間なら重心は合うが、密度は 2 山に分かれる。
        # OT 解は 1 山なので、最大値が線形補間より大きい
        f0, f1, _ = gaussian_result
        assert ft.max() > 1.2 * ((f0 + f1) / 2).max()

    def test_time_reversal_symmetry(self, gaussian_result):
        # 対称な設定なので f(x, t) = f(1-x, 1-t) が(反復の対称性
        # 保存により)厳密に成り立つ
        _, _, res = gaussian_result
        F = res.U.M[2]
        for k in range(0, Q + 1, 4):
            np.testing.assert_allclose(
                F[:, :, k], F[::-1, ::-1, Q - k], atol=1e-8
            )

    def test_consistency_V_equals_interp_U(self, gaussian_result):
        _, _, res = gaussian_result
        np.testing.assert_allclose(res.V, interp(res.U), atol=1e-12)
