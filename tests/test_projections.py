"""発散ゼロ射影 div_proj のテスト(実装ステップ 5-4)。"""

import numpy as np
import pytest

from ot_splitting.grid import Staggered
from ot_splitting.operators import div
from ot_splitting.projections import div_proj

DIMS = [(8, 10), (6, 8, 10), (12, 12, 24)]


def random_staggered(dim: tuple[int, ...], seed: int = 0) -> Staggered:
    rng = np.random.default_rng(seed)
    u = Staggered(dim)
    for k in range(len(dim)):
        u.M[k] = rng.standard_normal(u.component_shape(k))
    return u


def zero_boundary(u: Staggered) -> Staggered:
    """各成分の軸方向両端スライスをゼロにした場を返す。"""
    v = u.copy()
    for k in range(len(v.dim)):
        first = [slice(None)] * len(v.dim)
        last = [slice(None)] * len(v.dim)
        first[k] = 0
        last[k] = -1
        v.M[k][tuple(first)] = 0.0
        v.M[k][tuple(last)] = 0.0
    return v


def dot(a: Staggered, b: Staggered) -> float:
    return sum(float(np.sum(x * y)) for x, y in zip(a.M, b.M))


class TestDivProj:
    @pytest.mark.parametrize("dim", DIMS)
    def test_divergence_is_zero(self, dim):
        # 発散の定数モードは境界成分(総フラックス)で決まり射影では
        # 変えられないため、両立条件を満たす入力(境界ゼロ)を使う
        u = zero_boundary(random_staggered(dim))
        v = div_proj(u)
        assert np.abs(div(v)).max() < 1e-9

    @pytest.mark.parametrize("dim", DIMS)
    def test_incompatible_field_leaves_constant_mode(self, dim):
        # 総発散が非ゼロの場では、射影後の発散は一様な定数
        # (= 元の発散の平均)として残る。MATLAB 版と同じ挙動。
        u = random_staggered(dim)
        v = div_proj(u)
        mean = div(u).mean()
        np.testing.assert_allclose(div(v), np.full(dim, mean), atol=1e-9)

    @pytest.mark.parametrize("dim", DIMS)
    def test_idempotent(self, dim):
        u = random_staggered(dim, seed=1)
        v1 = div_proj(u)
        v2 = div_proj(v1)
        for k in range(len(dim)):
            np.testing.assert_allclose(v2.M[k], v1.M[k], atol=1e-10)

    @pytest.mark.parametrize("dim", DIMS)
    def test_boundary_slices_preserved(self, dim):
        # 各成分の両端スライス(f0, f1 を含む)は射影で変化しない
        u = random_staggered(dim, seed=2)
        v = div_proj(u)
        for k in range(len(dim)):
            first = [slice(None)] * len(dim)
            last = [slice(None)] * len(dim)
            first[k] = 0
            last[k] = -1
            np.testing.assert_array_equal(
                v.M[k][tuple(first)], u.M[k][tuple(first)]
            )
            np.testing.assert_array_equal(
                v.M[k][tuple(last)], u.M[k][tuple(last)]
            )

    def test_constant_field_unchanged(self):
        # 定数場は発散ゼロなので不動点になる
        dim = (6, 8, 10)
        u = Staggered(dim)
        for k in range(3):
            u.M[k][:] = float(k + 1)
        v = div_proj(u)
        for k in range(3):
            np.testing.assert_allclose(v.M[k], u.M[k], atol=1e-12)

    @pytest.mark.parametrize("dim", DIMS)
    def test_orthogonality(self, dim):
        # 残差 u - P(u) は、境界成分が一致する発散ゼロ場の空間に直交する。
        # 境界ゼロかつ発散ゼロの任意の方向 w に対して <u - P(u), w> = 0。
        u = random_staggered(dim, seed=3)
        w = div_proj(zero_boundary(random_staggered(dim, seed=4)))
        w = zero_boundary(w)  # div_proj は境界を保つが、明示的に保証する
        assert np.abs(div(w)).max() < 1e-9
        residual = u - div_proj(u)
        assert abs(dot(residual, w)) < 1e-8 * (u.norm() * w.norm())

    def test_with_lengths(self):
        # 非等方な領域長でも発散(同じ lengths で評価)がゼロになる
        dim = (8, 12)
        lengths = (2.0, 0.5)
        u = zero_boundary(random_staggered(dim, seed=5))
        v = div_proj(u, lengths=lengths)
        assert np.abs(div(v, lengths=lengths)).max() < 1e-9

    def test_bb_setup_preserves_endpoint_densities(self):
        # test_bb_dr.m と同じ初期化:M[2] の両端に f0, f1 を置いて射影しても
        # 両端(初期・最終密度)が保存されること
        N, P, Q = 8, 8, 8
        rng = np.random.default_rng(6)
        f0 = rng.random((N, P))
        f0 /= f0.sum()
        f1 = rng.random((N, P))
        f1 /= f1.sum()
        t = np.linspace(0.0, 1.0, Q + 1).reshape(1, 1, Q + 1)
        u = Staggered((N, P, Q))
        u.M[2] = (1 - t) * f0[:, :, None] + t * f1[:, :, None]

        v = div_proj(u)
        np.testing.assert_array_equal(v.M[2][:, :, 0], f0)
        np.testing.assert_array_equal(v.M[2][:, :, -1], f1)
        assert np.abs(div(v)).max() < 1e-9
