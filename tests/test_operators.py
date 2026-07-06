"""Staggered クラスと div / interp 作用素のテスト(実装ステップ 5-2)。"""

import numpy as np
import pytest

from ot_splitting.grid import Staggered
from ot_splitting.operators import div, interp

DIMS = [(4, 5), (3, 4, 5), (6, 6, 6)]


def linear_staggered(dim: tuple[int, ...], slopes: tuple[float, ...]) -> Staggered:
    """第 k 成分が軸 k 方向の座標に比例する線形場を作る。

    成分 k の格子点 i は位置 x = i / dim[k] にあり、値は slopes[k] * x。
    """
    u = Staggered(dim)
    for k, n in enumerate(dim):
        pos = np.arange(n + 1) / n
        shape = [1] * len(dim)
        shape[k] = n + 1
        u.M[k] = np.broadcast_to(
            slopes[k] * pos.reshape(shape), u.component_shape(k)
        ).copy()
    return u


class TestStaggered:
    @pytest.mark.parametrize("dim", DIMS)
    def test_zero_init_shapes(self, dim):
        u = Staggered(dim)
        assert u.dim == dim
        for k in range(len(dim)):
            expected = tuple(
                n + 1 if i == k else n for i, n in enumerate(dim)
            )
            assert u.M[k].shape == expected
            assert np.all(u.M[k] == 0)

    def test_component_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            Staggered((3, 4), [np.zeros((3, 4)), np.zeros((3, 5))])

    def test_arithmetic(self):
        rng = np.random.default_rng(0)
        dim = (3, 4, 5)
        u = Staggered(dim, [rng.standard_normal(Staggered(dim).component_shape(k)) for k in range(3)])
        v = Staggered(dim, [rng.standard_normal(Staggered(dim).component_shape(k)) for k in range(3)])

        w = u + v
        for k in range(3):
            np.testing.assert_allclose(w.M[k], u.M[k] + v.M[k])

        w = u - v
        for k in range(3):
            np.testing.assert_allclose(w.M[k], u.M[k] - v.M[k])

        w = 2.5 * u
        for k in range(3):
            np.testing.assert_allclose(w.M[k], 2.5 * u.M[k])

        w = -u
        for k in range(3):
            np.testing.assert_allclose(w.M[k], -u.M[k])

    def test_grid_mismatch_rejected(self):
        with pytest.raises(ValueError):
            Staggered((3, 4)) + Staggered((4, 3))

    def test_copy_is_independent(self):
        u = Staggered((3, 4))
        v = u.copy()
        v.M[0][0, 0] = 1.0
        assert u.M[0][0, 0] == 0.0

    def test_norm(self):
        dim = (3, 4)
        u = Staggered(dim)
        u.M[0][:] = 3.0
        u.M[1][:] = 0.0
        expected = np.sqrt(9.0 * u.M[0].size)
        assert np.isclose(u.norm(), expected)


class TestDiv:
    @pytest.mark.parametrize("dim", DIMS)
    def test_constant_field_has_zero_div(self, dim):
        u = Staggered(dim)
        for k in range(len(dim)):
            u.M[k][:] = 7.0
        np.testing.assert_array_equal(div(u), np.zeros(dim))

    @pytest.mark.parametrize("dim", DIMS)
    def test_linear_field_exact(self, dim):
        # M[k] = a_k * x_k なら div = sum_k a_k(定数)になる
        slopes = tuple(float(2 * k + 1) for k in range(len(dim)))
        u = linear_staggered(dim, slopes)
        np.testing.assert_allclose(
            div(u), np.full(dim, sum(slopes)), rtol=1e-12
        )

    def test_lengths_scaling(self):
        # 領域長 lx を 2 倍にすると格子幅も 2 倍になり div は半分になる
        dim = (4, 6)
        u = linear_staggered(dim, (1.0, 3.0))
        v1 = div(u)
        v2 = div(u, lengths=(2.0, 2.0))
        np.testing.assert_allclose(v2, v1 / 2)

    @pytest.mark.parametrize("dim", DIMS)
    def test_output_shape(self, dim):
        assert div(Staggered(dim)).shape == dim


class TestInterp:
    @pytest.mark.parametrize("dim", DIMS)
    def test_output_shape(self, dim):
        assert interp(Staggered(dim)).shape == dim + (len(dim),)

    @pytest.mark.parametrize("dim", DIMS)
    def test_linear_field_midpoints(self, dim):
        # 線形場の 2 点平均はセル中心 (i + 0.5) / n での値に一致する
        slopes = tuple(float(k + 1) for k in range(len(dim)))
        u = linear_staggered(dim, slopes)
        v = interp(u)
        for k, n in enumerate(dim):
            centers = (np.arange(n) + 0.5) / n
            shape = [1] * len(dim)
            shape[k] = n
            expected = np.broadcast_to(
                slopes[k] * centers.reshape(shape), dim
            )
            np.testing.assert_allclose(v[..., k], expected, rtol=1e-12)

    def test_constant_field_preserved(self):
        dim = (3, 4, 5)
        u = Staggered(dim)
        for k in range(3):
            u.M[k][:] = float(k + 1)
        v = interp(u)
        for k in range(3):
            np.testing.assert_array_equal(v[..., k], np.full(dim, k + 1))
