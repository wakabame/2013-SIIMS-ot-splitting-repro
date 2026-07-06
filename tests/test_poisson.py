"""Neumann Poisson ソルバのテスト(実装ステップ 5-3)。"""

import numpy as np
import pytest

from ot_splitting.poisson import poisson_neumann


def neumann_laplacian(
    u: np.ndarray, lengths: tuple[float, ...] | None = None
) -> np.ndarray:
    """テスト用の離散 Neumann Laplacian(境界は反射 = 端点複製)。"""
    if lengths is None:
        lengths = (1.0,) * u.ndim
    lap = np.zeros_like(u)
    for axis, (n, length) in enumerate(zip(u.shape, lengths)):
        h = length / n
        padded = np.pad(
            u, [(1, 1) if k == axis else (0, 0) for k in range(u.ndim)],
            mode="edge",
        )
        lap += np.diff(padded, n=2, axis=axis) / h**2
    return lap


def dct_eigenvector(n: int, k: int) -> np.ndarray:
    """Neumann Laplacian の固有ベクトル cos(πk(j+1/2)/n)。"""
    return np.cos(np.pi * k * (np.arange(n) + 0.5) / n)


class TestEigenfunctions:
    @pytest.mark.parametrize("n,k", [(8, 1), (16, 3), (32, 7)])
    def test_1d_eigenfunction(self, n, k):
        # f が固有ベクトルなら解は u = -f / λ_k(λ_k = (2cos(πk/n)-2)/h²)
        h = 1.0 / n
        f = dct_eigenvector(n, k)
        lam = (2.0 * np.cos(np.pi * k / n) - 2.0) / h**2
        u = poisson_neumann(f)
        np.testing.assert_allclose(u, -f / lam, atol=1e-12)

    def test_2d_separable_eigenfunction(self):
        n, p = 16, 12
        kx, ky = 2, 3
        f = np.outer(dct_eigenvector(n, kx), dct_eigenvector(p, ky))
        lam = (2 * np.cos(np.pi * kx / n) - 2) * n**2 + (
            2 * np.cos(np.pi * ky / p) - 2
        ) * p**2
        u = poisson_neumann(f)
        np.testing.assert_allclose(u, -f / lam, atol=1e-12)


class TestResidual:
    """解に Laplacian を適用して -f が復元されること(往復整合)。"""

    @pytest.mark.parametrize("shape", [(16,), (12, 16), (8, 10, 12)])
    def test_laplacian_roundtrip_zero_mean(self, shape):
        rng = np.random.default_rng(42)
        f = rng.standard_normal(shape)
        f -= f.mean()  # 定数モードを除く(可解条件)
        u = poisson_neumann(f)
        np.testing.assert_allclose(
            neumann_laplacian(u), -f, atol=1e-9
        )

    def test_roundtrip_with_lengths(self):
        rng = np.random.default_rng(7)
        lengths = (2.0, 0.5)
        f = rng.standard_normal((12, 18))
        f -= f.mean()
        u = poisson_neumann(f, lengths=lengths)
        np.testing.assert_allclose(
            neumann_laplacian(u, lengths=lengths), -f, atol=1e-9
        )

    def test_bb_problem_size(self):
        # 本番と同じ 50x50x100 でも往復整合が成り立つこと
        rng = np.random.default_rng(0)
        f = rng.standard_normal((50, 50, 100))
        f -= f.mean()
        u = poisson_neumann(f)
        residual = neumann_laplacian(u) + f
        assert np.abs(residual).max() < 1e-6


class TestConventions:
    def test_constant_input_gives_constant_output(self):
        # 定数モードは denom=1 に置換され uhat0 = -fhat0 となる
        f = np.full((8, 8), 3.0)
        u = poisson_neumann(f)
        np.testing.assert_allclose(u, -f, atol=1e-12)

    def test_output_shape_and_dtype(self):
        f = np.zeros((5, 6, 7))
        u = poisson_neumann(f)
        assert u.shape == f.shape
        assert u.dtype == np.float64

    def test_solution_neumann_compatible(self):
        # 解が Neumann 境界条件(境界での法線微分ゼロ)と整合すること:
        # 反射 Laplacian の残差が境界セルでも小さいことで確認済みだが、
        # ここでは解のミラー拡張が滑らかであること(端の差分が小さい)
        # までは要求されないため、境界を含む残差のみを確認する。
        rng = np.random.default_rng(3)
        f = rng.standard_normal((10, 10))
        f -= f.mean()
        u = poisson_neumann(f)
        residual = neumann_laplacian(u) + f
        # 境界行・列を含めて残差ゼロ
        assert np.abs(residual[0, :]).max() < 1e-9
        assert np.abs(residual[-1, :]).max() < 1e-9
        assert np.abs(residual[:, 0]).max() < 1e-9
        assert np.abs(residual[:, -1]).max() < 1e-9
