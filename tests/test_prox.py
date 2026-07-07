"""prox_J と 3 次方程式ソルバのテスト(実装ステップ 5-6)。"""

import numpy as np
import pytest
from scipy.optimize import minimize

from ot_splitting.prox import prox_j, solve_cubic_largest_root


def reference_largest_real_root(a, b, c):
    """np.roots による逐点参照実装(最大実根)。"""
    out = np.empty(np.shape(a))
    it = np.nditer(
        [np.asarray(a), np.asarray(b), np.asarray(c)],
        flags=["multi_index"],
    )
    for ai, bi, ci in it:
        roots = np.roots([1.0, float(ai), float(bi), float(ci)])
        real = roots[np.abs(roots.imag) < 1e-8].real
        out[it.multi_index] = real.max()
    return out


class TestCubicSolver:
    def test_matches_np_roots_random(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(200) * 3
        b = rng.standard_normal(200) * 3
        c = -np.abs(rng.standard_normal(200)) * 3  # prox_J では c <= 0
        x = solve_cubic_largest_root(a, b, c)
        expected = reference_largest_real_root(a, b, c)
        np.testing.assert_allclose(x, expected, rtol=1e-8, atol=1e-10)

    def test_residual_is_zero(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((10, 10))
        b = rng.standard_normal((10, 10))
        c = -np.abs(rng.standard_normal((10, 10)))
        x = solve_cubic_largest_root(a, b, c)
        residual = x**3 + a * x**2 + b * x + c
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)

    def test_known_roots(self):
        # (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6 → 最大根 3
        assert np.isclose(solve_cubic_largest_root(-6.0, 11.0, -6.0), 3.0)
        # (x-2)(x²+x+1) = x³ - x² - x - 2 → 実根は 2 のみ
        assert np.isclose(solve_cubic_largest_root(-1.0, -1.0, -2.0), 2.0)
        # 三重根 x³ = 0
        assert np.isclose(solve_cubic_largest_root(0.0, 0.0, 0.0), 0.0)

    def test_prox_cubic_has_unique_positive_root(self):
        # prox_J の係数(f0 ≥ 0, γ > 0)では正根がただ 1 つ
        rng = np.random.default_rng(2)
        gamma = 1.0 / 230.0
        f0 = rng.random(500)
        msq = rng.random(500)
        a = 4 * gamma - f0
        b = 4 * gamma**2 - 4 * gamma * f0
        c = -gamma * msq - 4 * gamma**2 * f0
        x = solve_cubic_largest_root(a, b, c)
        assert np.all(x > 0)
        for i in range(0, 500, 50):
            roots = np.roots([1.0, a[i], b[i], c[i]])
            pos = roots[(np.abs(roots.imag) < 1e-8) & (roots.real > 1e-12)]
            assert len(pos) == 1
            assert np.isclose(x[i], pos[0].real)


class TestProxJ:
    GAMMA = 1.0 / 230.0
    EPS = 1e-8

    def make_input(self, shape=(6, 7, 8), seed=0):
        rng = np.random.default_rng(seed)
        V = rng.standard_normal(shape + (3,)) * 0.1
        V[..., 2] = rng.random(shape) + 0.01  # 密度は正
        return V

    def test_output_shape_and_input_not_mutated(self):
        V0 = self.make_input()
        V0_bak = V0.copy()
        V = prox_j(V0, self.GAMMA, self.EPS)
        assert V.shape == V0.shape
        np.testing.assert_array_equal(V0, V0_bak)

    def test_cubic_stationarity(self):
        # 出力密度 f は(クランプされない限り)3 次方程式の根
        V0 = self.make_input(seed=1)
        V = prox_j(V0, self.GAMMA, self.EPS)
        f0 = V0[..., 2]
        f = V[..., 2]
        msq = np.sum(V0[..., :2] ** 2, axis=-1)
        g = self.GAMMA
        residual = (
            f**3
            + (4 * g - f0) * f**2
            + (4 * g**2 - 4 * g * f0) * f
            - g * msq
            - 4 * g**2 * f0
        )
        free = f > self.EPS
        assert free.any()
        np.testing.assert_allclose(residual[free], 0.0, atol=1e-10)

    def test_momentum_shrinkage_formula(self):
        V0 = self.make_input(seed=2)
        V = prox_j(V0, self.GAMMA, self.EPS)
        f = V[..., 2]
        expected_m = V0[..., :2] / (1 + 2 * self.GAMMA / f)[..., None]
        np.testing.assert_allclose(V[..., :2], expected_m, rtol=1e-12)

    def test_gamma_to_zero_is_identity(self):
        # γ → 0 で prox は恒等写像に近づく(正の密度に対して)
        V0 = self.make_input(seed=3)
        V = prox_j(V0, 1e-12, self.EPS)
        np.testing.assert_allclose(V, V0, atol=1e-8)

    @pytest.mark.parametrize("seed", [4, 5, 6])
    def test_matches_scipy_minimize_single_point(self, seed):
        # prox の定義:argmin_{m,f≥ε} 0.5(|m-m0|² + (f-f0)²) + γ|m|²/f
        rng = np.random.default_rng(seed)
        m0 = rng.standard_normal(2) * 0.2
        f0 = rng.random() + 0.05
        gamma = self.GAMMA

        def objective(z):
            m, f = z[:2], z[2]
            return 0.5 * (np.sum((m - m0) ** 2) + (f - f0) ** 2) + (
                gamma * np.sum(m**2) / f
            )

        res = minimize(
            objective,
            x0=[*m0, max(f0, 0.1)],
            bounds=[(None, None), (None, None), (self.EPS, None)],
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-12},
        )
        V0 = np.array([[*m0, f0]])
        V = prox_j(V0, gamma, self.EPS)
        np.testing.assert_allclose(V[0], res.x, rtol=1e-5, atol=1e-7)

    def test_obstacle_forces_epsilon_density(self):
        V0 = self.make_input(seed=7)
        obstacle = np.zeros(V0.shape[:-1])
        obstacle[2:4, 3:5, :] = 1.0
        V = prox_j(V0, self.GAMMA, self.EPS, obstacle=obstacle)
        inside = obstacle > 0
        np.testing.assert_array_equal(V[..., 2][inside], self.EPS)
        # 障害物外は影響を受けない
        V_free = prox_j(V0, self.GAMMA, self.EPS)
        np.testing.assert_array_equal(
            V[..., 2][~inside], V_free[..., 2][~inside]
        )
        # 運動量はクランプ後の f で縮小される(障害物内はほぼ 0)
        expected_m = V0[..., :2] / (1 + 2 * self.GAMMA / V[..., 2])[..., None]
        np.testing.assert_allclose(V[..., :2], expected_m, rtol=1e-12)

    def test_density_at_least_epsilon(self):
        # 密度が負・ゼロの入力でも出力は ε 以上
        V0 = self.make_input(seed=8)
        V0[..., 2] -= 0.5  # 一部を負にする
        V = prox_j(V0, self.GAMMA, self.EPS)
        assert V[..., 2].min() >= self.EPS

    def test_numba_kernel_matches_numpy_reference(self):
        # 高速化プラン 2-1:融合カーネルと numpy 実装の数値同等性
        # (fastmath 使用のため rtol=1e-8 を許容。§3 の検証方針)
        from ot_splitting.prox import _prox_j_numpy

        V0 = self.make_input(seed=10)
        V0[..., 2] -= 0.2  # 負の密度も混ぜる
        obstacle = np.zeros(V0.shape[:-1])
        obstacle[1:3, 2:5, :] = 1.0
        for obs in (None, obstacle):
            np.testing.assert_allclose(
                prox_j(V0, self.GAMMA, self.EPS, obstacle=obs),
                _prox_j_numpy(V0, self.GAMMA, self.EPS, obstacle=obs),
                rtol=1e-8,
                atol=1e-12,
            )

    def test_production_size_performance(self):
        # 本番サイズ (50,50,100) が 1 回あたり現実的な時間で動くこと
        import time

        V0 = self.make_input(shape=(50, 50, 100), seed=9)
        t0 = time.perf_counter()
        prox_j(V0, self.GAMMA, self.EPS)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"prox_j too slow: {elapsed:.2f}s"
