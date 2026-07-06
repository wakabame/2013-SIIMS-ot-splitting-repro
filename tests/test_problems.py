"""問題設定(problems.py)のテスト(実装ステップ 5-8)。"""

import warnings

import numpy as np
import pytest

from ot_splitting.problems import (
    DEFAULT_MAZE_PATH,
    gaussian_field,
    gaussian_problem,
    obstacle_problem,
)
from ot_splitting.prox import prox_j
from ot_splitting.solver import solve_ppxa


class TestGaussianField:
    def test_peak_location(self):
        g = gaussian_field(50, 50, 0.2, 0.6, 0.05)
        i, j = np.unravel_index(g.argmax(), g.shape)
        # 座標 (a, b) = (行/(N-1), 列/(P-1)) に最も近い格子点がピーク
        assert abs(i / 49 - 0.2) < 0.03
        assert abs(j / 49 - 0.6) < 0.03
        # 中心が格子点に乗るとは限らないので最大値は 1 弱
        assert 0.95 < g.max() <= 1.0

    def test_isotropic_symmetry(self):
        g = gaussian_field(41, 41, 0.5, 0.5, 0.1)
        np.testing.assert_allclose(g, g.T, atol=1e-12)
        np.testing.assert_allclose(g, g[::-1, ::-1], atol=1e-12)


class TestGaussianProblem:
    def test_basic_properties(self):
        prob = gaussian_problem()
        assert prob.f0.shape == prob.f1.shape == (32, 32)
        assert prob.Q == 32
        assert prob.obstacle is None
        np.testing.assert_allclose(prob.f0.sum(), 1.0)
        np.testing.assert_allclose(prob.f1.sum(), 1.0)
        assert prob.f0.min() > 0  # rho の床があるので正
        assert prob.epsilon == prob.f0.min()

    def test_symmetry_between_endpoints(self):
        # (.2,.2) と (.8,.8) は 180°回転で移り合う
        prob = gaussian_problem()
        np.testing.assert_allclose(
            prob.f0, prob.f1[::-1, ::-1], atol=1e-12
        )


@pytest.fixture(scope="module")
def prob():
    return obstacle_problem()


class TestObstacleProblem:

    def test_dimensions_from_maze(self, prob):
        # Labyrinthe.png は 50x50、Q = 2N = 100
        assert prob.f0.shape == (50, 50)
        assert prob.Q == 100
        assert prob.obstacle.shape == (50, 50, 100)

    def test_default_maze_path_exists(self):
        assert DEFAULT_MAZE_PATH.exists()

    def test_obstacle_mask(self, prob):
        wall = prob.obstacle[:, :, 0] > 0
        # 壁と通路の両方が存在する
        frac = wall.mean()
        assert 0.05 < frac < 0.6
        # 時間方向に一定
        assert np.all(prob.obstacle == prob.obstacle[:, :, :1])

    def test_densities_vanish_on_walls(self, prob):
        wall = prob.obstacle[:, :, 0] > 0
        assert np.all(prob.f0[wall] == 0.0)
        assert np.all(prob.f1[wall] == 0.0)
        np.testing.assert_allclose(prob.f0.sum(), 1.0)
        np.testing.assert_allclose(prob.f1.sum(), 1.0)

    def test_epsilon_is_zero(self, prob):
        # 壁上で密度ゼロのため epsilon = min(f0) = 0(MATLAB と同じ)
        assert prob.epsilon == 0.0

    def test_endpoint_positions(self, prob):
        # f0 のピークは左上 (.08, .08) 付近、f1 は右下 (.92, .92) 付近
        i0, j0 = np.unravel_index(prob.f0.argmax(), prob.f0.shape)
        i1, j1 = np.unravel_index(prob.f1.argmax(), prob.f1.shape)
        assert i0 / 49 < 0.2 and j0 / 49 < 0.2
        assert i1 / 49 > 0.8 and j1 / 49 > 0.8

    def test_start_and_goal_not_walled_in(self, prob):
        # 開始・終了位置の質量が壁に埋もれていない(有意な質量がある)
        assert prob.f0.max() > 0.01
        assert prob.f1.max() > 0.01


class TestEpsilonZeroRobustness:
    def test_prox_j_no_warnings_with_zero_epsilon(self):
        # ε = 0 かつ障害物あり(f = 0 になる)でも警告なく有限値を返す
        rng = np.random.default_rng(0)
        V0 = rng.standard_normal((8, 8, 8, 3)) * 0.1
        V0[..., 2] = rng.random((8, 8, 8))
        obstacle = np.zeros((8, 8, 8))
        obstacle[3:5, 3:5, :] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            V = prox_j(V0, 1.0 / 230.0, 0.0, obstacle=obstacle)
        assert np.all(np.isfinite(V))
        inside = obstacle > 0
        assert np.all(V[..., 2][inside] == 0.0)
        assert np.all(V[..., 0][inside] == 0.0)  # 運動量もゼロ

    def test_solver_smoke_on_obstacle_problem(self):
        # 本物の迷路設定でソルバが警告・NaN なしで数反復回ること
        prob = obstacle_problem()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = solve_ppxa(
                prob.f0, prob.f1, prob.Q,
                niter=3, epsilon=prob.epsilon, obstacle=prob.obstacle,
            )
        assert np.all(np.isfinite(res.U.M[2]))
        assert np.all(np.isfinite(res.J))
        # 境界(端点密度)は更新式の構造上、丸め誤差を除いて不変
        np.testing.assert_allclose(
            res.U.M[2][:, :, 0], prob.f0, atol=1e-14
        )
        np.testing.assert_allclose(
            res.U.M[2][:, :, -1], prob.f1, atol=1e-14
        )
