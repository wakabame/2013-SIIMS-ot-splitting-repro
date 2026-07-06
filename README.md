# Optimal Transport with Proximal Splitting

近接分離法(proximal splitting)を用いて、Benamou–Brenier による動的最適輸送(dynamic optimal transport)問題を数値的に解く MATLAB 実装です。以下の論文の図を再現するためのソースコードと論文原稿を含みます。

> N. Papadakis, G. Peyré, E. Oudet. [Optimal Transport with Proximal Splitting](https://hal.archives-ouvertes.fr/hal-00816211). *SIAM Journal on Imaging Sciences*, 7(1), pp. 212–238, 2014.

![最適輸送による密度補間の例](img/evolution.png)

## 概要

2 つの確率密度 `f0`(初期)と `f1`(最終)の間の L² Wasserstein 測地線(時間発展する密度の補間)を計算します。Benamou–Brenier 定式化では、最適輸送は次のエネルギー最小化問題として書けます。

```
min_{f,m}  ∫∫ |m(x,t)|² / f(x,t) dx dt
s.t.       ∂t f + div(m) = 0,   f(·,0) = f0,   f(·,1) = f1
```

本コードはこの問題を時空間のスタガード格子(staggered grid)上で離散化し、大規模凸最適化問題として以下の 2 種類の一階法で解きます。

- **Douglas–Rachford / PPXA**(`test_bb_dr.m`)— エネルギー項 J、発散ゼロ制約、スタガード格子と中心格子の補間整合制約の 3 つの近接作用素を並列に分離
- **Primal–Dual(Chambolle–Pock)**(`test_bb_pd.m`)— `min_U F(K(U)) + G(U)` の形に定式化し、前処理付き ADMM(primal-dual)で解く

さらに一般化コスト `∑_k f_k^(-α) |m_k|²`(`α ∈ [0,1]`)と、質量が通過できない**障害物(obstacle)**の指定に対応しています。`α = 1` が通常の L² Wasserstein 距離、`α = 0` が H⁻¹ ノルムに対応し、中間の値は両者の補間になります。

## ディレクトリ構成

```
.
├── code/                     MATLAB ソースコード
│   ├── test_bb_dr.m          Douglas–Rachford (PPXA) によるソルバのデモ
│   ├── test_bb_pd.m          Primal–Dual によるソルバのデモ
│   ├── animation_matlab.m    計算結果 U から等高線アニメーションを生成
│   ├── Labyrinthe.png        'obstacle' テストケース用の迷路画像
│   └── toolbox/              補助関数群
│       ├── @staggered/       スタガード格子クラス(div, interp, 射影などを実装)
│       ├── proxJ.m           BB エネルギー J の近接作用素(3 次多項式求解)
│       ├── perform_primal_dual.m  Chambolle–Pock 型 primal-dual 法
│       ├── perform_dr.m      Douglas–Rachford 法(2 項分離)
│       ├── perform_ppxa.m / perform_dr_spingarn.m / perform_sdmm.m  その他の分離法
│       ├── pd_operator.m     primal-dual 用の線形作用素 K / K*
│       ├── poisson*_Neumann.m  Neumann 境界条件下の Poisson 方程式ソルバ(DCT 使用、発散射影に利用)
│       ├── compute_dual_prox.m  Moreau 恒等式による双対 prox の計算
│       └── ...               表示・補間・ユーティリティ
├── paper/                    論文の LaTeX 原稿一式(ProxOT.pdf 含む)
└── img/                      README 用画像
```

## 必要環境

- MATLAB(追加ツールボックス不要。`toolbox/` に必要な関数がすべて同梱)
- Image Processing 系の機能は不要。'obstacle' ケースで `imread` を使用する程度

## 使い方

MATLAB で `code/` ディレクトリに移動して、いずれかのスクリプトを実行します。

```matlab
cd code
test_bb_dr   % Douglas–Rachford (PPXA) 版
test_bb_pd   % Primal–Dual 版
```

実行すると以下が表示されます。

- 時刻 t = 0 → 1 における密度 f(x,t) のスナップショット(20 コマ)
- エネルギー J の収束履歴と、発散ゼロ制約 `div = 0` の違反量の推移

計算後に `animation_matlab.m` を実行すると、結果 `U` から等高線アニメーション(GIF 用フレーム)を作成できます。

## 主なオプション

各テストスクリプト冒頭で編集します。

### 1. テストケースの選択

```matlab
test = 'gaussian';   % 'gaussian' | 'mixture' | 'obsession' | 'obstacle'
```

- `gaussian` — 単一ガウシアンの平行移動(基本ケース)
- `mixture` — 単一ガウシアン → 2 つのガウシアンの混合
- `obsession` — 分散の小さいガウシアン(より特異なケース)
- `obstacle` — `Labyrinthe.png` の迷路を障害物として質量を輸送

`f0`, `f1`(初期・最終密度)を自分で定義すれば独自のシナリオも作れます。

### 2. 離散化の次元

```matlab
N = 32; P = 32; Q = 32;
```

`N × P` が空間格子、`Q` が時間方向の離散化数です。

### 3. ソルバのパラメータ

Douglas–Rachford(`test_bb_dr.m`):

```matlab
mu    = 1.98;     % 緩和パラメータ、区間 ]0,2[ 内
gamma = 1/230;    % prox のステップ、> 0
niter = 1000;     % 反復回数
```

Primal–Dual(`test_bb_pd.m`):

```matlab
options.sigma = 85;
options.tau   = .99/(options.sigma*L);  % sigma*tau*||K||^2 < 1 を満たすこと
options.niter = 2000;   % 反復回数を増やすと精度が向上
```

### 4. 一般化コストと障害物

```matlab
alpha = 1;   % [0,1]。1: L2-Wasserstein、0: H^{-1}、中間値はその補間
```

障害物は 3 次元配列(空間 × 時間)で指定します。値が 1 の位置は質量が通過できません。

```matlab
obstacle = zeros(N,P,Q);
obstacle(N/2, P/2, :) = 1;   % 空間領域の中央に障害物を置く例
```

### 設定例

```matlab
test = 'gaussian';  N = 32; P = 32; Q = 32;  niter = 200;
```

```matlab
test = 'obstacle';  niter = 2000;
```

## 実装のポイント

- **スタガード格子**:密度と運動量を格子のずれた位置に配置する `@staggered` クラスを定義し、連続の式 `∂t f + div(m) = 0` を精度よく離散化しています。中心格子上の変数とは `interp` / `interp_adj` で相互変換します。
- **発散射影**:発散ゼロ制約への射影は Neumann 境界条件付き Poisson 方程式(`poisson3d_Neumann.m` など、DCT ベース)を解いて行います。
- **prox_J の計算**:BB エネルギーの近接作用素は各格子点で 3 次多項式の根を求める閉形式的な計算に帰着されます(`proxJ.m`, `poly_root.m`)。

## ライセンス・著作権

このリポジトリは由来の異なる 2 種類の成果物を含みます。

- **MATLAB ソースコード(`code/`)・論文原稿(`paper/`)・図(`img/`)** — 原著者による既存の成果物です。
  Copyright (c) 2013 Nicolas Papadakis, Gabriel Peyré, Édouard Oudet
- **Python 移植(`src/`, `tests/` および関連プロジェクトファイル)** — wakabame による再現実装です。
  Copyright (c) 2026 wakabame([MIT License](LICENSE))
