# MATLAB → Python 移植計画

## 1. ゴールとスコープ

**ゴール**: `img/evolution.png` に相当する図を Python で再現して出力する。

`img/evolution.png` は 'obstacle' テストケース(`code/Labyrinthe.png` の迷路を障害物とし、左上のガウシアンを右下へ輸送する)の計算結果を、時刻 t = 0, 1/9, …, 1 の 10 コマの等高線図として並べたモンタージュである。したがって移植の最小スコープは次のとおり。

- **アルゴリズム**: `test_bb_dr.m` の Douglas–Rachford(PPXA)ソルバのみを移植する。Primal–Dual 版(`test_bb_pd.m`)はスコープ外。
- **コスト関数**: `alpha = 1`(通常の L² Wasserstein)のみ対応する。`proxJ.m` の `0 < alpha < 1` の Newton 法分岐は移植しない。
- **次元**: 空間 2 次元 + 時間 1 次元(スタガード格子としては 3 次元)のみ。MATLAB 版にある 2D/4D 分岐は移植しない。
- **問題サイズ**: `Labyrinthe.png` に合わせて N = P = 50、Q = 2N = 100(`test_bb_dr.m` の 'obstacle' ケースと同じ)。

**スコープ外**: PD ソルバ、`alpha ≠ 1`、4 次元(リーマン多様体上の OT)、`perform_sdmm.m` / `perform_dr_spingarn.m` などの代替ソルバ、MATLAB の対話的表示。

## 2. 技術スタックとライブラリ管理(uv)

Python 3.12 以上を想定し、依存管理はすべて uv で行う。

```bash
uv init --package ot-splitting   # pyproject.toml の生成
uv add numpy scipy matplotlib pillow
uv add --dev pytest
```

| ライブラリ | 用途 |
|---|---|
| numpy | 配列演算全般(スタガード格子、prox 計算) |
| scipy | `scipy.fft.dctn` / `idctn`(Poisson ソルバ)、`scipy.linalg.pinv`(補間制約への射影行列) |
| matplotlib | 等高線モンタージュの描画・PNG 出力 |
| pillow | `Labyrinthe.png` の読み込み(matplotlib の `imread` でも代用可) |
| pytest(dev) | 各モジュールの単体テスト |

実行はすべて `uv run` 経由(例: `uv run python -m ot_splitting.run_obstacle`)。

## 3. プロジェクト構成

リポジトリ直下に uv プロジェクトを置く(ステップ 5-1 で初期化済み)。

```
.                        # リポジトリルート = uv プロジェクトルート
├── pyproject.toml
├── src/ot_splitting/
│   ├── __init__.py
│   ├── grid.py          # Staggered クラス(格子データ + 加減算・スカラー倍)
│   ├── operators.py     # div, interp
│   ├── poisson.py       # DCT による Neumann 境界 Poisson ソルバ
│   ├── projections.py   # div_proj(発散ゼロ射影)、interp_proj(補間整合射影)
│   ├── prox.py          # proxJ(alpha=1 の 3 次方程式の閉形式解)
│   ├── solver.py        # PPXA(並列 Douglas–Rachford)反復
│   ├── problems.py      # テストケース定義(f0, f1, obstacle の生成)
│   ├── viz.py           # 等高線モンタージュの描画(evolution.png の再現)
│   └── run_obstacle.py  # エントリポイント:計算 → out/evolution.png 出力
└── tests/
    ├── test_operators.py
    ├── test_poisson.py
    ├── test_projections.py
    └── test_prox.py
```

## 4. 移植対象の対応表

| MATLAB | Python | 内容・注意点 |
|---|---|---|
| `@staggered/staggered.m` | `grid.py: Staggered` | 各軸方向に 1 点多い配列 `M[k]`(形状 `(N+1,P,Q)`, `(N,P+1,Q)`, `(N,P,Q+1)`)を持つデータクラス。`+`, `-`, スカラー倍のみ実装すれば十分(境界条件フィールドは DR 経路では未使用) |
| `@staggered/div.m` | `operators.py: div` | 各軸の `np.diff` × 格子数の和 |
| `@staggered/interp.m` | `operators.py: interp` | スタガード格子 → 中心格子への 2 点平均。出力形状 `(N,P,Q,3)` |
| `@staggered/div_proj.m` | `projections.py: div_proj` | `div=0` への射影。Poisson 方程式を解いて勾配を引く。**内部境界(配列の 2..end-1)のみ更新**する点に注意 |
| `poisson3d_Neumann.m` | `poisson.py` | `scipy.fft.dctn(f, type=2, norm='ortho')` → 固有値 `2cos(πk/N)−2` で除算 → `idctn`。MATLAB の `mirt_dctn` と正規化規約が一致するか要検証(§6) |
| `@staggered/interp_proj.m` | `projections.py: interp_proj` | 制約 `{v = S u, u(1)=a, u(end)=b}` へのアフィン射影。軸ごとに小行列 `pinv(A)` と `B = I − pA·A` を構築。MATLAB の `persistent` は Python ではモジュールレベルの `functools.lru_cache` 等で代替 |
| `proxJ.m`(alpha=1 分岐) + `poly_root_new.m` | `prox.py: prox_j` | 3 次方程式 `f³ + (4γ−f₀)f² + (4γ²−4γf₀)f − γ|m₀|² − 4γ²f₀ = 0` の正実根を全格子点で一括計算。250 万点/反復になるためループ不可 → **Cardano の公式でベクトル化**(`poly_root_new.m` が閉形式実装の参考)。その後 `f ← max(f, ε)`、障害物内は `f ← ε`、`m ← m₀ / (1 + 2γ/f)` |
| `test_bb_dr.m`(反復部) | `solver.py: ppxa` | 3 つの prox(J、div=0、補間整合)を並列適用する PPXA 反復。`mu = 1.98`, `gamma = 1/230`, `omega = 1/3` |
| `test_bb_dr.m`('obstacle' ケース) | `problems.py` | 迷路画像から `obstacle` マスク生成、ガウシアン `f0`, `f1` を配置・正規化、線形補間で初期化 |
| `imageplot.m` + 図の体裁 | `viz.py` | matplotlib の `contour`(jet カラーマップ)で 2 行 × 5 列のモンタージュ。障害物は黒で重ね描き、各コマに `t = k/9` のラベル |

移植不要: `perform_primal_dual.m`, `pd_*.m`, `compute_dual_prox.m`, `interp_adj.m`(PD 専用)、`perform_cg.m`, `perform_sdmm.m`, `perform_ppxa.m`(`test_bb_dr.m` は自前ループを持つため)、`mirt_dctn.m` / `mirt_idctn.m`(scipy で代替)、`getoptions.m`, `progressbar.m` ほか表示系ユーティリティ。

## 5. 実装ステップ

依存の少ない順に下から積み上げ、各ステップで単体テストを書いてから次に進む。
(全ステップ実装済み・2026-07-06 完了)

1. **プロジェクト初期化** — `uv init` でリポジトリルートに雛形を作り、依存を追加。
2. **Staggered 格子と基本作用素**(`grid.py`, `operators.py`) — `Staggered` クラス、`div`, `interp` を実装。
   - テスト: 定数場で `div = 0`、線形場で厳密値と一致すること。
3. **Poisson ソルバ**(`poisson.py`) — DCT ベースの Neumann Poisson。
   - テスト: 既知の固有関数 `cos(πkx/N)` を入力し解析解と比較。`poisson(div(u))` の往復整合。
4. **発散ゼロ射影**(`projections.py: div_proj`) — テスト: 射影後 `‖div(u)‖ < 1e-10`、冪等性(2 回射影しても不変)、境界成分(初期・最終密度 `f0`, `f1` を保持する `M[2]` の両端スライス)が変化しないこと。
5. **補間整合射影**(`projections.py: interp_proj`) — 射影行列を軸ごとにキャッシュ。
   - テスト: 射影後 `v = interp(u)` が成立、冪等性、`(u,v)` が既に制約を満たすとき不変。
6. **prox_J**(`prox.py`) — Cardano によるベクトル化 3 次方程式ソルバと prox 本体。
   - テスト: `np.roots` による逐点計算(小配列)と一致すること。`γ → 0` で恒等写像に近づくこと。prox の定義(最小化問題)を数値微分で検証。
7. **PPXA ソルバ**(`solver.py`) — `test_bb_dr.m` の反復ループを忠実に移植。エネルギー `J`、制約違反 `‖div‖` の履歴を記録。
   - テスト(結合): まず 'gaussian' ケース(N=P=Q=16 程度)で実行し、エネルギーが減少傾向・制約違反が 0 に収束・密度が非負・総質量が保存されることを確認。対称な 2 つのガウシアン間の輸送で、中間時刻の密度が輸送経路上を移動すること。
8. **'obstacle' 問題設定**(`problems.py`) — `Labyrinthe.png` 読み込み(赤チャネル > 0 が通路)、マスク生成、`f0`/`f1` の配置(σ=0.04、位置 (.08,.08) と (.92,.92))、障害物内の密度ゼロ化と正規化、`ε = min(f0)` の設定。
9. **可視化**(`viz.py`, `run_obstacle.py`) — `Q+1` 枚のスナップショットから `t = 0, 1/9, …, 1` に対応する 10 枚を等間隔抽出し、jet カラーマップの `contour` + 黒い迷路壁を 2×5 に配置して `out/evolution.png` へ保存。
10. **本番実行と調整** — N=P=50, Q=100, `niter = 2000`(旧 README の推奨値)で実行し、目視で `img/evolution.png` と比較。質量が迷路の通路に沿って移動していれば達成。収束が不十分なら `niter` / `gamma` を調整。

## 6. MATLAB → Python 移植時の注意点

- **インデックス**: MATLAB は 1 始まり・両端含む。`2:(end-1)` は Python では `[1:-1]`。`div_proj` の内部更新スライスで特に間違えやすい。
- **配列レイアウト**: MATLAB は column-major。numpy はデフォルト row-major だが、要素順に依存する処理(reshape/permute)は `interp_proj` の軸入れ替えに限られるため、`np.transpose` + `reshape(order='F')` に頼らず「軸 i を先頭に持ってくる `np.moveaxis` → 行列積 → 戻す」で書き直すのが安全。
- **DCT の正規化**: `mirt_dctn.m` の DCT-II 正規化と `scipy.fft.dctn(norm='ortho')` の規約差を必ず突き合わせる。Poisson ソルバは「DCT → 除算 → IDCT」の往復なので、直交正規化なら定数倍の差は打ち消されるが、固有値スケールの検証テスト(ステップ 3)で確認する。
- **3 次方程式の根の選択**: `poly_root_new` は正実根を返す。Cardano 実装では判別式の符号による分岐と、複数実根がある場合に最大の正根を選ぶことを明示的に扱う。`f0 ≥ 0, γ > 0` の下では正根は一意であることが論文で保証されている。
- **`persistent` 変数**: `interp_proj` の射影行列キャッシュは、格子サイズをキーにした辞書 or `lru_cache` で置き換える。
- **性能**: 1 反復あたり Poisson ソルバ(DCT 3 回)+ 3 次方程式 25 万点 + 小行列積。全体をベクトル化すれば numpy で 2000 反復が数分オーダーに収まる見込み。遅い場合は `scipy.fft` の `workers` 指定や float32 化を検討(まずは float64 で正しさ優先)。
- **参照値の取得(任意)**: 環境に Octave があれば MATLAB コードをそのまま実行して中間量(`proxJ` の入出力など)をダンプし、Python 実装との数値一致テストに使える。なければ §5 の性質ベーステストのみで進める。

## 7. 完了条件

1. `uv run python -m ot_splitting.run_obstacle` が単体で完走し、`out/evolution.png` を生成する。
2. 生成画像が `img/evolution.png` と定性的に一致する:質量が迷路の通路のみを通って左上 → 右下へ移動し、10 コマのモンタージュとして描画されている。
3. 収束診断で `‖div‖` が初期値から十分減少(目安: 1e-4 以下)している。
4. `uv run pytest` が全テスト通過。
