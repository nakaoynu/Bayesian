# 引き継ぎドキュメント: `test3.py`

## 1. このファイルが何をするのか

`test3.py` は、GGG の THz 透過スペクトルをベイズ推定で解析するスクリプトです。  
目的は、実験データに対して物理モデルを当てはめ、以下を同時に行うことです。

- 物理パラメータ `g`, `a`, `B4`, `B6`, `eps_bg`, `gamma_1 ... gamma_7` を推定する
- H-form と B-form の 2 つのモデルを比較する
- 推定結果から事後予測スペクトルや感受率などを可視化する

このスクリプトの特徴は、**ポラリトン領域のスペクトル形状を最優先で合わせる**ことです。  
共振器領域は「スペクトル全体」を直接合わせるのではなく、**ピーク位置**と **FWHM** だけを軽く制約として使います。

---

## 2. 後輩向けの最短理解

まずは次の 4 点だけ掴めば十分です。

1. 実験データを読み込む
2. 物理モデルから透過スペクトルを計算する
3. 実験とモデルのズレを尤度として定義する
4. PyMC の SMC サンプラーでパラメータを推定する

コード全体は長いですが、流れはこの 4 ステップです。

---

## 3. 解析の全体フロー

`main()` の処理を文章にすると、次の順です。

1. 実行設定を読む
2. 参照用の v8 パラメータ CSV を読む
3. Excel から 10 個の実験データセットを読む
4. 各データセットを正規化し、ピーク位置と FWHM を抽出する
5. H-form のベイズモデルを作って SMC でサンプリングする
6. B-form のベイズモデルを作って SMC でサンプリングする
7. 2 モデルの ESS とベイズファクターを比較する
8. 事後分布、事後予測スペクトル、エネルギー準位、感受率を保存する
9. 推定結果を CSV / NetCDF / JSON で保存する

図にすると次の流れです。

```text
Excel実験データ
  ↓
前処理（正規化・ピーク検出）
  ↓
物理モデルで透過率を計算
  ↓
尤度を作る
  ↓
PyMC + SMC で事後分布を推定
  ↓
可視化とモデル比較
```

---

## 4. 入力ファイルと出力ファイル

### 入力

`test3.py` は次のファイルやディレクトリを前提にしています。

- `bayesian_inputs/BayesianInput_Raw_Transmittance_Temperature.xlsx`
- `bayesian_inputs/BayesianInput_Raw_Transmittance_Field.xlsx`
- `wls_v8_results_20260306_160250/parameters_H.csv`
- `wls_v8_results_20260306_160250/parameters_B.csv`

実験データの対象は `TARGET_DATA` に固定で書かれています。

- 温度依存 4 本: `4K`, `10K`, `20K`, `30K`
- 磁場依存 6 本: `4.2T`, `5T`, `6T`, `7T`, `8T`, `9T`

### 出力

実行すると、`march/` の下に次の形式のディレクトリが作られます。

```text
bayesian_v10_results_YYYYMMDD_HHMMSS/
```

その中に主に次のファイルが保存されます。

- `trace_H.nc`, `trace_B.nc`
- `summary_H.csv`, `summary_B.csv`
- `parameters_H.csv`, `parameters_B.csv`
- `posterior_predictive_spectra_H.png`, `posterior_predictive_spectra_B.png`
- `posterior_predictive_spectra_HB.png`
- `posterior_distributions_H.png`, `posterior_distributions_B.png`
- `energy_levels_H.png`, `energy_levels_B.png`
- `susceptibility_real_H.png`, `susceptibility_imag_H.png`
- `susceptibility_real_B.png`, `susceptibility_imag_B.png`
- `model_evaluation.json`

---

## 5. このスクリプトで推定しているパラメータ

### `g`

Gd イオンの g 因子です。  
だいたい 2 付近の値を取ります。

### `a`

感受率の強さを決めるスケール係数です。  
実験の吸収の強さを理論モデルに合わせる役目です。

### `B4`, `B6`

結晶場ハミルトニアンの係数です。  
エネルギー準位の並び方を決めます。

### `eps_bg`

背景誘電率です。  
Fabry-Perot 共振の位置に強く効きます。

### `gamma_1 ... gamma_7`

緩和係数です。  
スペクトルの線幅に効きます。

---

## 6. H-form と B-form の違い

このコードでは、磁気感受率 `chi` を使って比透磁率 `mu_r` を 2 通りで作ります。

### H-form

```python
mu_r = 1.0 + chi
```

### B-form

```python
mu_r = 1.0 / (1.0 - chi)
```

どちらの式が実験データをより良く説明するかを、同じ流れで別々に推定し、最後に比較します。

---

## 7. コードの読み方

長いファイルですが、次の順で読むと理解しやすいです。

### 7.1 設定値

ファイル先頭には解析条件があります。

- `SMC_DRAWS`
- `SMC_CHAINS`
- `NU_STUDENTT`
- `SIGMA_FWHM`
- `SIGMA_PEAK_FREQ`
- `SIGMA_SPECTRUM_BASE`
- `DEBUG_MODE`
- `FIX_B6_ZERO`

ここは「解析の実験条件を決める場所」です。

### 7.2 物理モデルを計算する関数

まず押さえるべき関数はこの 4 つです。

- `get_hamiltonian()`
- `calculate_susceptibility()`
- `calculate_transmission()`
- `calculate_transmission_for_params()`

役割は次の通りです。

```text
パラメータ
  ↓
ハミルトニアン
  ↓
磁気感受率 chi
  ↓
比透磁率 mu_r
  ↓
透過スペクトル
```

### 7.3 実験データの前処理

前処理で重要なのは次の関数です。

- `normalize_obs_spectrum()`
- `compute_fwhm_from_spectrum()`
- `compute_cavity_peak_info()`
- `detect_peaks_and_classify()`
- `create_weight_array()`
- `load_all_datasets()`

ここでやっていることは次の 3 つです。

- スペクトルを正規化する
- ポラリトン領域と共振器領域を分ける
- 共振器ピークの位置と線幅を取り出す

### 7.4 PyMC 用の出力を作るクラス

`MixedOutputModelOp` はこのファイルの中核です。

このクラスは、PyMC からパラメータを受け取って次の 3 種類を返します。

- 予測透過スペクトル
- 予測 FWHM
- 予測ピーク周波数

つまり、**「パラメータ → 観測と比較できる量」** に変換する橋渡し役です。

### 7.5 ベイズモデル本体

`build_pymc_model()` で事前分布と尤度を定義しています。

ここで重要なのは、尤度が 3 層構造になっている点です。

- ポラリトン領域: スペクトル形状を StudentT で比較
- 共振器領域: ピーク位置を StudentT で比較
- 共振器領域: FWHM を StudentT で比較

背景領域も弱い重みで比較しますが、主役はポラリトン領域です。

### 7.6 実行の入口

最後の `main()` が全体の司令塔です。  
困ったらまず `main()` を読めば、処理順が分かります。

---

## 8. 尤度の考え方

このコードの一番大事な設計思想は、**全部を同じ強さで合わせない**ことです。

### ポラリトン領域

ここは研究上もっとも重要なので、スペクトル形状そのものを強く合わせます。

### 共振器領域

ここはスペクトル全体を直接合わせず、次の 2 つだけを使います。

- ピーク位置
- FWHM

こうしている理由は、共振器領域全体を強く当てに行くと、過去版では `a` が不自然に大きくなり、ポラリトン領域の再現が悪化したためです。

### 背景領域

ここは補助的な情報として、ごく弱く使います。

---

## 9. 重要な関数のざっくり説明

### `resolve_sampling_resources()`

PC が過負荷にならないように、チェーン数とコア数を自動調整します。

### `get_hamiltonian()`

`B`, `g`, `B4`, `B6` から 8x8 ハミルトニアンを作ります。

### `calculate_susceptibility()`

ハミルトニアンを対角化し、遷移強度と緩和を考慮して感受率を計算します。

### `calculate_transmission()`

感受率から比透磁率を作り、Fabry-Perot 透過率を計算します。

### `load_all_datasets()`

Excel からデータを読み、正規化、ピーク検出、重み付けまでまとめて行います。

### `MixedOutputModelOp.perform()`

PyMC の中で何度も呼ばれる重い計算部分です。  
速度や挙動の確認をするときは、まずここを疑います。

### `build_pymc_model()`

事前分布と尤度の定義をまとめた関数です。  
解析方針を変更したいときは、ここを一番よく触ります。

### `main()`

実行順序と保存処理をまとめています。

---

## 10. 実行方法

このスクリプトは `march/` 配下で実行する想定です。

```bash
python test3.py
```

デバッグ時は、ファイル先頭の

```python
DEBUG_MODE = True
```

にすると、データセット数とサンプル数を減らして試せます。

また、CPU 負荷は環境変数でも調整できます。

```bash
export BAYES_BLAS_THREADS=1
export BAYES_MAX_CHAINS=8
export BAYES_MIN_CHAINS=2
python test3.py
```

---

## 11. よくある詰まりポイント

### 11.1 Excel が見つからない

`bayesian_inputs` の置き場所が違うと読み込みに失敗します。  
スクリプトは次の 2 箇所を探します。

- `march/bayesian_inputs`
- `Bayesian/bayesian_inputs`

### 11.2 v8 の参照 CSV が見つからない

`load_v8_optimized_params()` は固定ディレクトリ

```text
wls_v8_results_20260306_160250
```

を見に行きます。名前を変えると失敗します。

### 11.3 実行が重い

PyMC の SMC はかなり重いです。  
まずは以下で軽くしてください。

- `DEBUG_MODE = True`
- `SMC_DRAWS` を減らす
- `SMC_CHAINS` を減らす

### 11.4 どこを直せばよいか分からない

目的別に見る場所は次です。

- 物理式を変えたい: `get_hamiltonian`, `calculate_susceptibility`, `calculate_transmission`
- データ処理を変えたい: `load_all_datasets` 周辺
- 尤度を変えたい: `build_pymc_model`
- 保存物を変えたい: `main`

---

## 12. このコードを改良するときの優先順位

後輩が最初に安全に触るなら、この順がよいです。

1. `main()` のログ文や保存ファイル名を整理する
2. `DEBUG_MODE` で動かして処理の流れを確認する
3. `load_all_datasets()` の出力内容を print で追う
4. `build_pymc_model()` の尤度構造を理解する
5. 最後に `MixedOutputModelOp.perform()` の内部を触る

`MixedOutputModelOp.perform()` は頻繁に呼ばれるので、軽い変更でも解析結果に大きく影響します。

---

## 13. 引き継ぎ時に必ず伝えるべきこと

### 研究上の主張

この版は「ポラリトン領域を最優先に合わせる」ための実装です。  
共振器領域全体を強く当てる版ではありません。

### 実装上の主張

共振器は次の 2 指標だけで拘束しています。

- ピーク周波数
- FWHM

### 運用上の主張

このコードは、1 本の関数で全部やるというより、**前処理・物理計算・PyMC・可視化**を 1 ファイルにまとめた形です。  
将来的には分割した方が保守しやすいですが、現状は 1 ファイル完結型です。

---

## 14. 引き継ぎメモ

後輩が読む順番のおすすめは次です。

1. この MD を読む
2. `main()` を読む
3. `build_pymc_model()` を読む
4. `MixedOutputModelOp` を読む
5. `load_all_datasets()` を読む
6. 余裕が出たら物理関数を読む

最初から数式を全部理解する必要はありません。  
先に「この関数は何の役割か」を掴んでから、中身に入る方が理解しやすいです。

---

## 15. 今後の改善候補

このファイルは動作する一方で、保守性の面では改善余地があります。

- 物理計算部を別ファイルに分離する
- 前処理部を別ファイルに分離する
- プロット関数を別ファイルに分離する
- パラメータ設定を YAML などに逃がす
- ピークマッチングの仕様を明文化する
- 推定結果の保存フォーマットを統一する

---

## 16. ひとことでまとめると

`test3.py` は、

**「GGG の THz 透過スペクトルを、ポラリトン領域を重視したベイズ推定で解析し、H-form と B-form を比較するための一括実行スクリプト」**

です。

最初は全部を理解しようとせず、まず

- `main()`
- `load_all_datasets()`
- `build_pymc_model()`
- `MixedOutputModelOp`

の 4 つを押さえると、全体像がかなり見えるようになります。

---

## 17. 関数対応表

`test3.py` の主要な関数とクラスを、役割ごとに整理した表です。  
「どこを読めば何が分かるか」をすぐ引けるようにしています。

| 種別 | 名前 | 行番号 | 役割 |
| --- | --- | --- | --- |
| 実行設定 | `resolve_sampling_resources()` | L161 | CPU 数や並列数を見て、`chains` と `cores` を安全側に調整する |
| 物理モデル | `get_hamiltonian()` | L189 | `g`, `B4`, `B6`, 外部磁場 `B` から 8x8 ハミルトニアンを作る |
| 物理モデル | `construct_spin_operators()` | L211 | スピン演算子 `Sx`, `Sy`, `Sz` を作る |
| 物理モデル | `calculate_susceptibility()` | L232 | ハミルトニアンを対角化して磁気感受率 `chi` を計算する |
| 物理モデル | `calculate_transmission()` | L312 | 比透磁率から Fabry-Perot 透過率を計算し、観測と同じスケールで正規化する |
| 物理モデル | `calculate_transmission_for_params()` | L362 | パラメータ一式から透過スペクトルを一気に計算する |
| 前処理 | `compute_fwhm_from_spectrum()` | L382 | 共振器領域の全ピークについて FWHM とピーク位置を求める |
| 前処理 | `compute_cavity_peak_info()` | L413 | 共振器領域で最も目立つピーク 1 本の FWHM と位置を返す |
| 前処理 | `detect_peaks_and_classify()` | L441 | ピークを検出し、ポラリトン領域と共振器領域に分類する |
| 前処理 | `create_weight_array()` | L466 | 周波数ごとの重み配列を作る |
| 前処理 | `normalize_obs_spectrum()` | L480 | 観測スペクトルを 0-1 に正規化し、そのときの `min/max` も返す |
| 前処理 | `load_all_datasets()` | L508 | Excel から全データセットを読み、正規化・ピーク抽出・重み付けまで行う |
| 参照値読み込み | `load_v8_optimized_params()` | L583 | v8 の最適化結果 CSV を読み、事前分布の中心として使う |
| PyMC 連携 | `MixedOutputModelOp` | L628 | PyMC から呼ばれる中核クラス。予測スペクトル、FWHM、ピーク位置を返す |
| 評価 | `compute_model_evaluation()` | L773 | R-hat や ESS をまとめて評価する |
| 評価 | `compare_models()` | L798 | H-form と B-form の ESS を比較する |
| 評価補助 | `_extract_lml_scalars()` | L813 | `log_marginal_likelihood` をフラット化して取り出す |
| 評価 | `compute_bayes_factor_smc()` | L832 | SMC の周辺尤度からベイズファクターを計算する |
| 可視化 | `plot_posterior_predictive_spectra()` | L873 | 1 モデル分の事後予測スペクトルを描く |
| 可視化 | `plot_posterior_predictive_spectra_combined()` | L970 | H-form と B-form の事後予測スペクトルを同じ図に重ねる |
| 可視化 | `plot_posterior_distributions()` | L1100 | 事後分布ヒストグラムを描く |
| 可視化 | `plot_energy_levels()` | L1160 | 推定されたパラメータからエネルギー準位図を描く |
| 可視化 | `plot_susceptibility()` | L1223 | 感受率の実部・虚部を描く |
| ベイズモデル | `build_pymc_model()` | L1330 | 事前分布と尤度を定義して PyMC モデルを構築する |
| 実行入口 | `main()` | L1559 | 全処理の流れをまとめて実行する |

### まず読むべき関数

初学者なら、次の順で読むのがおすすめです。

1. `main()`  
2. `load_all_datasets()`  
3. `build_pymc_model()`  
4. `MixedOutputModelOp`  
5. `calculate_transmission_for_params()`  

### 目的別に見る場所

| やりたいこと | 最初に見る関数 |
| --- | --- |
| 実験データの読み込み方法を知りたい | `load_all_datasets()` |
| 物理モデルの式の流れを知りたい | `get_hamiltonian()`, `calculate_susceptibility()`, `calculate_transmission()` |
| 尤度の作り方を知りたい | `build_pymc_model()` |
| ピーク位置や FWHM の扱いを知りたい | `compute_fwhm_from_spectrum()`, `MixedOutputModelOp.perform()` |
| モデル比較のやり方を知りたい | `compare_models()`, `compute_bayes_factor_smc()` |
| 出力ファイルがどこで保存されるか知りたい | `main()` |

---

## 18. 処理フロー図

`test3.py` の処理を、コードの実行順に近い形で図にすると次のようになります。

```text
[開始]
  ↓
設定値の読み込み
  - SMC_DRAWS
  - SMC_CHAINS
  - DEBUG_MODE
  - FIX_B6_ZERO
  ↓
実行リソース調整
  - resolve_sampling_resources()
  ↓
v8 の参照パラメータ読み込み
  - load_v8_optimized_params('H')
  - load_v8_optimized_params('B')
  ↓
実験データ読み込み
  - load_all_datasets()
  ↓
各データセットで前処理
  - normalize_obs_spectrum()
  - detect_peaks_and_classify()
  - create_weight_array()
  - compute_fwhm_from_spectrum()
  - compute_cavity_peak_info()
  ↓
H-form の PyMC モデル構築
  - build_pymc_model(..., 'H', ...)
  ↓
H-form の SMC サンプリング
  - pm.sample_smc()
  ↓
B-form の PyMC モデル構築
  - build_pymc_model(..., 'B', ...)
  ↓
B-form の SMC サンプリング
  - pm.sample_smc()
  ↓
モデル評価
  - compute_model_evaluation()
  - compare_models()
  - compute_bayes_factor_smc()
  ↓
可視化
  - plot_posterior_distributions()
  - plot_posterior_predictive_spectra()
  - plot_posterior_predictive_spectra_combined()
  - plot_energy_levels()
  - plot_susceptibility()
  ↓
結果保存
  - trace_*.nc / *.pkl
  - summary_*.csv
  - parameters_*.csv
  - model_evaluation.json
  ↓
[終了]
```

### 物理計算だけ抜き出した流れ

実験データとの比較の中で、モデルが 1 回呼ばれると内部では次の順に計算しています。

```text
パラメータ
(g, a, B4, B6, eps_bg, gamma)
  ↓
get_hamiltonian()
  ↓
calculate_susceptibility()
  ↓
H-form なら mu_r = 1 + chi
B-form なら mu_r = 1 / (1 - chi)
  ↓
calculate_transmission()
  ↓
予測スペクトル
  ↓
ピーク位置 / FWHM 抽出
  ↓
尤度計算
```

### 後輩向けの見方

処理の本線は次の 3 ブロックです。

- 入力と前処理: `load_all_datasets()`
- ベイズ推定本体: `build_pymc_model()` と `MixedOutputModelOp`
- 結果確認: `plot_*()` と `main()`

---

## 19. 変数対応表

このスクリプトで特に重要な物理パラメータの意味をまとめます。

| 変数名 | 意味 | コード中で主に使う場所 | スペクトルへの主な影響 | 初学者向けメモ |
| --- | --- | --- | --- | --- |
| `g` | g 因子 | `get_hamiltonian()`, `calculate_transmission_for_params()` | 共鳴位置、エネルギー準位差 | 磁場に対してどれだけ準位が割れるかを決める |
| `a` | 感受率の強度スケール | `calculate_transmission_for_params()` | 吸収や共鳴の強さ | 実験と理論の強度差を吸収する補正係数 |
| `B4` | 結晶場係数 4 次項 | `get_hamiltonian()` | 準位構造、共鳴位置 | GGG 固有の結晶場の効き方を表す |
| `B6` | 結晶場係数 6 次項 | `get_hamiltonian()` | 準位構造の細かい補正 | 情報量が少ないと不安定になりやすいので固定する場合もある |
| `eps_bg` | 背景誘電率 | `calculate_transmission()`, `build_pymc_model()` | Fabry-Perot のピーク位置、FSR | 共振器の山の間隔に強く効く |
| `gamma` | 緩和係数群 | `calculate_susceptibility()` | 線幅、ピークの太さ | このコードでは 7 個まとめて扱う |

### `gamma` が 7 個ある理由

`gamma` は 1 個の値ではなく、`gamma_1` から `gamma_7` まであります。  
これは、全ての遷移を同じ線幅で扱うのではなく、**遷移ごとに異なる広がりを許す**ためです。

コード上では、低エネルギー側の準位番号に応じて各遷移へ `gamma` を割り当てています。

### 変数名と保存名の対応

コード内部の変数名と、出力ファイルに保存される名前はほぼ同じです。

| コード内部 | 保存時の主な名前 |
| --- | --- |
| `g_factor_scaled` | `g` |
| `a_scale_scaled` | `a` |
| `B4_scaled` | `B4` |
| `B6_scaled` | `B6` |
| `eps_bg_scaled` | `eps` |
| `gamma_1_scaled` ... `gamma_7_scaled` | `gamma` または `gamma_1` ... `gamma_7` |

### 変更時の注意

- `g`, `B4`, `B6` を変えると、まずハミルトニアンが変わる
- `a` と `gamma` は、主に線の強さと幅に効く
- `eps_bg` は、共振器のピーク間隔や位置に効く
- 尤度を触る前に、どの変数がどの見た目に効くかを把握しておくと安全

---

## 20. 出力ファイル対応表

実行後に結果ディレクトリへ保存される主なファイルと、その意味をまとめます。

### 数値データ

| ファイル名 | 形式 | 何が入っているか | 主な用途 |
| --- | --- | --- | --- |
| `trace_H.nc` / `trace_B.nc` | NetCDF | H-form / B-form の事後サンプル全体 | 後で再解析したいとき |
| `trace_H.pkl` / `trace_B.pkl` | Pickle | NetCDF 保存に失敗したときの代替保存 | 緊急退避用 |
| `summary_H.csv` / `summary_B.csv` | CSV | ArviZ のサマリー統計量 | 平均、HDI、ESS、R-hat の確認 |
| `parameters_H.csv` / `parameters_B.csv` | CSV | 主要パラメータの平均値 | 結果の一覧確認 |
| `model_evaluation.json` | JSON | モデル比較結果や設定値 | 解析条件の記録、比較結果の記録 |

### 図ファイル

| ファイル名 | 何の図か | 見るポイント |
| --- | --- | --- |
| `posterior_distributions_H.png` / `posterior_distributions_B.png` | 事後分布ヒストグラム | パラメータがどこに集まっているか |
| `posterior_predictive_spectra_H.png` / `posterior_predictive_spectra_B.png` | 各モデル単体の事後予測スペクトル | 実験スペクトルと理論スペクトルの一致具合 |
| `posterior_predictive_spectra_HB.png` | H/B 重ね描きスペクトル | どちらのモデルが見た目で良いか |
| `energy_levels_H.png` / `energy_levels_B.png` | エネルギー準位図 | パラメータ推定後の準位構造 |
| `susceptibility_real_H.png` / `susceptibility_real_B.png` | 感受率の実部 | 分散的な応答の形 |
| `susceptibility_imag_H.png` / `susceptibility_imag_B.png` | 感受率の虚部 | 吸収に対応する形 |

### どの関数がどのファイルを作るか

| 保存ファイル | 主に保存する関数 |
| --- | --- |
| `posterior_predictive_spectra_H.png`, `posterior_predictive_spectra_B.png` | `plot_posterior_predictive_spectra()` |
| `posterior_predictive_spectra_HB.png` | `plot_posterior_predictive_spectra_combined()` |
| `posterior_distributions_H.png`, `posterior_distributions_B.png` | `plot_posterior_distributions()` |
| `energy_levels_H.png`, `energy_levels_B.png` | `plot_energy_levels()` |
| `susceptibility_real_*.png`, `susceptibility_imag_*.png` | `plot_susceptibility()` |
| `trace_*.nc`, `trace_*.pkl` | `main()` |
| `summary_*.csv` | `main()` |
| `parameters_*.csv` | `main()` |
| `model_evaluation.json` | `main()` |

### 後輩がまず確認すべき出力

全部見るのは大変なので、最初は次の順がおすすめです。

1. `model_evaluation.json`
2. `summary_H.csv`, `summary_B.csv`
3. `posterior_predictive_spectra_HB.png`
4. `posterior_distributions_H.png`, `posterior_distributions_B.png`

### `model_evaluation.json` で見るべき項目

特に次を見ると、解析の状況がすぐ分かります。

- `comparison_ess`
- `comparison_bayes_factor`
- `sampler`
- `likelihood`
- `fix_B6_zero`
- `sigma_fwhm`
- `peak_match_max_distance`
- `normalization`
