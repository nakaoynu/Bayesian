# Bayesian

GGG (`Gd3Ga5O12`) の THz 透過スペクトル解析用リポジトリです。  
現在の作業用コードは主に `Programs/` にまとまっており、Bayesian 推定、WLS フィッティング、引き継ぎ資料、既存の解析結果を同じ場所で管理しています。

このリポジトリで主に扱っている内容は次の 2 系統です。

- Bayesian 推定: PyMC + SMC を用いた H-form / B-form の比較
- WLS フィッティング: 参照値作成と初期値評価

## 現在の構成

```text
Bayesian/
├── README.md
├── bayesian_inputs/
│   ├── BayesianInput_Raw_Transmittance_Temperature.xlsx
│   └── BayesianInput_Raw_Transmittance_Field.xlsx
├── Programs/
│   ├── ggg_bayesian_last.py
│   ├── ggg_bayesian_last.md
│   ├── bayesian_v10_polariton_priority.py
│   ├── handover_document_v10.md
│   ├── wls_v8_mixed_fitting.py
│   ├── WLS_v8_changelog.md
│   ├── BayesFacotr.md
│   ├── test3_regression_tests.py
│   ├── issues/
│   ├── wls_v8_results_20260306_160250/
│   └── bayesian_v10_results_*/
└── Reference/
    ├── Master_thesis_24NC230_Nakao.pdf
    └── 関連論文 PDF
```

## 主なファイル

| パス | 役割 |
| --- | --- |
| `Programs/ggg_bayesian_last.py` | 現在の Bayesian 解析コード |
| `Programs/ggg_bayesian_last.md` | 上記コードの引き継ぎ資料 |
| `Programs/bayesian_v10_polariton_priority.py` | v10 系の Bayesian スクリプト |
| `Programs/wls_v8_mixed_fitting.py` | WLS フィッティングコード |
| `Programs/wls_v8_results_20260306_160250/` | Bayesian 側が参照する WLS 結果 |
| `bayesian_inputs/` | 実験入力データ |
| `Reference/` | 修士論文と参照論文 |

`Programs/ggg_bayesian_last.py` と `Programs/ggg_bayesian_last.md` が、現在の引き継ぎ対象として最も見やすい組（code, document）です。

## 解析の概要

Bayesian 側では、GGG の THz 透過スペクトルに対して次を行います。

1. Excel から温度依存・磁場依存データを読み込む
2. スペクトルを前処理し、ピーク位置と FWHM を抽出する
3. 物理モデルから透過スペクトルを計算する
4. PyMC の SMC サンプラーでパラメータを推定する
5. H-form と B-form を Bayes factor などで比較する

設計上の特徴は、**ポラリトン領域のスペクトル形状を最優先で合わせる**ことです。  
共振器領域は補助拘束として使い、主にピーク位置と FWHM で効かせます。

## 実行環境

推奨は Python 3.10 以上です。  
主要な依存ライブラリは次の通りです。

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `pymc`
- `arviz`
- `pytensor`
- `openpyxl`

## セットアップ

### 1. 仮想環境

```bash
conda create -n research python=3.11 -y
conda activate research
```

### 2. 依存パッケージ

```bash
pip install numpy pandas scipy matplotlib pymc arviz pytensor openpyxl
```

## 使い方

作業ディレクトリはリポジトリ直下でも `Programs/` でも構いませんが、現状は `Programs/` に入って実行するのが分かりやすいです。

```bash
cd Programs
```

### Bayesian 推定

現行の入口は次です。

```bash
python ggg_bayesian_last.py
```

互換用の v10 スクリプトを使う場合は次です。

```bash
python bayesian_v10_polariton_priority.py
```

入力:

- `../bayesian_inputs/`
- `./wls_v8_results_20260306_160250/parameters_H.csv`
- `./wls_v8_results_20260306_160250/parameters_B.csv`

出力:

- `Programs/bayesian_v10_results_YYYYMMDD_HHMMSS/`

主な出力ファイル:

- `trace_H.nc`, `trace_B.nc`
- `summary_H.csv`, `summary_B.csv`
- `parameters_H.csv`, `parameters_B.csv`
- `model_evaluation.json`
- `posterior_predictive_spectra_HB.png`
- `posterior_distributions_H.png`, `posterior_distributions_B.png`

### WLS フィッティング

```bash
python wls_v8_mixed_fitting.py
```

出力:

- `Programs/wls_v8_results_YYYYMMDD_HHMMSS/`

主な出力ファイル:

- `parameters_H.csv`, `parameters_B.csv`
- `summary_H.csv`, `summary_B.csv`
- `correlation_H.csv`, `correlation_B.csv`
- `model_evaluation.json`

## CPU 負荷の調整

Bayesian 推定は SMC を使うため重いです。  
`ggg_bayesian_last.py` では次の環境変数で負荷を調整できます。

- `BAYES_BLAS_THREADS`
- `BAYES_MAX_CHAINS`
- `BAYES_MIN_CHAINS`

例:

```bash
BAYES_BLAS_THREADS=1 BAYES_MAX_CHAINS=4 python ggg_bayesian_last.py
```

コード内の設定で軽く試したい場合は、`DEBUG_MODE = True` も使えます。

## 最初に見るとよい資料

読む順番は次がおすすめです。

1. `Programs/ggg_bayesian_last.md`
2. `Programs/ggg_bayesian_last.py`
3. `Programs/wls_v8_results_20260306_160250/parameters_H.csv`
4. `Programs/BayesFacotr.md`

## トラブルシュート

| 症状 | 確認すること |
| --- | --- |
| 入力ファイルが見つからない | `bayesian_inputs/` がリポジトリ直下にあるか |
| WLS 参照 CSV が見つからない | `Programs/wls_v8_results_20260306_160250/` の場所と名前 |
| `ModuleNotFoundError` が出る | 仮想環境と依存パッケージ |
| 実行が重い | `BAYES_MAX_CHAINS` を下げる、`BAYES_BLAS_THREADS=1` を使う |

## 補足

過去の文書や一部テストファイルには `march/` や `test3.py` など旧名称が残っている可能性があります。  
現在の実作業では、まず `Programs/ggg_bayesian_last.py` を基準に読むのが安全です。
