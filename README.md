# Bayesian

GGG (Gd3Ga5O12) の THz 透過スペクトル解析リポジトリです。  
主に以下 2 系統の解析コードを含みます。

- Bayesian 推定（PyMC/SMC）
- WLS（Weighted Least Squares）最適化

本リポジトリでは、H-form / B-form の 2 モデルを比較し、スペクトル形状・共振器ピーク位置・FWHM などを用いてパラメータ推定を行います。

## ディレクトリ構成

- `bayesian_inputs/`
  - 入力データ（Excel）
- `march/`
  - 主要スクリプトと結果
  - `bayesian_v10_polariton_priority.py`: 現行の Bayesian 推定コード
  - `wls_v8_mixed_fitting.py`: 現行の WLS コード
  - `handover_document_v10.md`: v10 の引き継ぎ解説
  - `WLS_v8_changelog.md`: WLS 変更履歴
  - `bayesian_v9_results_*/`, `wls_v8_results_*/`: 出力結果
- `Programs/`
  - 検証・補助スクリプト
- `Reference/`
  - 参照資料

## 実行環境

推奨: Python 3.10+（3.11 でも動作）

必要ライブラリ（主要）

- numpy
- pandas
- scipy
- matplotlib
- pymc
- arviz
- pytensor
- openpyxl

## セットアップ

### 1. 仮想環境の作成（例: conda）

```bash
conda create -n research python=3.11 -y
conda activate research
```

### 2. 依存パッケージの導入

```bash
pip install numpy pandas scipy matplotlib pymc arviz pytensor openpyxl
```

## 使い方

作業ディレクトリを `march/` にして実行するのが簡単です。

```bash
cd march
```

### Bayesian 推定（推奨メイン）

```bash
python bayesian_v10_polariton_priority.py
```

- 入力: `../bayesian_inputs/`（または `march/bayesian_inputs/`）
- 出力: `march/bayesian_v9_results_YYYYMMDD_HHMMSS/`

主な出力ファイル

- `trace_H.nc`, `trace_B.nc`
- `summary_H.csv`, `summary_B.csv`
- `parameters_H.csv`, `parameters_B.csv`
- `model_evaluation.json`
- 各種プロット（スペクトル、事後分布、エネルギー準位、感受率など）

### WLS フィッティング

```bash
python wls_v8_mixed_fitting.py
```

- 出力: `march/wls_v8_results_YYYYMMDD_HHMMSS/`

主な出力ファイル

- `parameters_H.csv`, `parameters_B.csv`
- `summary_H.csv`, `summary_B.csv`
- `correlation_H.csv`, `correlation_B.csv`
- `model_evaluation.json`
- 各種フィット図・残差図

## 計算負荷に関する注意

Bayesian 実行は CPU 負荷が高くなります。  
`bayesian_v10_polariton_priority.py` にはローカル端末向けの自動負荷制御が入っています。

- `BAYES_BLAS_THREADS`（BLAS スレッド数）
- `BAYES_MAX_CHAINS`（SMC 並列チェーン上限）
- `BAYES_MIN_CHAINS`（SMC 並列チェーン下限）

例:

```bash
BAYES_BLAS_THREADS=1 BAYES_MAX_CHAINS=4 python bayesian_v10_polariton_priority.py
```

## 現在の主対象コード

- Bayesian: `march/bayesian_v10_polariton_priority.py`
- WLS: `march/wls_v8_mixed_fitting.py`

旧バージョン（v8/v9 系）は比較・再現用として残しています。

## トラブルシュート

- 入力ファイルが見つからない場合
  - `bayesian_inputs/` がリポジトリ直下にあるか確認
- `ModuleNotFoundError` が出る場合
  - 仮想環境を有効化し、依存パッケージを再インストール
- 負荷が高すぎる場合
  - `BAYES_MAX_CHAINS` を 2〜4 に下げる
  - `BAYES_BLAS_THREADS=1` を指定

## ライセンス

必要に応じて追記してください（現状未設定）。
