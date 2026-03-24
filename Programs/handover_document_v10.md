# 引き継ぎドキュメント: `bayesian_v10_polariton_priority.py`

## GGG (Gd₃Ga₅O₁₂) THz磁気光学応答のベイズ推定プログラム

---

## 0. このドキュメントの目的

本ドキュメントは、`bayesian_v10_polariton_priority.py` の理論的背景・プログラム構造・各関数の詳細を、物理工学専攻学部4年生が理解できるレベルで解説する引き継ぎ資料である。

理論的根拠は「Master_thesis_24NC230_Nakao.pdf」（修士論文）に基づく。

---

## 1. 研究の背景と目的

### 1.1 何を推定しているのか

GGG (Gadolinium Gallium Garnet; Gd₃Ga₅O₁₂) という希土類磁性体結晶の**THz帯における透過スペクトル**を、物理モデルから理論的に計算し、Kritzellらの実験データ（磁場依存性6条件 + 温度依存性4条件 = 計10データセット）と比較する。

具体的には、2つの物理モデル（**Hモデル**と**Bモデル**）のどちらが実験データをより良く説明するかを、**ベイズ推定**によって統計的に判定する。これは**超放射相転移 (SRPT)** の発現可能性に直結する物理的に重要な問題である。

### 1.2 Hモデル vs Bモデル

GGG中のGd³⁺イオンの磁気モーメント **d̂** = −g_J μ_B Ŝ と電磁波の相互作用（Zeeman相互作用）の記述方法が2通りある：

- **Bモデル**: Ĥ'_B = −d̂ · B （磁束密度Bで駆動）
  - 磁気モーメントは光子場のみに結合 → no-go定理を回避でき、SRPTが実現可能
- **Hモデル**: Ĥ'_H = −d̂ · μ₀H （磁場Hで駆動）
  - H = B/μ₀ − M であるため、磁化Mを通じてスピン間の相互作用が実効的に含まれる → A²項に相当する相互作用が現れ、SRPTが抑制される

つまり、**どちらのモデルが実験を再現するかは SRPT の実現可否に直結する**。

### 1.3 ベイズ推定を使う理由

従来の最小二乗法は単一の最適解（点推定）を求めるだけで、パラメータの不確実性や相関を評価できない。ベイズ推定は：

- パラメータを確率変数とみなし、**事後分布**（パラメータがどの値をとりやすいか）全体を求める
- パラメータ間の相関を定量的に評価できる
- **ベイズファクター**（周辺尤度比）を用いて、HモデルとBモデルの統計的な比較が可能

---

## 2. 物理モデルの理論（プログラムの裏にある物理）

プログラムの計算パイプラインは以下の順序で動作する：

```
パラメータ (g, a, B4, B6, εbg, γ)
  ↓
ハミルトニアン構築 & 対角化
  ↓
磁気感受率 χ(ω) の計算（線形応答理論）
  ↓
比透磁率 μr(ω) の計算（H形式 or B形式で異なる）
  ↓
転送行列法による透過スペクトル T(ω) の計算
  ↓
実験データとの比較（尤度の計算）
```

### 2.1 ハミルトニアン（修論 式(2.2)-(2.4)）

GGG中のGd³⁺イオン（4f⁷, S = 7/2, 8準位系）の無摂動ハミルトニアン：

```
Ĥ₀ = Ĥ_CF + Ĥ_Zeeman
```

#### 2.1.1 結晶場ハミルトニアン Ĥ_CF（修論 式(2.3), 付録C）

```
Ĥ_CF = B₄(Ô⁰₄ + 5Ô⁴₄) + B₆(Ô⁰₆ − 21Ô⁴₆)
```

- **B₄, B₆** は結晶場パラメータ（推定対象、mKオーダー）
- **Ô^q_k** はStevens演算子（結晶の対称性を反映する演算子）
- GGGはガーネット構造（立方晶の一種）で、対称性に許される項のみ残る
- **物理的意味**: 結晶中の周囲のイオンが作る静電場により、スピン準位の縮退が解ける（ゼロ磁場分裂）

#### 2.1.2 Zeeman項 Ĥ_Zeeman（修論 式(2.4)）

```
Ĥ_Zeeman = g_J μ_B Ŝ · B_ext
```

- **g_J** はランデのg因子（Gd³⁺の理論値 ≈ 2.0、推定対象）
- **μ_B** はボーア磁子
- **B_ext** は外部印加静磁場（実験パラメータ、4.2T〜9T）
- **物理的意味**: 外部磁場によりスピン準位がさらに分裂する（Zeeman分裂）

#### 2.1.3 プログラムでの実装: `get_hamiltonian()`（L146-165）

```python
def get_hamiltonian(B_ext_z, g_factor, B4, B6, s=3.5):
    n_states = 8  # 2*3.5 + 1 = 8準位
    m_values = [3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5]

    # Sz = diag(m_values) → 8×8対角行列

    # Stevens演算子 O40, O44, O60, O64 を 8×8行列として構築
    # 具体的な行列要素は修論 付録C（Hutchingsの表）に基づく

    # 結晶場ハミルトニアン
    H_cf = B4 * (O40 + 5*O44) + B6 * (O60 - 21*O64)

    # Zeeman項（エネルギー単位: ケルビン [K]）
    # kBで割ることでケルビン単位に変換
    H_zee = g_factor * muB * B_ext_z * Sz / kB

    return H_cf + H_zee  # 8×8実対称行列
```

**注意点**:
- Stevens演算子の行列要素はハードコードされている（8×8に限定）
- 返される行列のエネルギー単位はケルビン [K]（kB で割っている）

### 2.2 磁気感受率の計算（修論 式(2.8), 付録F-H）

#### 2.2.1 理論的背景

線形応答理論（久保公式）により、周波数 ω の交流磁場に対するスピン系の応答関数を計算する：

```
χ^R_ij(ω) = −(Nμ₀/ℏ) Σ_{n,n'} (P_n − P_{n'}) ×
             { <ψ_n|d̂_i|ψ_{n'}><ψ_{n'}|d̂_j|ψ_n> / (ω + iγ − Δω_{n',n})
             − <ψ_n|d̂_j|ψ_{n'}><ψ_{n'}|d̂_i|ψ_n> / (ω + iγ + Δω_{n',n}) }
```

ここで：
- **N** = スピン密度（1.9386×10²⁸ m⁻³）
- **P_n** = ボルツマン分布に基づく準位nの占有確率
- **Δω_{n',n}** = (E_{n'} − E_n)/ℏ （準位間のエネルギー差に対応する角振動数）
- **γ** = 緩和レート（スペクトルの線幅に対応）
- **d̂_i** = 磁気モーメント演算子の i 成分

Faraday配置（磁場 B ∥ k ∥ z）では、円偏光基底での感受率 χ⁺ を使用する（修論 式(2.9)）：

```
χ⁺ = χ^R_xx + i χ^R_xy
```

本プログラムでは、Sx成分とSy成分の遷移行列要素の平均 (|Sx|² + |Sy|²)/2 を計算することで、これを実装している。

#### 2.2.2 プログラムでの実装: `calculate_susceptibility()`（L189-266）

この関数は以下のステップで感受率を計算する：

**Step 1**: ハミルトニアンの対角化

```python
eigenvalues_K, eigenvectors = np.linalg.eigh(H)
```
- `eigh` は実対称行列の固有値・固有ベクトルを返す（エルミート行列用）
- 固有値はケルビン単位、固有ベクトルは |ψ_n⟩

**Step 2**: ボルツマン分布の計算

```python
E_shifted_K = eigenvalues_K - E_min  # 基底状態を0にシフト
boltzmann_exp = np.clip(E_shifted_K / T, -700, 700)  # オーバーフロー防止
Z = np.sum(np.exp(-boltzmann_exp))  # 分配関数
populations = np.exp(-boltzmann_exp) / Z  # 各準位の占有確率
```

**Step 3**: スピン演算子の構築と固有状態基底への変換

```python
Sx, Sy, Sz = construct_spin_operators()  # |m⟩基底でのスピン演算子
Sx_eb = eigenvectors.T.conj() @ Sx @ eigenvectors  # 固有状態基底へ変換
Sy_eb = eigenvectors.T.conj() @ Sy @ eigenvectors
```

**Step 4**: 遷移行列要素と感受率の計算

```python
transition_perp = (|Sx_eb|² + |Sy_eb|²) / 2  # 円偏光基底での遷移強度
pop_diff = P_n - P_{n'}  # 占有数差

# Lorentzian型の応答関数（反共鳴項は省略した近似形）
chi = Σ (pop_diff × transition_perp) / (f₀ - f - iγ)
```

**Step 5**: 共有γモデルによる緩和係数の割り当て

7つの緩和係数 γ₀〜γ₆ を、エネルギーの低い方の準位インデックスに基づいて各遷移に割り当てる（修論 表3.1）。

### 2.3 比透磁率の定式化（修論 式(2.13), (2.18)）

`calculate_susceptibility()` が返す `chi_raw` に対し、スケーリング係数とモデル固有の変換を適用する：

```python
# スケーリング: 結合強度を反映
G0 = a * μ₀ * N * (g * μ_B)² / (2ℏ) / (2π × 10¹²)
chi = G0 * chi_raw

# Hモデル (修論 式(2.13))
mu_r = 1 + χ

# Bモデル (修論 式(2.18))
mu_r = 1 / (1 − χ)
```

**パラメータ `a` の物理的意味**: 実効的なスピン密度や試料充填率に依存する吸収強度の補正係数。理論上 a = 1 だが、実際には試料の品質やモデルの近似に起因するずれを許容する。

**HモデルとBモデルの違い**:
- Hモデルでは μ\_r = 1 + χ → 磁気応答が加法的
- Bモデルでは μ\_r = 1/(1 − χ) → 磁化による自己無撞着的な増強効果が含まれる

### 2.4 転送行列法による透過スペクトル（修論 §2.4, 式(2.35)）

GGG試料を「真空 | GGG（厚さ d） | 真空」の3層構造としてモデル化し、Fabry-Pérot干渉を含む透過スペクトルを計算する。

#### プログラムでの実装: `calculate_transmission()`（L269-295）

```python
def calculate_transmission(freq_thz, mu_r, d, eps_bg):
    omega = freq_thz × 2π × 10¹²        # 角振動数 [rad/s]
    n_complex = √(eps_bg × mu_r)          # 複素屈折率
    impe = √(mu_r / eps_bg)               # 規格化特性インピーダンス Z
    delta = 2π × n_complex × d / λ₀       # 位相差

    # 転送行列法（修論 式(2.35)）
    t = 4Z × e^{iδ} / [(1+Z)² − (1−Z)² × e^{2iδ}]
    transmission = |t|²

    # 0-1に正規化
    return (transmission - min) / (max - min)
```

**注意**: 正規化は最小値-最大値間の線形スケーリング。これは実験データが正規化されているため。

---

## 3. 推定パラメータ一覧

| パラメータ | 物理的意味 | 推定値の目安 |
|---|---|---|
| **g** (g因子) | Zeeman分裂幅を決定 → スペクトルの共鳴周波数を支配 | ≈ 2.0 (Gd³⁺理論値) |
| **a** (スケーリング係数) | 吸収強度の補正、χ ∝ a·g² | 数値は物理的意味上 ≈ 1 だが、モデルにより変動 |
| **B₄** (結晶場パラメータ) | 4次Stevens演算子の係数、ゼロ磁場分裂の大きさ | mKオーダー |
| **B₆** (結晶場パラメータ) | 6次Stevens演算子の係数、微細構造への寄与 | mKオーダー |
| **ε\_bg** (背景誘電率) | Fabry-Pérot共振器のFSRを決定 | ≈ 14 |
| **γ₁〜γ₇** (緩和係数) | 各遷移のスペクトル線幅 (Lorentzian幅) | ≈ 0.01〜0.5 THz |

---

## 4. ベイズモデルの構造

### 4.1 全体のアーキテクチャ

```
┌─────────────────────────────────────────┐
│         PyMC 確率モデル                  │
│                                          │
│  事前分布 → パラメータ                    │
│      ↓                                   │
│  MixedOutputModelOp (カスタムOp)         │
│  = 物理計算を PyMC グラフ内で実行         │
│      ↓                                   │
│  3つの出力:                               │
│    1. trans_pred (全データセット連結)      │
│    2. fwhm_pred (共振器ピークFWHM)       │
│    3. peak_freq_pred (共振器ピーク位置)   │
│      ↓                                   │
│  尤度（4つの Potential）                  │
│    1. ポラリトン領域スペクトル（最優先）  │
│    2. 共振器ピーク位置                    │
│    3. 共振器ピークFWHM                   │
│    4. 背景領域スペクトル（補助）          │
│                                          │
│  → SMCサンプリングで事後分布を推定       │
└─────────────────────────────────────────┘
```

### 4.2 事前分布の設定（修論 表3.7, `build_pymc_model()` L1100-1253）

各パラメータの事前分布は物理的制約に基づく：

#### g因子: TruncatedNormal

```python
g_factor_scaled = pm.TruncatedNormal(
    mu=2.0 * 38.0,    # 理論値 g ≈ 2.0 をスケーリング
    sigma=0.05 * 38.0, # 狭い分布で理論値近傍に制約
    lower=1.5 * 38.0, upper=2.8 * 38.0)
```

- Gd³⁺の理論的g因子は ≈ 2.0（4f⁷, L=0, S=7/2 → g_J = 2.0）
- σ = 0.05 と狭く設定：g因子は物理的にほぼ確定している

#### a (スケーリング係数): HalfNormal

```python
a_raw = pm.HalfNormal(sigma=3.0)
a_scale_scaled = pm.Deterministic(clip(a_raw, 0.1, 12.0) * 10.2)
```

- HalfNormal: 正値かつ小さい値が優先される
- 物理的に a > 0 かつ過度に大きくならない制約

#### B₄, B₆ (結晶場パラメータ): Normal

```python
B4_raw = pm.Normal(mu=0, sigma=0.025)    # σ = 25mK
B4_scaled = clip(B4_raw, -0.075, 0.075) * 1672.0

B6_raw = pm.Normal(mu=0, sigma=0.005)    # σ = 5mK
B6_scaled = clip(B6_raw, -0.025, 0.025) * 25000.0
```

- ゼロ中心：先行研究で非常に小さい値が示唆されている
- B₄ の方が B₆ よりも大きい値をとりうる（4次の方が支配的）

#### ε\_bg (背景誘電率): TruncatedNormal

```python
eps_bg_scaled = pm.TruncatedNormal(
    mu=eps_v6_avg * 17.0,  # v6最小二乗法の結果を中心値に
    sigma=0.3 * 17.0,
    lower=13.0 * 17.0, upper=16.0 * 17.0)
```

- 先行研究のフィッティング結果を中心値として利用

#### γ (緩和係数): Non-centered 階層モデル

```python
# ハイパーパラメータ
log_gamma_mu = pm.Normal(mu=log(0.074), sigma=0.3)   # 対数空間の平均
log_gamma_sd = pm.HalfNormal(sigma=0.3)               # 対数空間の標準偏差

# Non-centered parameterization
gamma_raw = pm.Normal(mu=0, sigma=1, shape=7)  # 標準正規変数 z_i
gamma_i = exp(log_gamma_mu + log_gamma_sd * gamma_raw)  # 変換
```

**なぜ階層モデルか**: 7つのγを独立に推定すると、異なる組み合わせが同じ尤度を与える多峰性の問題が生じる。共通のハイパーパラメータ（log_gamma_mu, log_gamma_sd）から生成する構造にすることで、この問題を緩和する。

**なぜ Non-centered か**: 階層モデルでは sigma が小さい場合に「漏斗問題（funnel problem）」が発生し、サンプリングが非効率になる。Non-centered parameterization（z_i を標準正規から直接サンプル）により、ハイパーパラメータと個別パラメータのサンプリングを分離し、効率を改善する。

### 4.3 パラメータのスケーリング

物理パラメータは桁数が大きく異なる（g ≈ 2 vs B₆ ≈ 0.001 mK）。サンプラーの効率を上げるため、各パラメータにスケーリング係数を乗じて同程度のオーダーに揃える：

```python
SCALING_FACTORS = {
    'g':     38.0,   # g=2.0 → scaled=76
    'a':     10.2,   # a=1.0 → scaled=10.2
    'B4':    1672.0, # B4=0.02 → scaled=33.4
    'B6':    25000.0,# B6=-0.001 → scaled=-25
    'eps':   17.0,   # eps=14 → scaled=238
    'gamma': 100.0   # gamma=0.1 → scaled=10
}
```

物理値に戻すときは：`physical_value = scaled_value / SCALING_FACTOR`

### 4.4 カスタムOp: `MixedOutputModelOp`（L534-659）

PyMCは通常、微分可能な関数のみモデルに組み込める。しかし、本プログラムの物理計算（ハミルトニアンの対角化、ピーク検出等）は微分不可能なため、PyTensorの**カスタムOp**として実装している。

#### 仕組み

```python
class MixedOutputModelOp(Op):
    def make_node(self, ...):
        # PyTensorの計算グラフに組み込むためのインターフェース定義
        # 入力: 6つのスカラー/ベクトル
        # 出力: 3つのベクトル (trans, fwhm, peak_freq)

    def perform(self, _node, inputs, output_storage):
        # 実際の物理計算を NumPy で実行
        # PyMCのサンプラーが propose したパラメータ値を受け取り
        # 透過スペクトル・FWHM・ピーク位置を計算して返す
```

**注意**: カスタムOpは勾配情報を持たないため、NUTS（勾配ベースのサンプラー）は使えない。代わりに**SMC（Sequential Monte Carlo）**を使用する。SMCは勾配不要のサンプラーで、粒子フィルタの原理に基づく。

#### `perform()` の処理フロー

1. スケーリングされたパラメータを物理値に復元
2. 各データセット（10条件）について：
   - ハミルトニアン構築 → 対角化
   - 感受率計算 → 比透磁率 → 透過スペクトル
3. 共振器領域のピーク検出（`find_peaks()`）
4. **マルチピーク近傍マッチング**: 実験ピークと理論ピークの対応付け
   - 最大距離 = 0.12 THz（半FSR程度）以内の最近傍を対応付け
   - マッチしなかった場合はペナルティ値を出力

### 4.5 尤度構造（v10.0の核心）

v10.0の最大の特徴は「**ポラリトン最優先**」の尤度設計：

#### 4.5.1 ポラリトン領域（f < 0.3615 THz）: スペクトル形状尤度

```python
# StudentT分布（ν=4, 外れ値に頑健）
T_obs ~ StudentT(ν=4, μ=T_pred, σ=σ_eff)
# σ_eff = 0.01 / √w,   w=2.0（ポラリトン）→ σ≈0.007
```

- ポラリトン領域は Zeeman ポラリトン形成の核心領域
- 最も重みが大きい（σが小さい = 精度要求が高い）

#### 4.5.2 共振器領域（f > 0.45 THz）: ピーク位置 + FWHM制約のみ

```python
# ピーク位置マッチング
peak_freq_pred ~ StudentT(ν=4, σ=0.005 THz)

# FWHM マッチング
fwhm_pred ~ StudentT(ν=4, σ=0.005 THz)
```

- **v10.0の改善点**: 共振器領域のスペクトル全体をフィッティングしない
- v9.0では共振器全体のStudentT尤度が結合定数aを2倍に歪める問題があった
- 共振器は「ピーク位置（ε\_bg制約）」と「FWHM（γ制約）」の**点制約のみ**

#### 4.5.3 背景領域: 補助的スペクトル尤度

```python
# 低い重み w=0.01 → σ_eff = 0.01/√0.01 = 0.1（ゆるい制約）
T_obs ~ StudentT(ν=4, μ=T_pred, σ=0.1)
```

#### 4.5.4 なぜ `pm.Potential` を使うか

通常 PyMC では `pm.StudentT('name', observed=data)` で尤度を定義するが、今回は**領域ごとに異なる尤度**を適用する必要がある。`pm.Potential` は対数尤度を直接加算する機能で、柔軟な尤度設計が可能：

```python
pm.Potential('ll_polariton', log_likelihood_polariton.sum())
pm.Potential('ll_cavity_fwhm', log_likelihood_fwhm.sum())
pm.Potential('ll_cavity_peak_freq', log_likelihood_peak.sum())
pm.Potential('ll_background', log_likelihood_bg.sum())
```

---

## 5. プログラム構造（関数一覧）

### 5.1 物理計算関数群

| 関数 | 行番号 | 役割 |
|---|---|---|
| `get_hamiltonian()` | L146 | 結晶場 + Zeeman ハミルトニアン（8×8行列）を構築 |
| `construct_spin_operators()` | L168 | Sx, Sy, Sz 演算子を |m⟩ 基底で構築 |
| `calculate_susceptibility()` | L189 | 線形応答理論に基づく磁気感受率 χ(ω) を計算 |
| `calculate_transmission()` | L269 | 転送行列法で透過スペクトルを計算 |
| `calculate_transmission_for_params()` | L298 | 上記3関数をまとめたラッパー |

### 5.2 ピーク検出・重み関連

| 関数 | 行番号 | 役割 |
|---|---|---|
| `detect_peaks_and_classify()` | L400 | スペクトルのピークをポラリトン/共振器に分類 |
| `create_weight_array()` | L425 | 周波数ごとの重み配列を生成 |
| `compute_fwhm_from_spectrum()` | L317 | 共振器ピークのFWHM計算（全ピーク対応） |
| `compute_cavity_peak_info()` | L372 | 最も顕著なピークのFWHMと位置を返す |

### 5.3 データ入出力

| 関数 | 行番号 | 役割 |
|---|---|---|
| `load_all_datasets()` | L439 | Excelファイルから10データセットを読み込み |
| `load_v6_optimized_params()` | L502 | v6最小二乗法の結果をJSON読み込み（事前分布の中心値に使用） |

### 5.4 ベイズモデル構築

| 関数 | 行番号 | 役割 |
|---|---|---|
| `MixedOutputModelOp` (class) | L534 | PyTensorカスタムOp: 物理計算をサンプラーに接続 |
| `build_pymc_model()` | L1100 | PyMCモデルの構築（事前分布 + 尤度の定義） |

### 5.5 モデル評価

| 関数 | 行番号 | 役割 |
|---|---|---|
| `compute_model_evaluation()` | L665 | R-hat, ESS等の収束診断 |
| `compare_models()` | L690 | H/BモデルのESS比較 |
| `compute_bayes_factor_smc()` | L724 | SMC周辺尤度からベイズファクターを計算 |

### 5.6 可視化

| 関数 | 行番号 | 役割 |
|---|---|---|
| `plot_posterior_predictive_spectra()` | L765 | 事後予測スペクトル（実験 vs 理論、94% HDI） |
| `plot_posterior_distributions()` | L857 | パラメータの事後分布ヒストグラム |
| `plot_energy_levels()` | L917 | 磁場 vs エネルギー準位図（94% HDI） |
| `plot_susceptibility()` | L993 | 磁気感受率 Re(χ₊), Im(χ₊) の周波数依存性 |

### 5.7 メイン実行フロー

`main()` (L1259-1392):

```
1. v6最適化結果の読み込み
2. 結果保存ディレクトリの作成
3. 10データセットの読み込み
4. H形式モデル構築 → SMCサンプリング
5. B形式モデル構築 → SMCサンプリング
6. モデル評価（R-hat, ESS, ベイズファクター）
7. 可視化（4種類のプロット × H/B）
8. 結果保存（trace, summary, parameters, evaluation）
```

---

## 6. 設定パラメータ（実行前に確認すべき項目）

### ファイル冒頭の定数（L38-70）

| 変数 | 値 | 意味 |
|---|---|---|
| `SAMPLER_TYPE` | 'SMC' | サンプラー種類（勾配不要） |
| `SMC_DRAWS` | 10000 | 各チェインのサンプル数 |
| `SMC_CHAINS` | 16 | チェイン数（並列実行） |
| `NU_STUDENTT` | 4 | Student-t分布の自由度 |
| `SIGMA_FWHM` | 0.005 | FWHM尤度の σ [THz] |
| `SIGMA_PEAK_FREQ` | 0.005 | ピーク位置尤度の σ [THz] |
| `PEAK_MATCH_MAX_DISTANCE` | 0.12 | ピークマッチングの最大距離 [THz] |
| `USE_BACKGROUND_LIKELIHOOD` | True | 背景尤度を使用するか |
| `DEBUG_MODE` | False | Trueで2データセット・500サンプルに縮小 |

---

## 7. 実行方法と必要環境

### 7.1 依存パッケージ

```
numpy, pandas, matplotlib, scipy
pymc (v5以上), arviz, pytensor
openpyxl (Excel読み込み用)
```

### 7.2 必要な入力ファイル

```
march/
├── bayesian_inputs/
│   ├── BayesianInput_Raw_Transmittance_Temperature.xlsx
│   └── BayesianInput_Raw_Transmittance_Field.xlsx
├── global_fitting_results_H_v6/
│   └── shared_gamma_params.json
└── global_fitting_results_B_v6/
    └── shared_gamma_params.json
```

- Excelファイル: Kritzellらの実験データ（正規化済み透過率スペクトル）
- JSONファイル: v6最小二乗法の結果（事前分布の中心値として使用）

### 7.3 実行コマンド

```bash
cd march/
python bayesian_v10_polariton_priority.py
```

### 7.4 出力

`march/bayesian_v9_results_YYYYMMDD_HHMMSS/` に以下が保存される：

- `trace_H.nc`, `trace_B.nc` — 事後分布のサンプル（NetCDF形式）
- `summary_H.csv`, `summary_B.csv` — ArviZ統計サマリー
- `parameters_H.csv`, `parameters_B.csv` — 事後平均パラメータ
- `model_evaluation.json` — ベイズファクター等の評価指標
- `posterior_predictive_spectra_H.png`, `_B.png` — 事後予測スペクトル
- `posterior_distributions_H.png`, `_B.png` — パラメータ事後分布
- `energy_levels_H.png`, `_B.png` — エネルギー準位図
- `susceptibility_real_H.png`, `_imag_H.png`, `_B.png` — 磁気感受率

---

## 8. バージョン変遷と設計思想

| バージョン | 主な変更点 |
|---|---|
| v6 | 最小二乗法によるベースライン（事前分布の中心値を提供） |
| v7 | ベイズ推定の導入、Non-centered 階層γモデル |
| v8 | 複合尤度（ポラリトン + 共振器スペクトル + FWHM） |
| v9 | マルチキャビティピーク対応、近傍マッチング導入 |
| **v10** | **共振器スペクトル全体の尤度を除去**（ポラリトン最優先方式） |

**v10.0の核心**: v9.0では共振器全体のStudentT尤度がパラメータ `a` を2倍に歪め、ポラリトンRMSEを 0.0918 → 0.2224 に劣化させていた（Ablation studyで確認）。v10.0では共振器はピーク位置とFWHMの点制約のみに変更し、ポラリトン精度を維持しつつ共振器の情報も活用する設計とした。

---

## 9. 今後の課題（修論 §4.4.4 参照）

1. **事前分布のパターン検討**: 現在のv6ベースの中心値を、v8最小二乗法結果に更新する可能性（コード中のコメントに記載）
2. **ベイズファクターの精度向上**: SMCの周辺尤度推定はチェイン数に依存するため、チェイン数の増加やrepeat実行を検討
3. **モデル拡張**: スピン間相互作用や結晶場パラメータの追加項の検討
4. **計算効率**: カスタムOpの高速化（JAX移植など）

---

## 10. 用語集

| 用語 | 説明 |
|---|---|
| **GGG** | Gd₃Ga₅O₁₂ (Gadolinium Gallium Garnet), 希土類ガーネット結晶 |
| **Zeeman分裂** | 外部磁場による準位分裂 |
| **Zeemanポラリトン** | 光子とスピン歳差運動の混成準粒子 |
| **EPR** | 電子常磁性共鳴 (Electron Paramagnetic Resonance) |
| **SRPT** | 超放射相転移 (Superradiant Phase Transition) |
| **Stevens演算子** | 結晶場を記述する角運動量多項式 |
| **FSR** | Free Spectral Range, Fabry-Pérot共振器の共鳴間隔 |
| **FWHM** | Full Width at Half Maximum, 半値全幅 |
| **HDI** | Highest Density Interval, 最高密度区間（94%HDI = 94%の確率でパラメータが存在する区間） |
| **SMC** | Sequential Monte Carlo, 勾配不要のベイズサンプラー |
| **ESS** | Effective Sample Size, 有効サンプルサイズ |
| **R-hat** | Gelman-Rubin統計量 (≈1.0で収束) |
| **StudentT** | 裾の重い分布、外れ値に頑健（ν=4で正規分布より裾が長い） |
| **Non-centered** | 階層モデルの漏斗問題を回避するためのパラメータ化手法 |
| **PyMC** | Pythonの確率的プログラミングライブラリ |
| **ArviZ** | ベイズ推定結果の可視化・診断ライブラリ |
