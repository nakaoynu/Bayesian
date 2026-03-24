"""
Bayesian Hierarchical Analysis with Mixed Likelihood (v10.0)
領域別尤度: ポラリトン→透過率スペクトル形状(最優先), 共振器→ピーク位置+FWHM制約のみ

【v10.0 改善 — ポラリトン最優先・共振器ピーク制約方式】
★ v9.0 のトレードオフ問題を根本解決
  - v9.0 では共振器スペクトル全体の StudentT 尤度が結合定数 a を 2倍に歪め、
    ポラリトン RMSE を 0.0918→0.2224 に劣化させていた (Ablation study で確認)
  - v10.0 では共振器スペクトル形状尤度を完全除去
  - 共振器はピーク位置 (eps_bg制約) と FWHM (γ制約) のみの軽い点制約
  - ポラリトンスペクトル形状は v8 同等の精度を維持

★ 尤度構造 (v10.0):
  1. ポラリトン領域: StudentT スペクトル形状尤度 (σ=weight依存) ← 最優先
  2. 共振器マルチピーク位置: StudentT (σ_peak=0.005 THz) ← eps_bg制約
  3. 共振器マルチピーク FWHM: StudentT (σ_fwhm=0.005 THz) ← γ制約
  4. 背景領域: StudentT スペクトル形状 (weight=0.01) ← 補助
  ※ 共振器スペクトル全体の尤度は使用しない (v9.0 からの最大変更点)

★ マルチピーク近傍マッチング (v9.0 から継承)
  - 全キャビティピークを検出し近傍マッチング (max_distance = 半FSR)
  - 10 datasets × ~2 peaks = ~20 ピーク制約

【事前分布設定 (v8.1 修士論文に基づく改訂)】← 事前分布の設定を複数パターン用意する(2026/03/16 追記)
┌─────────┬──────────────┬────────────────────────────────────────┐
│ g       │TruncNormal   │理論値g≈2.0 (Gd³⁺), σ=0.05             │
│ a       │HalfNormal    │低値優先、σ=3.0、上限12                  │
│ B₄      │Normal        │負値許容、μ=0, σ=25mK, [-75,+75]mK     │
│ B₆      │Normal        │ゼロ中心対称、σ=5mK, [-25,+25]mK       │
│ ε_bg   │TruncNormal   │v8平均値中心、σ=0.3                    │
│log_γ_mu│Normal        │log空間で定義、μ=log(0.074)            │
│log_γ_sd│HalfNormal    │log空間標準偏差、σ=0.3                 │
│γ_raw_i │Normal(0,1)   │Non-centered: 標準正規分布              │
│ γ_i    │Deterministic │exp(log_μ + log_σ * z_i)               │
└─────────┴──────────────┴────────────────────────────────────────┘
"""

# ========== 設定 ==========
SAMPLER_TYPE = 'SMC'
USE_HIERARCHICAL_GAMMA = True
LIKELIHOOD_TYPE = 'mixed'       # 新: polariton=spectrum, cavity=FWHM
NU_STUDENTT = 4
RANDOM_SEED = 42

SMC_DRAWS   = 10000
SMC_CHAINS  = 16
SMC_PARALLEL = True

# 階層 γ ハイパーパラメータ (to do ; v7.1 準拠 → v8.1 修士論文に基づき改訂する)
GAMMA_HYPERPRIOR_MU    = 0.074
GAMMA_HYPERPRIOR_SIGMA = 0.160
GAMMA_STD_PRIOR        = 0.092

# 【新設定】FWHM 尤度のノイズレベル [THz]
# ポラリトン尤度 (~100点) とスケール整合するよう調整
SIGMA_FWHM = 0.005     # 初期値 5 GHz — チューニング対象
SIGMA_PEAK_FREQ = 0.005  # THz (5 GHz) — キャビティピーク位置の尤度σ

# スペクトル尤度の基準σ（WLS v8 の残差スケールを踏まえた現実的設定）
SIGMA_SPECTRUM_BASE = 0.12
SIGMA_SPECTRUM_MIN  = 0.01
SIGMA_SPECTRUM_MAX  = 0.20

# 【v9.0 新設定】共振器領域スペクトル形状フィッティングの σ
# v10.0: 共振器スペクトル全体の尤度は使用しない (v9のトレードオフ問題を解消)
# SIGMA_CAVITY_SPECTRUM は不要 — ピーク位置/FWHMの点制約のみ使用

# マルチピークマッチングの最大距離 [THz] (半FSR程度)
PEAK_MATCH_MAX_DISTANCE = 0.12

# 背景領域尤度を使用するか
USE_BACKGROUND_LIKELIHOOD = True

# B₆ を 0 に固定するか
# True : B₆ = 0 固定 (自由度削減・収束安定化。データが情報を持たない場合に推奨)
# False: B₆ を Normal 事前分布で推定 (真の事後分布を求める場合)
FIX_B6_ZERO = False

# デバッグモード: True にすると 2 データセット・500 サンプルで高速テスト
DEBUG_MODE = False

import os
import json
import time
import pathlib
import datetime
import warnings
warnings.filterwarnings('ignore')

# ローカル端末向け実行制御（必要に応じて環境変数で上書き）
DEFAULT_BLAS_THREADS = int(os.getenv('BAYES_BLAS_THREADS', '1'))
MAX_PARALLEL_CHAINS  = int(os.getenv('BAYES_MAX_CHAINS', '8'))
MIN_PARALLEL_CHAINS  = int(os.getenv('BAYES_MIN_CHAINS', '2'))

os.environ['OMP_NUM_THREADS'] = str(DEFAULT_BLAS_THREADS)
os.environ['MKL_NUM_THREADS'] = str(DEFAULT_BLAS_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(DEFAULT_BLAS_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(DEFAULT_BLAS_THREADS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(DEFAULT_BLAS_THREADS)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from scipy.signal import find_peaks, peak_widths

import logging
logging.getLogger('pytensor').setLevel(logging.ERROR)

# ============================================================================
# 物理定数
# ============================================================================
kB   = 1.380649e-23
muB  = 9.274010e-24
hbar = 1.054571e-34
c    = 299792458
mu0  = 4.0 * np.pi * 1e-7
eps0 = 8.854187817e-12

THZ_TO_HZ    = 1e12
THZ_TO_RAD_S = 2.0 * np.pi * THZ_TO_HZ
RAD_S_TO_THZ = 1.0 / THZ_TO_RAD_S

N_SPIN   = 1.9386e+28
d_fixed  = 157.8e-6

SCALING_FACTORS = {
    'g':    38.0,
    'a':    10.2,
    'B4':   1672.0,
    'B6':   25000.0,
    'eps':  17.0,
    'gamma': 100.0,
    'd':    1e6,     # m → μm スケール (157.8 μm → O(100) でサンプリング安定)
}

TARGET_DATA = [
    {'B': 9.0, 'T':  4.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '4K'},
    {'B': 9.0, 'T': 10.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '10K'},
    {'B': 9.0, 'T': 20.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '20K'},
    {'B': 9.0, 'T': 30.0, 'file': 'BayesianInput_Raw_Transmittance_Temperature.xlsx', 'sheet': 'Normalized Data', 'col': '30K'},
    {'B': 4.2, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '4.2T'},
    {'B': 5.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '5T'},
    {'B': 6.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '6T'},
    {'B': 7.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '7T'},
    {'B': 8.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '8T'},
    {'B': 9.0, 'T':  1.5, 'file': 'BayesianInput_Raw_Transmittance_Field.xlsx',       'sheet': 'Normalized Data', 'col': '9T'},
]

S_VALUE       = 3.5
N_TRANSITIONS = 7


def resolve_sampling_resources(requested_chains, parallel_enabled):
    """端末負荷の暴走を避けるため、chains/cores を安全側に自動調整する。"""
    logical_cpus = os.cpu_count() or 4
    # macOS + SMC では過剰並列にすると容易にスロットリングするため、
    # 論理CPUの半分程度を上限候補とする。
    safe_cap = max(1, logical_cpus // 2)
    env_cap = max(1, MAX_PARALLEL_CHAINS)
    min_cap = max(1, MIN_PARALLEL_CHAINS)
    max_allowed = max(min_cap, min(safe_cap, env_cap))

    if parallel_enabled:
        effective_chains = min(max_allowed, max(1, int(requested_chains)))
        effective_cores = effective_chains
    else:
        effective_chains = max(1, int(requested_chains))
        effective_cores = 1

    return {
        'logical_cpus': logical_cpus,
        'safe_cap': safe_cap,
        'effective_chains': effective_chains,
        'effective_cores': effective_cores,
    }


# ============================================================================
# 物理関数群 (test_fin_2a.py v7.1 と同一)
# ============================================================================
def get_hamiltonian(B_ext_z, g_factor, B4, B6, s=S_VALUE):
    n_states = int(2 * s + 1)
    m_values = np.arange(s, -s - 1, -1)
    Sz = np.diag(m_values)
    if n_states == 8:
        O40 = np.diag([7, -13, -3, 9, 9, -3, -13, 7]) / 60
        X_O44 = np.zeros((8, 8))
        X_O44[3, 7] = X_O44[4, 0] = np.sqrt(35) / 12
        X_O44[2, 6] = X_O44[5, 1] = 5 * np.sqrt(3) / 12
        O44 = X_O44 + X_O44.T
        O60 = np.diag([1, -5, 9, -5, -5, 9, -5, 1]) / 1260
        X_O64 = np.zeros((8, 8))
        X_O64[3, 7] = X_O64[4, 0] = 3 * np.sqrt(35) / 60
        X_O64[2, 6] = X_O64[5, 1] = -7 * np.sqrt(3) / 60
        O64 = X_O64 + X_O64.T
    else:
        raise ValueError(f"s={s} は未実装")
    H_cf  = B4 * (O40 + 5 * O44) + B6 * (O60 - 21 * O64)
    H_zee = g_factor * muB * B_ext_z * Sz / kB
    return H_cf + H_zee


def construct_spin_operators():
    s_val    = 3.5
    n_states = int(2 * s_val + 1)
    m_values = np.arange(s_val, -s_val - 1, -1)
    Sz = np.diag(m_values)
    Sx = np.zeros((n_states, n_states), dtype=float)
    Sy = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states - 1):
        m_lower = m_values[i + 1]
        coeff = np.sqrt((s_val - m_lower) * (s_val + m_lower + 1))
        Sx[i, i + 1] += coeff / 2.0
        Sy[i, i + 1] += -coeff / (2.0j)
    for i in range(1, n_states):
        m_upper = m_values[i - 1]
        coeff = np.sqrt((s_val + m_upper) * (s_val - m_upper + 1))
        Sx[i, i - 1] += coeff / 2.0
        Sy[i, i - 1] += coeff / (2.0j)
    Sy_real = np.imag(Sy)
    return Sx, Sy_real, Sz


def calculate_susceptibility(freq_thz, H, T, gamma_thz):
    gamma_uniform = 0.1
    gamma_array_7 = np.full(7, 0.1)
    if np.isscalar(gamma_thz):
        gamma_mode    = 'uniform'
        gamma_uniform = float(np.real(gamma_thz))
    elif hasattr(gamma_thz, '__len__'):
        gamma_array = np.atleast_1d(gamma_thz).real.astype(float)
        if len(gamma_array) == 7:
            gamma_mode    = '7gamma'
            gamma_array_7 = gamma_array
        else:
            gamma_mode    = 'uniform'
            gamma_uniform = float(gamma_array[0])
    else:
        gamma_mode    = 'uniform'
        gamma_uniform = float(np.real(gamma_thz))

    eigenvalues_K, eigenvectors = np.linalg.eigh(H)
    E_min       = np.min(eigenvalues_K)
    E_shifted_K = eigenvalues_K - E_min
    boltzmann_exp = np.clip(E_shifted_K / T, -700, 700)
    Z           = np.sum(np.exp(-boltzmann_exp))
    populations = np.exp(-boltzmann_exp) / Z
    E_shifted_J = E_shifted_K * kB

    Sx_zeeman, Sy_zeeman, _Sz_zeeman = construct_spin_operators()
    Sx_eb = eigenvectors.T.conj() @ Sx_zeeman @ eigenvectors
    Sy_eb = eigenvectors.T.conj() @ Sy_zeeman @ eigenvectors

    transition_xx   = np.abs(Sx_eb) ** 2
    transition_yy   = np.abs(Sy_eb) ** 2
    delta_E_matrix  = E_shifted_J[None, :] - E_shifted_J[:, None]
    omega_0_rad     = delta_E_matrix / hbar
    freq_0_matrix   = omega_0_rad * RAD_S_TO_THZ
    transition_perp = (transition_xx + transition_yy) / 2.0
    pop_diff_matrix = populations[:, None] - populations[None, :]
    strength_matrix = pop_diff_matrix * transition_perp
    non_diag_mask   = ~np.eye(8, dtype=bool)
    population_threshold = 1e-3
    occupied_mask   = populations[:, None] > population_threshold
    finite_mask     = (
        np.isfinite(freq_0_matrix) &
        np.isfinite(strength_matrix) &
        (np.abs(strength_matrix) > 1e-20) &
        occupied_mask &
        non_diag_mask
    )
    if not np.any(finite_mask):
        return np.zeros_like(freq_thz, dtype=complex)

    freq_0_valid   = freq_0_matrix[finite_mask]
    strength_valid = strength_matrix[finite_mask]
    n_indices, n_prime_indices = np.where(finite_mask)
    energy_order   = np.argsort(E_shifted_J)

    if gamma_mode == 'uniform':
        gamma_per_transition = np.full(len(freq_0_valid), gamma_uniform)
    elif gamma_mode == '7gamma':
        gamma_per_transition = np.zeros(len(freq_0_valid))
        for trans_idx in range(len(freq_0_valid)):
            n       = n_indices[trans_idx]
            n_prime = n_prime_indices[trans_idx]
            E_n       = E_shifted_J[n]
            E_n_prime = E_shifted_J[n_prime]
            lower_state = n if E_n <= E_n_prime else n_prime
            lower_state_energy_idx = np.where(energy_order == lower_state)[0][0]
            gamma_idx = min(lower_state_energy_idx, 6)
            gamma_per_transition[trans_idx] = gamma_array_7[gamma_idx]
    else:
        gamma_per_transition = np.full(len(freq_0_valid), 0.1)

    freq_diff  = freq_0_valid[None, :] - freq_thz[:, None]
    denominator = freq_diff - 1j * gamma_per_transition[None, :]
    safe_mask   = np.abs(denominator) > 1e-10
    denominator = np.where(safe_mask, denominator, 1e-10 + 1j * 1e-10)
    chi_array   = np.sum(strength_valid[None, :] / denominator, axis=1)
    return chi_array


def calculate_transmission(freq_thz, mu_r, d, eps_bg, t_min_ref=None, t_max_ref=None):
    """
    FP透過率を計算し正規化して返す。

    t_min_ref / t_max_ref を渡すと観測データの正規化スケールを共有する（推奨）。
    渡さない場合はモデル自身のポラリトン領域 min/max で自己正規化する（フォールバック）。
    """
    eps_bg    = max(eps_bg, 0.1)
    d         = max(d, 1e-6)
    omega     = freq_thz * THZ_TO_RAD_S
    mu_r_safe = np.where(np.isfinite(mu_r), mu_r, 1.0 + 0j)
    eps_mu    = eps_bg * mu_r_safe
    eps_mu    = np.where(eps_mu.real > 0, eps_mu, 0.1 + 1j * eps_mu.imag)
    n_complex = np.sqrt(eps_mu + 0j)
    impe      = np.sqrt(mu_r_safe / eps_bg + 0j)
    lambda_0  = np.where(omega > 1e-12, (2 * np.pi * c) / omega, np.inf)
    delta     = 2 * np.pi * n_complex * d / lambda_0
    delta     = np.clip(delta.real, -700, 700) + 1j * np.clip(delta.imag, -700, 700)
    numerator = 4 * impe
    exp_pos   = np.exp(-1j * delta)
    exp_neg   = np.exp(1j * delta)
    denom_fp  = (1 + impe) ** 2 * exp_pos - (1 - impe) ** 2 * exp_neg
    safe_mask = np.abs(denom_fp) > 1e-15
    t         = np.zeros_like(denom_fp, dtype=complex)
    t[safe_mask] = numerator[safe_mask] / denom_fp[safe_mask]
    transmission = np.abs(t) ** 2
    transmission = np.where(np.isfinite(transmission), transmission, 0.0)
    transmission = np.clip(transmission, 0, None)

    # ---- 正規化: 観測データの min/max を共有（スケール整合）----
    if (t_min_ref is not None and t_max_ref is not None
            and np.isfinite(t_min_ref) and np.isfinite(t_max_ref)
            and t_max_ref > t_min_ref):
        # 観測データ基準: 同一スケールで比較可能
        return (transmission - t_min_ref) / (t_max_ref - t_min_ref)
    else:
        # フォールバック: モデル自身のポラリトン領域で自己正規化
        sub_mask = freq_thz <= POLARITON_UPPER
        if np.any(sub_mask):
            t_min = np.min(transmission[sub_mask])
            t_max = np.max(transmission[sub_mask])
        else:
            t_min = np.min(transmission)
            t_max = np.max(transmission)
        if t_max > t_min and np.isfinite(t_max) and np.isfinite(t_min):
            return (transmission - t_min) / (t_max - t_min)
        else:
            return np.full_like(transmission, 0.5)


def calculate_transmission_for_params(freq, B, T, g, a, B4, B6, eps, gamma_array, model_form='H',
                                       t_min_ref=None, t_max_ref=None):
    H_ham   = get_hamiltonian(B, g, B4, B6)
    chi_raw = calculate_susceptibility(freq, H_ham, T, gamma_array)
    G0      = a * mu0 * N_SPIN * (g * muB) ** 2 / (2 * hbar) / THZ_TO_RAD_S
    chi     = G0 * chi_raw
    if model_form == 'H':
        mu_r = 1.0 + chi
    else:
        mu_r = 1.0 / (1.0 - chi)
    return calculate_transmission(freq, mu_r, d_fixed, eps, t_min_ref, t_max_ref)


# ============================================================================
# 【新規】FWHM 計算ユーティリティ
# ============================================================================
POLARITON_UPPER = 0.361505   # THz — ポラリトン領域上限
CAVITY_LOWER    = 0.45       # THz — 共振器領域下限


def compute_fwhm_from_spectrum(freq, trans, cavity_lower=CAVITY_LOWER):
    """
    透過スペクトルの共振器ピーク (f >= cavity_lower) から FWHM を計算する。

    Returns
    -------
    fwhm_list : list[float]
        各共振器ピークの FWHM [THz]。ピークが見つからない場合は空リスト。
    peak_freq_list : list[float]
        各共振器ピークの中心周波数 [THz]。
    """
    df   = freq[1] - freq[0] if len(freq) > 1 else 1.0
    mask = freq >= cavity_lower

    if not np.any(mask):
        return [], []

    freq_cav  = freq[mask]
    trans_cav = trans[mask]

    peaks, _props = find_peaks(trans_cav, prominence=0.03, width=2)
    if len(peaks) == 0:
        return [], []

    widths_samples, _, _, _ = peak_widths(trans_cav, peaks, rel_height=0.5)
    fwhm_list      = [float(w * df) for w in widths_samples]
    peak_freq_list = [float(freq_cav[p]) for p in peaks]

    return fwhm_list, peak_freq_list


def compute_cavity_peak_info(freq, trans, cavity_lower=CAVITY_LOWER):
    """
    共振器領域の最も顕著なピークの FWHM とピーク周波数を返す。
    Returns: (fwhm, peak_freq) — ピーク未検出時は (None, None)
    """
    mask = freq >= cavity_lower
    if not np.any(mask):
        return None, None

    freq_cav  = freq[mask]
    trans_cav = trans[mask]

    peaks, props = find_peaks(trans_cav, prominence=0.03, width=2)
    if len(peaks) == 0:
        return None, None

    best_idx = np.argmax(props['prominences'])
    peak_idx = peaks[best_idx]

    widths_samples, _, _, _ = peak_widths(trans_cav, [peak_idx], rel_height=0.5)
    df = freq[1] - freq[0] if len(freq) > 1 else 1.0

    return float(widths_samples[0] * df), float(freq_cav[peak_idx])


# ============================================================================
# ピーク検出・重み配列（test_fin_2a.py 準拠）
# ============================================================================
def detect_peaks_and_classify(freq, trans, polariton_upper=POLARITON_UPPER, cavity_lower=CAVITY_LOWER):
    peaks, properties = find_peaks(trans, prominence=0.05, width=3)
    if len(peaks) == 0:
        return [], []
    peak_freqs  = freq[peaks]
    peak_widths_arr = properties['widths'] * (freq[1] - freq[0])
    sort_idx    = np.argsort(peak_freqs)
    peak_freqs  = peak_freqs[sort_idx]
    peak_widths_arr = peak_widths_arr[sort_idx]
    polariton_regions = []
    cavity_regions    = []
    for pf, pw in zip(peak_freqs, peak_widths_arr):
        f_start = max(freq[0], pf - 1.5 * pw)
        f_end   = min(freq[-1], pf + 1.5 * pw)
        if pf <= polariton_upper:
            f_end_clipped = min(f_end, polariton_upper)
            if f_end_clipped > f_start:
                polariton_regions.append((f_start, f_end_clipped))
        elif pf >= cavity_lower:
            f_start_clipped = max(f_start, cavity_lower)
            if f_end > f_start_clipped:
                cavity_regions.append((f_start_clipped, f_end))
    return polariton_regions, cavity_regions


def create_weight_array(freq, _trans, polariton_regions, cavity_regions):
    weight_array = np.full_like(freq, 0.01)
    for f_start, f_end in polariton_regions:
        mask = (freq >= f_start) & (freq <= f_end)
        weight_array[mask] = 2.0
    for f_start, f_end in cavity_regions:
        mask = (freq >= f_start) & (freq <= f_end)
        weight_array[mask] = 1.0
    return weight_array


# ============================================================================
# 観測データ正規化 (事後予測スペクトルと同一スケーリング)
# ============================================================================
def normalize_obs_spectrum(freq, trans, polariton_upper=POLARITON_UPPER):
    """
    ポラリトン領域 (freq <= POLARITON_UPPER) を基準に 0-1 正規化する。
    共振器領域 (freq >= CAVITY_LOWER) は自然に 1 を超える。

    Returns
    -------
    trans_norm : ndarray  正規化済み透過率
    t_min      : float    正規化基準の最小値（モデル正規化に共有する）
    t_max      : float    正規化基準の最大値（モデル正規化に共有する）
    """
    sub_mask = freq <= polariton_upper
    if np.any(sub_mask):
        t_min = float(np.min(trans[sub_mask]))
        t_max = float(np.max(trans[sub_mask]))
    else:
        t_min = float(np.min(trans))
        t_max = float(np.max(trans))

    if t_max > t_min and np.isfinite(t_max) and np.isfinite(t_min):
        return (trans - t_min) / (t_max - t_min), t_min, t_max
    else:
        return trans.copy(), float(np.min(trans)), float(np.max(trans))


# ============================================================================
# データ読み込み
# ============================================================================
def load_all_datasets(target_data_list):
    print("\n--- データ読み込み ---")
    datasets  = []
    script_dir = pathlib.Path(__file__).parent
    candidate_dirs = [
        script_dir / 'bayesian_inputs',
        script_dir.parent / 'bayesian_inputs',
    ]
    base_dir = next((d for d in candidate_dirs if d.exists()), candidate_dirs[0])

    for config in target_data_list:
        excel_path = base_dir / config['file']
        if not excel_path.exists():
            print(f"❌ {excel_path} が見つかりません")
            continue
        try:
            df = pd.read_excel(excel_path, sheet_name=config['sheet'])
            if 'Frequency (THz)' not in df.columns or config['col'] not in df.columns:
                print(f"❌ {config['col']}: 必要な列が見つかりません")
                continue
            df_clean = df[['Frequency (THz)', config['col']]].dropna()
            freq  = df_clean['Frequency (THz)'].values.astype(np.float64)
            trans = df_clean[config['col']].values.astype(np.float64)

            # ポラリトン領域基準で正規化し、スケール(min/max)をdatasetに保存
            # → モデル計算時に同一スケールを共有するため
            trans_norm, obs_norm_min, obs_norm_max = normalize_obs_spectrum(freq, trans)
            trans = trans_norm

            polariton_regions, cavity_regions = detect_peaks_and_classify(freq, trans)
            weight_array = create_weight_array(freq, trans, polariton_regions, cavity_regions)

            # 全観測キャビティピーク検出（ピーク周波数・線幅対応）
            all_fwhms_obs, all_peak_freqs_obs = compute_fwhm_from_spectrum(freq, trans)
            # 後方互換: 最大振幅ピーク
            fwhm_obs, peak_freq_obs = compute_cavity_peak_info(freq, trans)

            label = f"{config['B']:.1f}T" if config['T'] == 1.5 else f"{config['T']:.0f}K"

            dataset = {
                'freq':             freq,
                'trans':            trans,
                'obs_norm_min':     obs_norm_min,   # 正規化基準（モデルと共有）
                'obs_norm_max':     obs_norm_max,   # 正規化基準（モデルと共有）
                'weight':           weight_array,
                'B':                config['B'],
                'T':                config['T'],
                'label':            label,
                'polariton_regions': polariton_regions,
                'cavity_regions':   cavity_regions,
                'sigma':            np.full_like(freq, 0.01),
                'fwhm_obs':         fwhm_obs,
                'peak_freq_obs':    peak_freq_obs,
                'all_fwhms_obs':    all_fwhms_obs,
                'all_peak_freqs_obs': all_peak_freqs_obs,
            }
            datasets.append(dataset)

            fwhm_str = f"{fwhm_obs*1000:.1f} GHz" if fwhm_obs is not None else "N/A"
            n_cav_peaks = len(all_peak_freqs_obs)
            peaks_str = ", ".join(f"{f:.3f}" for f in all_peak_freqs_obs) if n_cav_peaks > 0 else "none"
            print(f"✓ {label} (B={config['B']}T, T={config['T']}K): {len(freq)} points, "
                  f"cav FWHM={fwhm_str}, {n_cav_peaks} cavity peaks at [{peaks_str}] THz")
        except Exception as e:
            print(f"❌ {config['col']} 読み込みエラー: {e}")

    print(f"\n✅ 合計 {len(datasets)} データセット読み込み完了")
    n_with_fwhm = sum(1 for d in datasets if d['fwhm_obs'] is not None)
    print(f"   FWHM 観測値あり: {n_with_fwhm} / {len(datasets)} データセット")
    return datasets


# ============================================================================
# v8 結果読み込み（WLS v8 結果を初期参照値として利用）
# ============================================================================
def load_v8_optimized_params(model_form='H'):
    csv_path = pathlib.Path(__file__).parent / "wls_v8_results_20260306_160250" / f"parameters_{model_form}.csv"
    if not csv_path.exists():
        print(f"❌ {csv_path} が見つかりません")
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"❌ {csv_path} が空です")
            return None
        row = df.iloc[0]

        gamma_cols = [f'gamma_{i}' for i in range(1, 8)]
        missing = [c for c in ['g', 'a', 'B4', 'B6', 'eps_bg', *gamma_cols] if c not in df.columns]
        if missing:
            print(f"❌ {csv_path.name}: 必要列が不足しています -> {missing}")
            return None

        params = {
            'g': float(row['g']),
            'a': float(row['a']),
            'B4': float(row['B4']),
            'B6': float(row['B6']),
            'eps': float(row['eps_bg']),
            'gamma': np.array([float(row[c]) for c in gamma_cols], dtype=np.float64),
            'g_std': float(row['g_std']) if 'g_std' in df.columns else None,
            'a_std': float(row['a_std']) if 'a_std' in df.columns else None,
            'B4_std': float(row['B4_std']) if 'B4_std' in df.columns else None,
            'B6_std': float(row['B6_std']) if 'B6_std' in df.columns else None,
        }
        print(f"\n✓ {model_form}-form v8:")
        for k, v in params.items():
            if k != 'gamma' and k.endswith('_std') is False:
                print(f"  {k} = {v:.6f}")
        print(f"  gamma = {params['gamma']}")
        return params
    except Exception as e:
        print(f"❌ {model_form}-form 読み込みエラー: {e}")
        return None


def match_observed_and_predicted_peaks(obs_freqs, obs_fwhms, pred_freqs, pred_fwhms,
                                       max_distance=PEAK_MATCH_MAX_DISTANCE,
                                       fallback_peak_shift=0.05,
                                       fallback_fwhm_scale=2.0):
    """観測ピーク列に対して予測ピークを最近傍対応させる。

    `max_distance` を超える場合は、誤対応で尤度を甘くしないため
    ペナルティ用の擬似予測値を返す。
    """
    obs_freqs = np.asarray(obs_freqs, dtype=np.float64)
    obs_fwhms = np.asarray(obs_fwhms, dtype=np.float64)
    pred_freqs = np.asarray(pred_freqs, dtype=np.float64)
    pred_fwhms = np.asarray(pred_fwhms, dtype=np.float64)

    matched_fwhms = []
    matched_freqs = []
    details = []
    used_pred_peak_indices = set()

    for obs_freq, obs_fwhm in zip(obs_freqs, obs_fwhms):
        candidate_indices = [i for i in range(len(pred_freqs)) if i not in used_pred_peak_indices]
        if not candidate_indices:
            candidate_indices = list(range(len(pred_freqs)))

        matched = False
        if candidate_indices:
            distances = np.abs(pred_freqs[candidate_indices] - obs_freq)
            best_local_idx = int(np.argmin(distances))
            best_idx = candidate_indices[best_local_idx]
            best_distance = float(distances[best_local_idx])

            if best_distance <= max_distance:
                peak_freq_pred = float(pred_freqs[best_idx])
                if best_idx < len(pred_fwhms):
                    fwhm_pred = float(pred_fwhms[best_idx])
                else:
                    fwhm_pred = float(obs_fwhm)
                used_pred_peak_indices.add(best_idx)
                status = 'matched'
                matched = True

        if not matched:
            peak_freq_pred = float(obs_freq + fallback_peak_shift)
            fwhm_pred = float(obs_fwhm * fallback_fwhm_scale)
            best_distance = float(abs(peak_freq_pred - obs_freq))
            status = 'penalized'

        matched_freqs.append(peak_freq_pred)
        matched_fwhms.append(fwhm_pred)
        details.append({
            'obs_freq': float(obs_freq),
            'obs_fwhm': float(obs_fwhm),
            'pred_freq': peak_freq_pred,
            'pred_fwhm': fwhm_pred,
            'freq_error_ghz': abs(peak_freq_pred - obs_freq) * 1000.0,
            'fwhm_error_ghz': abs(fwhm_pred - obs_fwhm) * 1000.0,
            'distance_thz': best_distance,
            'status': status,
        })

    return np.asarray(matched_fwhms), np.asarray(matched_freqs), details


def summarize_peak_matching(obs_freqs, obs_fwhms, pred_freqs, pred_fwhms,
                            max_distance=PEAK_MATCH_MAX_DISTANCE):
    """ピーク対応の誤差を集計して返す。"""
    matched_fwhms, matched_freqs, details = match_observed_and_predicted_peaks(
        obs_freqs, obs_fwhms, pred_freqs, pred_fwhms, max_distance=max_distance
    )
    if not details:
        return {
            'n_obs_peaks': 0,
            'n_pred_peaks': int(len(pred_freqs)),
            'n_penalized': 0,
            'peak_mae_ghz': np.nan,
            'peak_max_err_ghz': np.nan,
            'fwhm_mae_ghz': np.nan,
        }

    freq_errors = np.asarray([item['freq_error_ghz'] for item in details], dtype=np.float64)
    fwhm_errors = np.asarray([item['fwhm_error_ghz'] for item in details], dtype=np.float64)
    n_penalized = sum(item['status'] != 'matched' for item in details)
    return {
        'n_obs_peaks': int(len(obs_freqs)),
        'n_pred_peaks': int(len(pred_freqs)),
        'n_penalized': int(n_penalized),
        'peak_mae_ghz': float(np.mean(freq_errors)),
        'peak_max_err_ghz': float(np.max(freq_errors)),
        'fwhm_mae_ghz': float(np.mean(fwhm_errors)),
        'matched_fwhms': matched_fwhms,
        'matched_freqs': matched_freqs,
        'details': details,
    }


def compute_dataset_fit_metrics(freq, trans_obs, trans_pred,
                                obs_peak_freqs=None, obs_fwhms=None,
                                pred_peak_freqs=None, pred_fwhms=None):
    """1 データセット分の再現度指標をまとめて返す。"""
    pol_mask = freq <= POLARITON_UPPER
    metrics = {
        'rmse_pol': float(np.sqrt(np.mean((trans_obs[pol_mask] - trans_pred[pol_mask]) ** 2))) if np.any(pol_mask) else np.nan,
        'rmse_full': float(np.sqrt(np.mean((trans_obs - trans_pred) ** 2))),
        'peak_mae_ghz': np.nan,
        'peak_max_err_ghz': np.nan,
        'fwhm_mae_ghz': np.nan,
        'n_obs_peaks': 0,
        'n_pred_peaks': 0,
        'n_penalized_peaks': 0,
    }
    if obs_peak_freqs is not None and obs_fwhms is not None and pred_peak_freqs is not None and pred_fwhms is not None:
        peak_summary = summarize_peak_matching(
            obs_peak_freqs, obs_fwhms, pred_peak_freqs, pred_fwhms
        )
        metrics.update({
            'peak_mae_ghz': peak_summary['peak_mae_ghz'],
            'peak_max_err_ghz': peak_summary['peak_max_err_ghz'],
            'fwhm_mae_ghz': peak_summary['fwhm_mae_ghz'],
            'n_obs_peaks': peak_summary['n_obs_peaks'],
            'n_pred_peaks': peak_summary['n_pred_peaks'],
            'n_penalized_peaks': peak_summary['n_penalized'],
        })
    return metrics


# ============================================================================
# 【新規】MixedOutputModelOp
# ポラリトン: 透過率ベクトル, 共振器: FWHM スカラーを出力　v10の最近傍mattchingに対応させる必要あり！
# ============================================================================
class MixedOutputModelOp(Op):
    """
    v9.0 領域別尤度用 Op。マルチキャビティピーク対応。
    perform() が trans_concat (全データセット連結) と
    fwhm_pred_vec, peak_freq_pred_vec (全キャビティピーク分) を出力する。
    """

    def __init__(self, datasets, model_form='H'):
        self.datasets    = datasets
        self.model_form  = model_form

        # 【v9.0】マルチピーク: 全データセットの全観測キャビティピークをフラット化
        self.peak_dataset_indices = []  # 各ピークの所属データセット index
        peak_obs_freqs_list = []
        peak_obs_fwhms_list = []
        for i, d in enumerate(datasets):
            all_freqs = d.get('all_peak_freqs_obs', [])
            all_fwhms = d.get('all_fwhms_obs', [])
            for f_val, w_val in zip(all_freqs, all_fwhms):
                self.peak_dataset_indices.append(i)
                peak_obs_freqs_list.append(f_val)
                peak_obs_fwhms_list.append(w_val)

        self.peak_obs_freqs_flat = np.array(peak_obs_freqs_list) if peak_obs_freqs_list else np.array([])
        self.peak_obs_fwhms_flat = np.array(peak_obs_fwhms_list) if peak_obs_fwhms_list else np.array([])
        self.n_total_peaks = len(self.peak_obs_freqs_flat)

        # データセットごとのピークインデックスをキャッシュ
        self._ds_peak_indices = {}
        for j, ds_idx in enumerate(self.peak_dataset_indices):
            if ds_idx not in self._ds_peak_indices:
                self._ds_peak_indices[ds_idx] = []
            self._ds_peak_indices[ds_idx].append(j)

        # 後方互換
        self.fwhm_indices = [i for i, d in enumerate(datasets) if d.get('fwhm_obs') is not None]
        self.fwhm_obs_vec = self.peak_obs_fwhms_flat
        self.peak_freq_obs_vec = self.peak_obs_freqs_flat

    def make_node(self, a_scale_scaled, gamma_vec_scaled, g_factor_scaled,
                  B4_scaled, B6_scaled, eps_bg_scaled):
        a_scale_scaled  = pt.as_tensor_variable(a_scale_scaled)
        gamma_vec_scaled = pt.as_tensor_variable(gamma_vec_scaled)
        g_factor_scaled = pt.as_tensor_variable(g_factor_scaled)
        B4_scaled       = pt.as_tensor_variable(B4_scaled)
        B6_scaled       = pt.as_tensor_variable(B6_scaled)
        eps_bg_scaled   = pt.as_tensor_variable(eps_bg_scaled)

        out_trans     = pt.dvector()
        out_fwhm      = pt.dvector()
        out_peak_freq = pt.dvector()

        return Apply(self,
                     [a_scale_scaled, gamma_vec_scaled, g_factor_scaled,
                      B4_scaled, B6_scaled, eps_bg_scaled],
                     [out_trans, out_fwhm, out_peak_freq])

    def perform(self, _node, inputs, output_storage):
        a_scale_scaled, gamma_vec_scaled, g_factor_scaled, B4_scaled, B6_scaled, eps_bg_scaled = inputs

        g_factor = float(g_factor_scaled) / SCALING_FACTORS['g']
        a_scale  = float(a_scale_scaled)  / SCALING_FACTORS['a']
        B4       = float(B4_scaled)       / SCALING_FACTORS['B4']
        B6       = float(B6_scaled)       / SCALING_FACTORS['B6']
        eps_bg   = float(eps_bg_scaled)   / SCALING_FACTORS['eps']

        gamma_array_scaled = np.atleast_1d(gamma_vec_scaled).astype(np.float64)
        if len(gamma_array_scaled) != 7:
            gamma_array_scaled = np.full(7, gamma_array_scaled[0])
        gamma_array = gamma_array_scaled / SCALING_FACTORS['gamma']

        all_trans_pred = []
        pred_peaks_cache = {}  # dataset_idx → (fwhm_list, freq_list)

        for idx, data in enumerate(self.datasets):
            freq = data['freq']
            B    = data['B']
            T    = data['T']

            H_ham   = get_hamiltonian(B, g_factor, B4, B6)
            chi_raw = calculate_susceptibility(freq, H_ham, T, gamma_array)
            G0      = a_scale * mu0 * N_SPIN * (g_factor * muB) ** 2 / (2 * hbar) / THZ_TO_RAD_S
            chi     = G0 * chi_raw

            if self.model_form == 'H':
                mu_r = 1.0 + chi
            else:
                denom = 1.0 - chi
                mu_r  = 1.0 / denom

            trans_pred = calculate_transmission(freq, mu_r, d_fixed, eps_bg,
                                               data.get('obs_norm_min'),
                                               data.get('obs_norm_max'))
            all_trans_pred.append(trans_pred)

            # 【v9.0】キャビティピーク検出をキャッシュ
            if idx in self._ds_peak_indices:
                pred_fwhms, pred_freqs = compute_fwhm_from_spectrum(freq, trans_pred)
                pred_peaks_cache[idx] = (pred_fwhms, pred_freqs)

        # 【v10.2】マルチピーク近傍マッチング
        # 観測ピークから遠すぎる予測ピークは誤対応とみなし、ペナルティ値を返す。
        fwhm_pred_list = []
        peak_freq_pred_list = []
        for ds_idx, peak_indices in self._ds_peak_indices.items():
            obs_freqs = self.peak_obs_freqs_flat[peak_indices]
            obs_fwhms = self.peak_obs_fwhms_flat[peak_indices]
            pred_fwhms, pred_freqs = pred_peaks_cache.get(ds_idx, ([], []))
            matched_fwhms, matched_freqs, _details = match_observed_and_predicted_peaks(
                obs_freqs, obs_fwhms, pred_freqs, pred_fwhms,
                max_distance=PEAK_MATCH_MAX_DISTANCE,
            )
            fwhm_pred_list.extend(matched_fwhms.tolist())
            peak_freq_pred_list.extend(matched_freqs.tolist())

        output_storage[0][0] = np.concatenate(all_trans_pred)
        output_storage[1][0] = np.array(fwhm_pred_list) if fwhm_pred_list else np.array([0.0])
        output_storage[2][0] = np.array(peak_freq_pred_list) if peak_freq_pred_list else np.array([0.5])


# ============================================================================
# モデル評価（test_fin_2a.py 準拠）
# ============================================================================
def compute_model_evaluation(trace, model_name='Model'):
    print(f"\n{'='*80}\nモデル評価: {model_name}\n{'='*80}")
    result: dict = {'model_name': model_name}
    try:
        summary    = az.summary(trace)
        mean_rhat  = summary['r_hat'].mean()    if 'r_hat'     in summary.columns else np.nan
        mean_ess   = summary['ess_bulk'].mean() if 'ess_bulk' in summary.columns else np.nan
        print(f"  平均 R-hat : {mean_rhat:.4f}" if not np.isnan(mean_rhat) else "  R-hat: N/A")
        print(f"  平均 ESS   : {mean_ess:.1f}"  if not np.isnan(mean_ess)  else "  ESS  : N/A")
        result['mean_rhat'] = float(mean_rhat) if not np.isnan(mean_rhat) else None
        result['mean_ess']  = float(mean_ess) if not np.isnan(mean_ess) else None
    except Exception as e:
        print(f"  ⚠️ サマリー計算エラー: {e}")
    try:
        posterior    = trace.posterior
        n_chains     = posterior.dims.get('chain', 1)
        n_draws      = posterior.dims.get('draw', 0)
        result['n_chains'] = n_chains
        result['n_draws'] = n_draws
        result['total_samples'] = n_chains * n_draws
    except Exception as e:
        print(f"  ⚠️ 事後分布統計エラー: {e}")
    return result


def compare_models(eval_H, eval_B):
    print(f"\n{'='*80}\nモデル比較\n{'='*80}")
    ess_H = eval_H.get('mean_ess', 0) or 0
    ess_B = eval_B.get('mean_ess', 0) or 0
    print(f"  H-form ESS: {ess_H:.1f}  /  B-form ESS: {ess_B:.1f}")
    if ess_H > ess_B * 1.1:
        winner = "H-form"
    elif ess_B > ess_H * 1.1:
        winner = "B-form"
    else:
        winner = "引き分け"
    print(f"  🏆 推奨モデル: {winner}")
    return {'ess_H': ess_H, 'ess_B': ess_B, 'winner': winner, 'method': 'ESS comparison'}


def _collect_lml_histories(raw_values):
    """SMC の log_marginal_likelihood からチェーンごとの履歴配列を抽出する。"""
    histories = []

    def _recurse(obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                for item in obj.flat:
                    _recurse(item)
            elif obj.ndim <= 1:
                histories.append(np.asarray(obj, dtype=np.float64).reshape(-1))
            else:
                for row in obj:
                    histories.append(np.asarray(row, dtype=np.float64).reshape(-1))
        elif isinstance(obj, (list, tuple)):
            if obj and all(not isinstance(item, (list, tuple, np.ndarray)) for item in obj):
                histories.append(np.asarray(obj, dtype=np.float64).reshape(-1))
            else:
                for item in obj:
                    _recurse(item)
        else:
            try:
                histories.append(np.asarray([float(obj)], dtype=np.float64))
            except Exception:
                pass

    _recurse(raw_values)
    return histories


def _extract_lml_scalars(raw_values):
    """各チェーンの最終的な有限 log_marginal_likelihood だけを抜き出す。"""
    finals = []
    for history in _collect_lml_histories(raw_values):
        finite_vals = history[np.isfinite(history)]
        if finite_vals.size:
            finals.append(float(finite_vals[-1]))
    return np.array(finals, dtype=np.float64)


def compute_bayes_factor_smc(trace_H, trace_B):
    print(f"\n{'='*80}\nベイズファクター計算\n{'='*80}")
    result = {}
    try:
        has_H = hasattr(trace_H, 'sample_stats') and 'log_marginal_likelihood' in trace_H.sample_stats
        has_B = hasattr(trace_B, 'sample_stats') and 'log_marginal_likelihood' in trace_B.sample_stats
        if has_H and has_B:
            lml_H_vals = _extract_lml_scalars(trace_H.sample_stats['log_marginal_likelihood'].values)
            lml_B_vals = _extract_lml_scalars(trace_B.sample_stats['log_marginal_likelihood'].values)
            lml_H = float(np.nanmean(lml_H_vals))
            lml_B = float(np.nanmean(lml_B_vals))
            lml_H_std = float(np.nanstd(lml_H_vals))
            lml_B_std = float(np.nanstd(lml_B_vals))
            log_BF = lml_H - lml_B
            log_BF_se = np.sqrt(lml_H_std ** 2 + lml_B_std ** 2)
            log10_BF  = log_BF / np.log(10)
            abs_log_BF = abs(log_BF)
            strength = ("ほぼ証拠なし" if abs_log_BF < 1.15 else
                        "弱い証拠"     if abs_log_BF < 2.3  else
                        "中程度の証拠"  if abs_log_BF < 4.6  else
                        "強い証拠")
            winner = "H-form" if log_BF > 0 else ("B-form" if log_BF < 0 else "引き分け")
            print(f"  H-form log(ML): {lml_H:.2f} ± {lml_H_std:.2f}")
            print(f"  B-form log(ML): {lml_B:.2f} ± {lml_B_std:.2f}")
            print(f"  log(BF_{{H/B}}): {log_BF:.2f} ± {log_BF_se:.2f}  → {strength}")
            print(f"  🏆 推奨モデル: {winner}")
            result = {'log_BF': log_BF, 'log10_BF': log10_BF,
                      'interpretation': strength, 'winner': winner,
                      'method': 'SMC marginal likelihood'}
        else:
            print("  ⚠️ SMC 周辺尤度が保存されていません")
            result = {'winner': 'N/A', 'method': 'unavailable'}
    except Exception as e:
        print(f"❌ ベイズファクター計算エラー: {e}")
        result = {'winner': 'N/A', 'error': str(e)}
    return result


# ============================================================================
# プロット関数（test_fin_2a.py から継承）
# ============================================================================
def plot_posterior_predictive_spectra(trace, datasets, model_form='H', save_dir=None, n_samples=300):
    print(f"\n{'='*80}\n事後予測スペクトルプロット ({model_form}-form)\n{'='*80}")
    posterior    = trace.posterior
    n_chains_val = posterior.dims['chain']
    n_draws_val  = posterior.dims['draw']
    total_samples = n_chains_val * n_draws_val
    if total_samples > n_samples:
        sample_indices = np.random.choice(total_samples, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
        n_samples = total_samples

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig.suptitle(f'Posterior Predictive Spectra ({model_form}-form) — v9.0 Multi Cavity Peak',
                 fontsize=12, y=0.995)
    axes = axes.flatten()
    metrics_rows = []

    for idx, data in enumerate(datasets):
        ax        = axes[idx]
        freq      = data['freq']
        trans_obs = data['trans']
        B, T      = data['B'], data['T']
        label     = data['label']
        fwhm_obs  = data['fwhm_obs']

        trans_samples = np.zeros((n_samples, len(freq)))
        for i, sample_idx in enumerate(sample_indices):
            ci = sample_idx // n_draws_val
            di = sample_idx % n_draws_val
            g   = float(posterior['g_factor_scaled'].values[ci, di]) / SCALING_FACTORS['g']
            a   = float(posterior['a_scale_scaled'].values[ci, di])  / SCALING_FACTORS['a']
            B4  = float(posterior['B4_scaled'].values[ci, di])       / SCALING_FACTORS['B4']
            B6  = float(posterior['B6_scaled'].values[ci, di])       / SCALING_FACTORS['B6']
            eps = float(posterior['eps_bg_scaled'].values[ci, di])   / SCALING_FACTORS['eps']
            gamma_array = np.array([
                float(posterior[f'gamma_{j+1}_scaled'].values[ci, di]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            trans_samples[i] = calculate_transmission_for_params(
                freq, B, T, g, a, B4, B6, eps, gamma_array, model_form,
                t_min_ref=data.get('obs_norm_min'),
                t_max_ref=data.get('obs_norm_max'))

        trans_median = np.median(trans_samples, axis=0)
        trans_hdi    = az.hdi(trans_samples, hdi_prob=0.94)

        pred_fwhms, pred_peak_freqs = compute_fwhm_from_spectrum(freq, trans_median)
        metrics = compute_dataset_fit_metrics(
            freq, trans_obs, trans_median,
            obs_peak_freqs=data.get('all_peak_freqs_obs'),
            obs_fwhms=data.get('all_fwhms_obs'),
            pred_peak_freqs=pred_peak_freqs,
            pred_fwhms=pred_fwhms,
        )
        metrics_rows.append({
            'label': label,
            'B': B,
            'T': T,
            **metrics,
        })

        # 領域ハイライト
        for f_s, f_e in data['polariton_regions']:
            ax.axvspan(f_s, f_e, alpha=0.12, color='orange', label='Polariton' if f_s == data['polariton_regions'][0][0] else None)
        for f_s, f_e in data['cavity_regions']:
            ax.axvspan(f_s, f_e, alpha=0.12, color='green',  label='Cavity'    if f_s == data['cavity_regions'][0][0]    else None)

        ax.plot(freq, trans_obs,    'ko',  markersize=2.5, alpha=0.6, label='Obs')
        ax.plot(freq, trans_median, 'r-',  lw=2,           label='Median')
        ax.fill_between(freq, trans_hdi[:, 0], trans_hdi[:, 1], color='red', alpha=0.2, label='94% HDI')

        peak_str = (
            f" peak_MAE={metrics['peak_mae_ghz']:.1f}GHz"
            if not np.isnan(metrics['peak_mae_ghz']) else ""
        )
        fwhm_str = (
            f"FWHM_MAE={metrics['fwhm_mae_ghz']:.1f} GHz "
            f"(penalty={metrics['n_penalized_peaks']})"
            if not np.isnan(metrics['fwhm_mae_ghz']) else ""
        )
        ax.set_title(
            f"{label}  RMSE_pol={metrics['rmse_pol']:.4f} (full={metrics['rmse_full']:.4f}){peak_str}\n{fwhm_str}",
            fontsize=8, fontweight='bold')
        ax.set_xlabel('Frequency (THz)', fontsize=9)
        ax.set_ylabel('Transmittance',   fontsize=9)
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([freq.min(), freq.max()])
        ax.set_ylim([0, max(1.05, np.nanmax(trans_median) * 1.05)])

    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'posterior_predictive_spectra_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
        pd.DataFrame(metrics_rows).to_csv(save_dir / f'fit_metrics_{model_form}.csv', index=False)
        print(f"  ✓ fit_metrics_{model_form}.csv saved")
    plt.close()


def plot_posterior_predictive_spectra_combined(trace_H, trace_B, datasets, save_dir=None, n_samples=200):
    """H/B モデルの事後予測スペクトルを同一図に重ね描きする。"""
    print(f"\n{'='*80}\n事後予測スペクトル重ね描き (H vs B)\n{'='*80}")

    posterior_H = trace_H.posterior
    posterior_B = trace_B.posterior

    n_chains_H = posterior_H.dims['chain']
    n_draws_H = posterior_H.dims['draw']
    total_samples_H = n_chains_H * n_draws_H
    if total_samples_H > n_samples:
        sample_indices_H = np.random.choice(total_samples_H, size=n_samples, replace=False)
    else:
        sample_indices_H = np.arange(total_samples_H)
        n_samples = total_samples_H

    n_chains_B = posterior_B.dims['chain']
    n_draws_B = posterior_B.dims['draw']
    total_samples_B = n_chains_B * n_draws_B
    if total_samples_B > n_samples:
        sample_indices_B = np.random.choice(total_samples_B, size=n_samples, replace=False)
    else:
        sample_indices_B = np.arange(total_samples_B)

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig.suptitle('Posterior Predictive Spectra (H/B Overlay) — H:red, B:blue', fontsize=12, y=0.995)
    axes = axes.flatten()

    for idx, data in enumerate(datasets):
        ax = axes[idx]
        freq = data['freq']
        trans_obs = data['trans']
        B = data['B']
        T = data['T']
        label = data['label']

        trans_samples_H = np.zeros((len(sample_indices_H), len(freq)))
        for i, sample_idx in enumerate(sample_indices_H):
            ci = sample_idx // n_draws_H
            di = sample_idx % n_draws_H
            g = float(posterior_H['g_factor_scaled'].values[ci, di]) / SCALING_FACTORS['g']
            a = float(posterior_H['a_scale_scaled'].values[ci, di]) / SCALING_FACTORS['a']
            B4 = float(posterior_H['B4_scaled'].values[ci, di]) / SCALING_FACTORS['B4']
            B6 = float(posterior_H['B6_scaled'].values[ci, di]) / SCALING_FACTORS['B6']
            eps = float(posterior_H['eps_bg_scaled'].values[ci, di]) / SCALING_FACTORS['eps']
            gamma_array = np.array([
                float(posterior_H[f'gamma_{j+1}_scaled'].values[ci, di]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            trans_samples_H[i] = calculate_transmission_for_params(
                freq, B, T, g, a, B4, B6, eps, gamma_array, 'H',
                t_min_ref=data.get('obs_norm_min'),
                t_max_ref=data.get('obs_norm_max'))

        trans_samples_B = np.zeros((len(sample_indices_B), len(freq)))
        for i, sample_idx in enumerate(sample_indices_B):
            ci = sample_idx // n_draws_B
            di = sample_idx % n_draws_B
            g = float(posterior_B['g_factor_scaled'].values[ci, di]) / SCALING_FACTORS['g']
            a = float(posterior_B['a_scale_scaled'].values[ci, di]) / SCALING_FACTORS['a']
            B4 = float(posterior_B['B4_scaled'].values[ci, di]) / SCALING_FACTORS['B4']
            B6 = float(posterior_B['B6_scaled'].values[ci, di]) / SCALING_FACTORS['B6']
            eps = float(posterior_B['eps_bg_scaled'].values[ci, di]) / SCALING_FACTORS['eps']
            gamma_array = np.array([
                float(posterior_B[f'gamma_{j+1}_scaled'].values[ci, di]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])
            trans_samples_B[i] = calculate_transmission_for_params(
                freq, B, T, g, a, B4, B6, eps, gamma_array, 'B',
                t_min_ref=data.get('obs_norm_min'),
                t_max_ref=data.get('obs_norm_max'))

        median_H = np.median(trans_samples_H, axis=0)
        hdi_H = az.hdi(trans_samples_H, hdi_prob=0.94)
        median_B = np.median(trans_samples_B, axis=0)
        hdi_B = az.hdi(trans_samples_B, hdi_prob=0.94)

        # 領域ハイライト（個別プロットと統一）
        for f_s, f_e in data['polariton_regions']:
            ax.axvspan(f_s, f_e, alpha=0.12, color='orange',
                       label='Polariton' if f_s == data['polariton_regions'][0][0] else None)
        for f_s, f_e in data['cavity_regions']:
            ax.axvspan(f_s, f_e, alpha=0.12, color='green',
                       label='Cavity' if f_s == data['cavity_regions'][0][0] else None)

        ax.plot(freq, trans_obs, 'ko', markersize=2.3, alpha=0.55, label='Obs')
        ax.plot(freq, median_H, 'r-', lw=2.0, label='H median')
        ax.fill_between(freq, hdi_H[:, 0], hdi_H[:, 1], color='red', alpha=0.15, label='H 94% HDI')
        ax.plot(freq, median_B, 'b-', lw=2.0, label='B median')
        ax.fill_between(freq, hdi_B[:, 0], hdi_B[:, 1], color='blue', alpha=0.12, label='B 94% HDI')

        pred_fwhms_H, pred_peak_freqs_H = compute_fwhm_from_spectrum(freq, median_H)
        pred_fwhms_B, pred_peak_freqs_B = compute_fwhm_from_spectrum(freq, median_B)
        metrics_H = compute_dataset_fit_metrics(
            freq, trans_obs, median_H,
            obs_peak_freqs=data.get('all_peak_freqs_obs'),
            obs_fwhms=data.get('all_fwhms_obs'),
            pred_peak_freqs=pred_peak_freqs_H,
            pred_fwhms=pred_fwhms_H,
        )
        metrics_B = compute_dataset_fit_metrics(
            freq, trans_obs, median_B,
            obs_peak_freqs=data.get('all_peak_freqs_obs'),
            obs_fwhms=data.get('all_fwhms_obs'),
            pred_peak_freqs=pred_peak_freqs_B,
            pred_fwhms=pred_fwhms_B,
        )

        ax.set_title(
            f"{label}  RMSE_pol: H={metrics_H['rmse_pol']:.4f} B={metrics_B['rmse_pol']:.4f}"
            f"  (full: H={metrics_H['rmse_full']:.4f} B={metrics_B['rmse_full']:.4f})\n"
            f"peak_MAE[GHz]: H={metrics_H['peak_mae_ghz']:.1f} B={metrics_B['peak_mae_ghz']:.1f}"
            f"  FWHM_MAE[GHz]: H={metrics_H['fwhm_mae_ghz']:.1f} B={metrics_B['fwhm_mae_ghz']:.1f}",
            fontsize=7.5, fontweight='bold'
        )
        ax.set_xlabel('Frequency (THz)', fontsize=9)
        ax.set_ylabel('Transmittance', fontsize=9)
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([freq.min(), freq.max()])
        ax.set_ylim([0, max(1.05, max(np.nanmax(median_H), np.nanmax(median_B)) * 1.05)])

    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_dir:
        path = save_dir / 'posterior_predictive_spectra_HB.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


def plot_posterior_distributions(trace, model_form='H', save_dir=None):
    print(f"\n事後分布プロット作成 ({model_form}-form)...")
    posterior = trace.posterior
    param_info = [
        ('g_factor_scaled', SCALING_FACTORS['g'],   'g-factor',  ''),
        ('a_scale_scaled',  SCALING_FACTORS['a'],   'a (coupling)', ''),
        ('B4_scaled',       SCALING_FACTORS['B4'],  'B₄',         'mK'),
        ('B6_scaled',       SCALING_FACTORS['B6'],  'B₆',         'mK'),
        ('eps_bg_scaled',   SCALING_FACTORS['eps'], 'ε_bg',      ''),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Posterior Distributions ({model_form}-form) — v8.0', fontsize=14)
    axes = axes.flatten()
    for i, (var, scale, label, unit) in enumerate(param_info):
        ax = axes[i]
        samples = posterior[var].values.flatten() / scale
        if unit == 'mK':
            samples *= 1000
            xlabel = f'{label} ({unit})'
        else:
            xlabel = label
        ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', lw=0.5)
        mean_val   = np.mean(samples)
        median_val = np.median(samples)
        hdi        = az.hdi(samples, hdi_prob=0.94)
        ax.axvline(mean_val,   color='red',    linestyle='--', lw=1.5, label=f'Mean: {mean_val:.3g}')
        ax.axvline(median_val, color='orange', linestyle='-.', lw=1.5, label=f'Med: {median_val:.3g}')
        ax.axvspan(hdi[0], hdi[1], alpha=0.2, color='green', label='94% HDI')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    for i in range(7):
        ax = axes[5 + i]
        var_name = f'gamma_{i+1}_scaled'
        samples  = posterior[var_name].values.flatten() / SCALING_FACTORS['gamma']
        ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', lw=0.5)
        mean_val   = np.mean(samples)
        median_val = np.median(samples)
        hdi        = az.hdi(samples, hdi_prob=0.94)
        ax.axvline(mean_val,   color='red',    linestyle='--', lw=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='-.', lw=1.5, label=f'Med: {median_val:.2f}')
        ax.axvspan(hdi[0], hdi[1], alpha=0.2, color='green', label='94% HDI')
        ax.set_xlabel(f'γ_{i+1} (THz)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'γ_{i+1}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'posterior_distributions_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


# ============================================================================
# 【新規】エネルギー準位図 (点プロット, 絶対エネルギー)
# ============================================================================
def plot_energy_levels(trace, datasets, model_form='H', save_dir=None, n_samples=300):
    """
    事後分布からサンプリングしたパラメータでエネルギー固有値を計算し、
    磁場 B に対するエネルギー準位図を点でプロットする（絶対エネルギー）。
    """
    print(f"\nエネルギー準位プロット作成 ({model_form}-form)...")
    posterior = trace.posterior
    n_chains_val = posterior.dims['chain']
    n_draws_val  = posterior.dims['draw']
    total_samples = n_chains_val * n_draws_val
    if total_samples > n_samples:
        sample_indices = np.random.choice(total_samples, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
        n_samples = total_samples

    dataset_Bs = np.array(sorted(set(d['B'] for d in datasets)), dtype=float)
    if dataset_Bs.size == 0:
        B_fields = np.linspace(0.0, 10.0, 20)
    else:
        # datasets に含まれる磁場値をそのまま使用（補間しない）
        B_fields = dataset_Bs.copy()
    n_states = int(2 * S_VALUE + 1)

    # shape (n_samples, n_B, n_states)
    all_eigenvals = np.zeros((n_samples, len(B_fields), n_states))

    for i, sample_idx in enumerate(sample_indices):
        ci = sample_idx // n_draws_val # サンプルインデックスからチェーンとドローの位置を計算
        di = sample_idx % n_draws_val
        g  = float(posterior['g_factor_scaled'].values[ci, di]) / SCALING_FACTORS['g']
        B4 = float(posterior['B4_scaled'].values[ci, di])       / SCALING_FACTORS['B4']
        B6 = float(posterior['B6_scaled'].values[ci, di])       / SCALING_FACTORS['B6']
        for j, B_val in enumerate(B_fields):
            H_ham = get_hamiltonian(B_val, g, B4, B6)
            evals = np.sort(np.linalg.eigh(H_ham)[0])
            all_eigenvals[i, j] = evals # i番目のサンプルの j 番目の磁場でのエネルギー固有値を保存

    median_evals = np.median(all_eigenvals, axis=0)

    _fig, ax = plt.subplots(figsize=(10, 7))
    colors = [plt.colormaps['tab10'](i / n_states) for i in range(n_states)]
    for k in range(n_states):
        ax.scatter(B_fields, median_evals[:, k], c=[colors[k]], s=15, zorder=3,
                   label=f'|{k}⟩', edgecolors='none')

    ax.set_xlabel('Magnetic Field B (T)', fontsize=12)
    ax.set_ylabel('Energy E (K)', fontsize=12)
    ax.set_title(f'Energy Level Diagram ({model_form}-form) — Absolute\n'
                 f'S={S_VALUE}, {n_samples} posterior samples', fontsize=12)
    ax.legend(fontsize=7, ncol=4, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_dir:
        path = save_dir / f'energy_levels_{model_form}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path.name} saved")
    plt.close()


# ============================================================================
# 【新規】磁気感受率プロット (実部・虚部別, 94% HDI)
# ============================================================================
def plot_susceptibility(trace, datasets, model_form='H', save_dir=None, n_samples=200):
    """
    事後分布からサンプリングしたパラメータで磁気感受率 χ+(ω) を計算し、
    Re(χ+) と Im(χ+) を別々の図にプロットする。94% HDI バンド付き。
    """
    print(f"\n磁気感受率プロット作成 ({model_form}-form)...")
    posterior = trace.posterior
    n_chains_val = posterior.dims['chain']
    n_draws_val  = posterior.dims['draw']
    total_samples = n_chains_val * n_draws_val
    if total_samples > n_samples:
        sample_indices = np.random.choice(total_samples, size=n_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
        n_samples = total_samples

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2

    fig_re, axes_re = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig_re.suptitle(f'Magnetic Susceptibility Re(χ₊) ({model_form}-form) — 94% HDI',
                    fontsize=12, y=0.995)
    axes_re = axes_re.flatten()

    fig_im, axes_im = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    fig_im.suptitle(f'Magnetic Susceptibility Im(χ₊) ({model_form}-form) — 94% HDI',
                    fontsize=12, y=0.995)
    axes_im = axes_im.flatten()

    for idx, data in enumerate(datasets):
        freq  = data['freq']
        B_ext = data['B']
        T_val = data['T']
        label = data['label']

        chi_re_samples = np.zeros((n_samples, len(freq)))
        chi_im_samples = np.zeros((n_samples, len(freq)))

        for i, sample_idx in enumerate(sample_indices):
            ci = sample_idx // n_draws_val
            di = sample_idx % n_draws_val
            g   = float(posterior['g_factor_scaled'].values[ci, di]) / SCALING_FACTORS['g']
            a   = float(posterior['a_scale_scaled'].values[ci, di])  / SCALING_FACTORS['a']
            B4  = float(posterior['B4_scaled'].values[ci, di])       / SCALING_FACTORS['B4']
            B6  = float(posterior['B6_scaled'].values[ci, di])       / SCALING_FACTORS['B6']
            gamma_array = np.array([
                float(posterior[f'gamma_{j+1}_scaled'].values[ci, di]) / SCALING_FACTORS['gamma']
                for j in range(7)
            ])

            H_ham   = get_hamiltonian(B_ext, g, B4, B6)
            chi_raw = calculate_susceptibility(freq, H_ham, T_val, gamma_array)
            G0      = a * mu0 * N_SPIN * (g * muB) ** 2 / (2 * hbar) / THZ_TO_RAD_S
            chi     = G0 * chi_raw

            chi_re_samples[i] = np.real(chi)
            chi_im_samples[i] = np.imag(chi)

        # Re(χ+)
        ax_re = axes_re[idx]
        re_median = np.median(chi_re_samples, axis=0)
        re_hdi    = az.hdi(chi_re_samples, hdi_prob=0.94)
        ax_re.plot(freq, re_median, 'b-', lw=1.5, label='Median')
        ax_re.fill_between(freq, re_hdi[:, 0], re_hdi[:, 1],
                           color='blue', alpha=0.2, label='94% HDI')
        ax_re.set_title(f'{label} (B={B_ext}T, T={T_val}K)', fontsize=9, fontweight='bold')
        ax_re.set_xlabel('Frequency (THz)', fontsize=8)
        ax_re.set_ylabel('Re(χ₊)', fontsize=8)
        ax_re.legend(fontsize=6, loc='best')
        ax_re.grid(alpha=0.3)
        ax_re.axhline(0, color='gray', lw=0.5)

        # Im(χ+)
        ax_im = axes_im[idx]
        im_median = np.median(chi_im_samples, axis=0)
        im_hdi    = az.hdi(chi_im_samples, hdi_prob=0.94)
        ax_im.plot(freq, im_median, 'r-', lw=1.5, label='Median')
        ax_im.fill_between(freq, im_hdi[:, 0], im_hdi[:, 1],
                           color='red', alpha=0.2, label='94% HDI')
        ax_im.set_title(f'{label} (B={B_ext}T, T={T_val}K)', fontsize=9, fontweight='bold')
        ax_im.set_xlabel('Frequency (THz)', fontsize=8)
        ax_im.set_ylabel('Im(χ₊)', fontsize=8)
        ax_im.legend(fontsize=6, loc='best')
        ax_im.grid(alpha=0.3)
        ax_im.axhline(0, color='gray', lw=0.5)

    for idx in range(n_datasets, len(axes_re)):
        axes_re[idx].axis('off')
        axes_im[idx].axis('off')

    fig_re.tight_layout()
    fig_im.tight_layout()
    if save_dir:
        path_re = save_dir / f'susceptibility_real_{model_form}.png'
        fig_re.savefig(path_re, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path_re.name} saved")
        path_im = save_dir / f'susceptibility_imag_{model_form}.png'
        fig_im.savefig(path_im, dpi=300, bbox_inches='tight')
        print(f"  ✓ {path_im.name} saved")
    plt.close(fig_re)
    plt.close(fig_im)


# ============================================================================
# ベイズモデル構築ヘルパー（H / B 共通）
# ============================================================================
def build_pymc_model(datasets, model_form, v8_params_H, v8_params_B):
    """
    v8.0 複合尤度モデルを構築して返す。

    尤度構造:
    - ポラリトン領域: StudentT(ν=4) で trans_pred vs trans_obs
    - 共振器領域   : StudentT(ν=4) で fwhm_pred  vs fwhm_obs
    - 背景領域     : StudentT(ν=4) で trans_pred vs trans_obs (weight=0.01)
    """
    v8_params_model = v8_params_H if model_form == 'H' else v8_params_B

    # ------------------------------------------------------------------
    # ε_bg の FSR 由来事前推定
    # FP共振条件: FSR = c / (2 * sqrt(ε) * d)  →  ε = (c / (2*d*FSR))²
    # キャビティピーク間隔からデータセットごとに ε を逆算し中央値を prior に使用
    # （ポラリトン尤度が ε を過大推定する degeneracy を抑制する）
    # ------------------------------------------------------------------
    eps_fsr_list = []
    fsr_obs_list = []
    for d_set in datasets:
        peaks = d_set.get('all_peak_freqs_obs', [])
        if len(peaks) >= 2:
            for k in range(len(peaks) - 1):
                fsr_thz = peaks[k + 1] - peaks[k]   # THz
                fsr_obs_list.append(fsr_thz)
                n_eff = c / (2.0 * d_fixed * fsr_thz * 1e12)
                eps_fsr_list.append(n_eff ** 2)

    if eps_fsr_list:
        eps_fsr_median = float(np.median(eps_fsr_list))
        fsr_obs_median = float(np.median(fsr_obs_list))
        fsr_obs_std    = float(np.std(fsr_obs_list)) if len(fsr_obs_list) > 1 else 0.003
        # FSR 測定不確かさ: 観測ばらつき or 最低 3 GHz
        sigma_fsr = max(fsr_obs_std, 0.003)
        print(f"  [FSR prior] eps_fsr={eps_fsr_median:.3f}  "
              f"fsr_obs={fsr_obs_median*1000:.1f}±{sigma_fsr*1000:.1f} GHz")
    else:
        eps_fsr_median = (v8_params_H['eps'] + v8_params_B['eps']) / 2
        fsr_obs_median = c / (2.0 * d_fixed * np.sqrt(eps_fsr_median) * 1e12)
        sigma_fsr      = 0.005
        print(f"  [FSR prior] FSR データなし → v8 eps_avg={eps_fsr_median:.3f} を使用")

    eps_v8_avg = eps_fsr_median   # FSR 由来推定値を prior 中心に使用

    # Op の生成
    model_op = MixedOutputModelOp(datasets, model_form)

    # 観測データ
    trans_obs_list  = [d['trans']  for d in datasets]
    weight_list     = [d['weight'] for d in datasets]

    # 領域マスク（ポラリトン / 背景）
    polariton_masks = []
    bg_masks        = []
    for d in datasets:
        weight = d['weight']
        polariton_masks.append(weight == 2.0)
        bg_masks.append(weight == 0.01)

    # 観測値の連結
    trans_obs_concat = np.concatenate(trans_obs_list)
    weight_concat    = np.concatenate(weight_list)
    polariton_mask_concat = np.concatenate(polariton_masks)
    bg_mask_concat        = np.concatenate(bg_masks)

    # 有効 σ (weight が大きいほど σ が小さい = 精度が高い)
    sigma_eff = np.clip(
        SIGMA_SPECTRUM_BASE / np.sqrt(weight_concat),
        SIGMA_SPECTRUM_MIN,
        SIGMA_SPECTRUM_MAX,
    )

    # FWHM 観測値 + ピーク位置観測値
    fwhm_obs_vec = model_op.fwhm_obs_vec  # shape (n_fwhm,)
    peak_freq_obs_vec = model_op.peak_freq_obs_vec  # shape (n_fwhm,)
    sigma_fwhm   = np.full_like(fwhm_obs_vec, SIGMA_FWHM) if len(fwhm_obs_vec) > 0 else np.array([SIGMA_FWHM])
    sigma_peak_freq = np.full_like(peak_freq_obs_vec, SIGMA_PEAK_FREQ) if len(peak_freq_obs_vec) > 0 else np.array([SIGMA_PEAK_FREQ])

    model = pm.Model()
    with model:
        # ------------------------------------------
        # 1. g 因子: v8 中心 TruncNormal
        # ------------------------------------------
        g_sigma = max(0.03, 3.0 * (v8_params_model.get('g_std') or 0.0))
        g_factor_scaled = pm.TruncatedNormal(
            'g_factor_scaled',
            mu=v8_params_model['g'] * SCALING_FACTORS['g'],
            sigma=g_sigma * SCALING_FACTORS['g'],
            lower=1.5 * SCALING_FACTORS['g'],
            upper=2.8 * SCALING_FACTORS['g'])

        # ------------------------------------------
        # 2. a: v8 中心 TruncNormal + clip
        # ------------------------------------------
        a_raw_name = f'a_raw_{model_form}'
        a_sigma = max(0.4, 3.0 * (v8_params_model.get('a_std') or 0.0))
        a_raw = pm.TruncatedNormal(
            a_raw_name,
            mu=v8_params_model['a'],
            sigma=a_sigma,
            lower=0.1,
            upper=12.0,
        )
        a_scale_scaled = pm.Deterministic('a_scale_scaled',
            pt.clip(a_raw, 0.1, 12.0) * SCALING_FACTORS['a'])

        # ------------------------------------------
        # 3. B₄: v8 中心 Normal (負値許容・clip なし → 真の事後分布)
        # ------------------------------------------
        B4_raw_name = f'B4_raw_{model_form}'
        B4_sigma = max(0.05, 3.0 * (v8_params_model.get('B4_std') or 0.0))
        B4_raw = pm.Normal(B4_raw_name, mu=v8_params_model['B4'], sigma=B4_sigma)
        B4_scaled = pm.Deterministic('B4_scaled',
            B4_raw * SCALING_FACTORS['B4'])

        # ------------------------------------------
        # 4. B₆: FIX_B6_ZERO モードで分岐
        # ------------------------------------------
        if FIX_B6_ZERO:
            # 固定モード: B₆ = 0 (自由度削減・収束安定化)
            B6_scaled = pm.Deterministic('B6_scaled',
                pt.as_tensor_variable(0.0))
        else:
            # 推定モード: Normal 事前分布 (clip なし → 真の事後分布)
            B6_raw_name = f'B6_raw_{model_form}'
            B6_sigma = max(0.010, 3.0 * (v8_params_model.get('B6_std') or 0.0))
            B6_raw = pm.Normal(B6_raw_name, mu=v8_params_model['B6'], sigma=B6_sigma)
            B6_scaled = pm.Deterministic('B6_scaled',
                B6_raw * SCALING_FACTORS['B6'])

        # ------------------------------------------
        # 5. ε_bg: FSR 由来推定値を中心に TruncNormal
        # ------------------------------------------
        eps_bg_scaled = pm.TruncatedNormal(
            'eps_bg_scaled',
            mu=eps_v8_avg * SCALING_FACTORS['eps'],
            sigma=0.3 * SCALING_FACTORS['eps'],
            lower=12.0 * SCALING_FACTORS['eps'],
            upper=16.0 * SCALING_FACTORS['eps'])

        # FSR 尤度: キャビティ FSR から ε_bg を直接拘束
        # FSR_pred = c / (2 * sqrt(ε_bg) * d)  を StudentT で観測 FSR に整合させる
        if len(fsr_obs_list) > 0:
            eps_bg_val  = eps_bg_scaled / SCALING_FACTORS['eps']
            n_bg        = pt.sqrt(eps_bg_val)
            fsr_pred_thz = c / (2.0 * d_fixed * n_bg * 1e12)   # THz
            dist_fsr = pm.StudentT.dist(nu=NU_STUDENTT, mu=fsr_pred_thz, sigma=sigma_fsr)
            pm.Potential('ll_eps_fsr',
                pm.logp(dist_fsr, pt.as_tensor_variable(fsr_obs_median)))

        # ------------------------------------------
        # 6. γ: Non-centered 階層モデル (clip なし → exp() で正定値保証)
        # ------------------------------------------
        log_gamma_mu = pm.Normal('log_gamma_mu',
            mu=np.log(GAMMA_HYPERPRIOR_MU), sigma=0.5)
        log_gamma_sd = pm.HalfNormal('log_gamma_sd', sigma=0.5)

        gamma_raw = pm.Normal('gamma_raw', mu=0, sigma=1, shape=7)
        gamma_vec_unscaled = pm.Deterministic('gamma_vec',
            pt.exp(log_gamma_mu + log_gamma_sd * gamma_raw))
        # exp() が常に正を保証するため clip 不要
        gamma_vec_scaled = pm.Deterministic('gamma_vec_scaled',
            gamma_vec_unscaled * SCALING_FACTORS['gamma'])
        # 各 gamma への弱い v8 アンカー（同定性補助）
        gamma_v8 = np.asarray(v8_params_model['gamma'], dtype=np.float64)
        gamma_prior_sigma = np.maximum(0.01, 0.5 * np.abs(gamma_v8))
        gamma_anchor_z = (
            gamma_vec_unscaled - gamma_v8
        ) / gamma_prior_sigma
        pm.Potential('ll_gamma_v8_anchor', -0.1 * pt.sum(gamma_anchor_z ** 2))
        for i in range(7):
            pm.Deterministic(f'gamma_{i+1}_scaled', gamma_vec_scaled[i])
        pm.Deterministic('gamma_mean_scaled',
            pt.exp(log_gamma_mu) * SCALING_FACTORS['gamma'])
        pm.Deterministic('gamma_std_scaled',
            log_gamma_sd * SCALING_FACTORS['gamma'])

        # ------------------------------------------
        # 7. 【新規】MixedOutputModelOp で trans_pred + fwhm_pred + peak_freq_pred を取得
        # ------------------------------------------
        trans_pred_concat, fwhm_pred_vec, peak_freq_pred_vec = model_op(
            a_scale_scaled, gamma_vec_scaled, g_factor_scaled,
            B4_scaled, B6_scaled, eps_bg_scaled)

        # ------------------------------------------
        # 8. 【v9.0 改訂】複合尤度: pm.Potential で log 尤度を加算
        # ------------------------------------------
        # 8-a. ポラリトン領域 (スペクトル形状)
        # PyMC v5 では pm.logp(dist, value) を使用する
        if np.any(polariton_mask_concat):
            obs_pol  = trans_obs_concat[polariton_mask_concat]
            pred_pol = trans_pred_concat[polariton_mask_concat]
            sig_pol  = sigma_eff[polariton_mask_concat]
            dist_pol = pm.StudentT.dist(nu=NU_STUDENTT, mu=pred_pol, sigma=sig_pol)
            ll_pol   = pm.logp(dist_pol, obs_pol)
            pm.Potential('ll_polariton', ll_pol.sum())

        # 8-a2. 【v10.0】共振器スペクトル形状尤度は使用しない
        # v9.0 ではここに共振器全スペクトルの StudentT 尤度があったが、
        # Ablation study で結合定数 a を 2倍に歪めポラリトン RMSE を
        # 0.0918→0.2224 に劣化させる原因と判明したため除去。
        # 共振器の制約はピーク位置と FWHM の点制約のみで行う。

        # 8-b. 共振器マルチピーク (FWHM + ピーク位置)
        if len(fwhm_obs_vec) > 0:
            dist_fwhm = pm.StudentT.dist(nu=NU_STUDENTT, mu=fwhm_pred_vec, sigma=sigma_fwhm)
            ll_fwhm   = pm.logp(dist_fwhm, fwhm_obs_vec)
            pm.Potential('ll_cavity_fwhm', ll_fwhm.sum())

            # 8-b2. 共振器マルチピーク位置マッチング (v9.0: 近傍マッチング)
            dist_peak = pm.StudentT.dist(nu=NU_STUDENTT, mu=peak_freq_pred_vec, sigma=sigma_peak_freq)
            ll_peak   = pm.logp(dist_peak, peak_freq_obs_vec)
            pm.Potential('ll_cavity_peak_freq', ll_peak.sum())

        # 8-c. 背景領域 (スペクトル、weight=0.01)
        if USE_BACKGROUND_LIKELIHOOD and np.any(bg_mask_concat):
            obs_bg  = trans_obs_concat[bg_mask_concat]
            pred_bg = trans_pred_concat[bg_mask_concat]
            sig_bg  = sigma_eff[bg_mask_concat]
            dist_bg = pm.StudentT.dist(nu=NU_STUDENTT, mu=pred_bg, sigma=sig_bg)
            ll_bg   = pm.logp(dist_bg, obs_bg)
            pm.Potential('ll_background', ll_bg.sum())

    return model


# ============================================================================
# メイン処理
# ============================================================================
def main():
    global TARGET_DATA, SMC_DRAWS, SMC_CHAINS

    start_time = time.time()

    if DEBUG_MODE:
        print("\n" + "🔧" * 40)
        print("デバッグモード: ON (2データセット, 500サンプル)")
        print("🔧" * 40 + "\n")
        TARGET_DATA = TARGET_DATA[:2]
        SMC_DRAWS   = 500
        SMC_CHAINS  = 2

    print(f"\n{'='*80}")
    print("Bayesian Analysis v9.0 — Multi Cavity Peak (Polariton + Cavity spectrum / Multi-peak FWHM+position)")
    print(f"v8 ボトルネック修正: 共振器スペクトル形状 + マルチピーク近傍マッチング")
    print(f"{'='*80}")

    runtime_cfg = resolve_sampling_resources(SMC_CHAINS, SMC_PARALLEL)
    effective_chains = runtime_cfg['effective_chains']
    effective_cores = runtime_cfg['effective_cores']
    print(
        "実行リソース設定: "
        f"logical_cpu={runtime_cfg['logical_cpus']}, "
        f"safe_cap={runtime_cfg['safe_cap']}, "
        f"chains={effective_chains}, cores={effective_cores}, "
        f"blas_threads={DEFAULT_BLAS_THREADS}"
    )

    # v8 参照値読み込み
    v8_params_H = load_v8_optimized_params('H')
    v8_params_B = load_v8_optimized_params('B')
    if v8_params_H is None or v8_params_B is None:
        print("❌ v8 最適化結果の読み込みに失敗しました")
        return

    # 結果ディレクトリ
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = pathlib.Path(__file__).parent / f"bayesian_v10_results_{timestamp}"
    results_dir.mkdir(exist_ok=True)
    print(f"\n📁 結果保存先: {results_dir}")

    # データ読み込み
    datasets = load_all_datasets(TARGET_DATA)
    if not datasets:
        print("❌ データがありません")
        return

    print(f"\n{'='*80}\nH 形式モデル構築・サンプリング\n{'='*80}")
    with build_pymc_model(datasets, 'H', v8_params_H, v8_params_B):
        print(f"  Draws={SMC_DRAWS}, Chains={effective_chains}, Cores={effective_cores}")
        trace_H = pm.sample_smc(
            draws=SMC_DRAWS, chains=effective_chains,
            cores=effective_cores,
            return_inferencedata=True, progressbar=True,
            random_seed=RANDOM_SEED)
    print("✅ H 形式サンプリング完了")

    print(f"\n{'='*80}\nB 形式モデル構築・サンプリング\n{'='*80}")
    with build_pymc_model(datasets, 'B', v8_params_H, v8_params_B):
        print(f"  Draws={SMC_DRAWS}, Chains={effective_chains}, Cores={effective_cores}")
        trace_B = pm.sample_smc(
            draws=SMC_DRAWS, chains=effective_chains,
            cores=effective_cores,
            return_inferencedata=True, progressbar=True,
            random_seed=RANDOM_SEED)
    print("✅ B 形式サンプリング完了")

    # モデル評価
    eval_H = compute_model_evaluation(trace_H, 'H-form')
    eval_B = compute_model_evaluation(trace_B, 'B-form')
    comparison_result = compare_models(eval_H, eval_B)
    bf_result = compute_bayes_factor_smc(trace_H, trace_B)

    # 可視化
    print(f"\n{'='*80}\n📊 プロット生成\n{'='*80}")
    plot_posterior_distributions(trace_H, 'H', results_dir)
    plot_posterior_predictive_spectra(trace_H, datasets, 'H', results_dir)
    plot_posterior_distributions(trace_B, 'B', results_dir)
    plot_posterior_predictive_spectra(trace_B, datasets, 'B', results_dir)
    plot_posterior_predictive_spectra_combined(trace_H, trace_B, datasets, results_dir)
    plot_energy_levels(trace_H, datasets, 'H', results_dir)
    plot_susceptibility(trace_H, datasets, 'H', results_dir)
    plot_energy_levels(trace_B, datasets, 'B', results_dir)
    plot_susceptibility(trace_B, datasets, 'B', results_dir)
    print("✅ 全プロット生成完了")

    # 結果保存
    print(f"\n{'='*80}\n結果保存\n{'='*80}")
    for form, trace in [('H', trace_H), ('B', trace_B)]:
        try:
            trace.to_netcdf(str(results_dir / f'trace_{form}.nc'))
            print(f"  ✓ trace_{form}.nc")
        except Exception:
            import pickle
            with open(results_dir / f'trace_{form}.pkl', 'wb') as fh:
                pickle.dump(trace, fh)
            print(f"  ✓ trace_{form}.pkl (pickle)")

        summary = az.summary(trace)
        summary.to_csv(results_dir / f'summary_{form}.csv')
        print(f"  ✓ summary_{form}.csv")

        posterior = trace.posterior
        params_out = {
            'g':   float(posterior['g_factor_scaled'].mean()) / SCALING_FACTORS['g'],
            'a':   float(posterior['a_scale_scaled'].mean())  / SCALING_FACTORS['a'],
            'B4':  float(posterior['B4_scaled'].mean())       / SCALING_FACTORS['B4'],
            'B6':  0.0 if FIX_B6_ZERO else float(posterior['B6_scaled'].mean()) / SCALING_FACTORS['B6'],
            'eps': float(posterior['eps_bg_scaled'].mean())   / SCALING_FACTORS['eps'],
            'gamma': [float(posterior[f'gamma_{i+1}_scaled'].mean()) / SCALING_FACTORS['gamma'] for i in range(7)],
            'gamma_mean': float(posterior['gamma_mean_scaled'].mean()) / SCALING_FACTORS['gamma'],
        }
        pd.DataFrame([{k: v if not isinstance(v, list) else str(v) for k, v in params_out.items()}]).to_csv(
            results_dir / f'parameters_{form}.csv', index=False)
        print(f"  ✓ parameters_{form}.csv")

    eval_results = {
        'H_form': eval_H, 'B_form': eval_B,
        'comparison_ess': comparison_result,
        'comparison_bayes_factor': bf_result,
        'timestamp': timestamp,
        'sampler': SAMPLER_TYPE,
        'likelihood': LIKELIHOOD_TYPE,
        'fix_B6_zero': FIX_B6_ZERO,
        'sigma_fwhm': SIGMA_FWHM,
        'sigma_cavity_spectrum': 'N/A (v10: removed)',
        'peak_match_max_distance': PEAK_MATCH_MAX_DISTANCE,
        'use_background_likelihood': USE_BACKGROUND_LIKELIHOOD,
        'normalization': 'obs_shared',   # 観測データ min/max をモデルと共有
    }
    with open(results_dir / 'model_evaluation.json', 'w') as fh:
        json.dump(eval_results, fh, indent=2, default=str)
    print("  ✓ model_evaluation.json")

    total_time = time.time() - start_time
    print(f"\n{'='*80}\n🎉 全処理完了\n{'='*80}")
    print(f"  実行時間  : {total_time:.1f} 秒 ({total_time/60:.1f} 分)")
    print(f"  結果保存先: {results_dir}")
    print(f"  尤度      : {LIKELIHOOD_TYPE} (ポラリトン=スペクトル / 共振器=ピーク位置+FWHM)")
    print(f"  推奨モデル: {comparison_result.get('winner', 'N/A')} (ESS比較)")
    if bf_result:
        print(f"  推奨モデル: {bf_result.get('winner', 'N/A')} (ベイズファクター)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
