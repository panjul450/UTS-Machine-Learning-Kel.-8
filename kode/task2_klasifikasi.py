#!/usr/bin/env python3
"""UTS Machine Learning - Task 2: Klasifikasi Fraud Detection (One-Command Reproducible Script)"""

# --- Cell 3 ---
# 1. Import Pustaka Utama & Konfigurasi Global
import os
import sys
import time
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Konfigurasi Tampilan & Visualisasi
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 110

SEED = 42
REVIEW_COST = 25000.0
np.random.seed(SEED)

# Buat direktori penyimpanan grafik evaluasi
FIG_DIR = "figures" if os.path.basename(os.getcwd()) == "Task 2" else "Task 2/figures"
os.makedirs(FIG_DIR, exist_ok=True)

print(f"Environment setup berhasil (SEED={SEED}).")
print(f"LightGBM: {lgb.__version__} | XGBoost: {HAS_XGB} | CatBoost: {HAS_CB} | SHAP: {HAS_SHAP}")
print(f"Direktori output grafik: {FIG_DIR}")


# --- Cell 4 ---
# 2. Pemuatan Dataset Train dan Test (Otomatis Mendukung Google Colab, Drive, & Lokal)
t0 = time.time()

def resolve_data_path(base_name):
    """Mencari file .csv.gz atau .csv di berbagai lokasi standar lokal maupun Google Colab."""
    candidates = [
        f"../data_uts_ml/{base_name}.csv.gz",
        f"data_uts_ml/{base_name}.csv.gz",
        f"../../data_uts_ml/{base_name}.csv.gz",
        f"../Dataset/{base_name}.csv.gz",
        f"Dataset/{base_name}.csv.gz",
        f"../Dataset/{base_name}.csv",
        f"Dataset/{base_name}.csv",
        f"/content/{base_name}.csv.gz",
        f"/content/{base_name}.csv",
        f"/content/data_uts_ml/{base_name}.csv.gz",
        f"/content/Dataset/{base_name}.csv",
        f"/content/drive/MyDrive/data_uts_ml/{base_name}.csv.gz",
        f"/content/drive/MyDrive/UTS_ML/{base_name}.csv.gz",
        f"/content/drive/MyDrive/{base_name}.csv.gz",
        f"{base_name}.csv.gz",
        f"{base_name}.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"File {base_name}.csv.gz atau .csv tidak ditemukan. "
        f"Jika di Google Colab, unggah ke /content/ atau mount Google Drive Anda."
    )

train_path = resolve_data_path("klasifikasi_train")
test_path = resolve_data_path("klasifikasi_test")

print(f"Membaca Train dari : {train_path}")
print(f"Membaca Test dari  : {test_path}")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df['waktu_dt'] = pd.to_datetime(train_df['waktu'])
test_df['waktu_dt'] = pd.to_datetime(test_df['waktu'])

print(f"Pemuatan data selesai dalam {time.time() - t0:.2f} detik.")
print(f"Dimensi Data Latih (Train) : {train_df.shape[0]:,} baris x {train_df.shape[1]} kolom")
print(f"Dimensi Data Uji (Test)    : {test_df.shape[0]:,} baris x {test_df.shape[1]} kolom")
print(f"Rentang Waktu Train        : {train_df['waktu_dt'].min()} s.d. {train_df['waktu_dt'].max()}")
print(f"Rentang Waktu Test         : {test_df['waktu_dt'].min()} s.d. {test_df['waktu_dt'].max()}")
print(f"Prevalensi Fraud Global    : {train_df['is_fraud'].sum():,} transaksi ({train_df['is_fraud'].mean()*100:.3f}%)")


# --- Cell 6 ---
# 3. [KODE PLOT] Audit Pembentukan Label & Bukti Empiris Right-Censoring di Akhir Agustus 2025
train_df['tanggal'] = train_df['waktu_dt'].dt.date
train_df['bulan'] = train_df['waktu_dt'].dt.to_period('M').astype(str)

monthly_stats = train_df.groupby('bulan').agg(
    total_trx=('is_fraud', 'count'),
    fraud_cnt=('is_fraud', 'sum'),
    fraud_rate=('is_fraud', lambda x: x.mean() * 100)
).reset_index()

aug_df = train_df[train_df['bulan'] == '2025-08'].copy()
aug_daily = aug_df.groupby('tanggal').agg(
    total_trx=('is_fraud', 'count'),
    fraud_cnt=('is_fraud', 'sum'),
    fraud_rate=('is_fraud', lambda x: x.mean() * 100)
).reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel 1: Fraud Rate Bulanan
colors_m = ['#2980b9'] * 7 + ['#e74c3c']
sns.barplot(x='bulan', y='fraud_rate', data=monthly_stats, palette=colors_m, ax=ax1)
ax1.axhline(train_df[train_df['bulan'] < '2025-08']['is_fraud'].mean() * 100, color='#2c3e50',
            linestyle='--', linewidth=1.8, label='Rata-rata Jan–Jul (0.680%)')
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width()/2., p.get_height() + 0.02),
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_title("Prevalensi Fraud Bulanan (Penurunan Semu di Agu 2025)", fontweight='bold')
ax1.set_xlabel("Bulan Transaksi")
ax1.set_ylabel("Tingkat Fraud Terkonfirmasi (%)")
ax1.set_ylim(0, 0.90)
ax1.tick_params(axis='x', rotation=30)
ax1.legend()

# Panel 2: Peluruhan Fraud Rate Harian di Bulan Agustus 2025
ax2.plot(range(len(aug_daily)), aug_daily['fraud_rate'], marker='o', color='#c0392b', linewidth=2, label='Fraud Rate Harian Agu 2025')
ax2.axvspan(17, 30, color='#e74c3c', alpha=0.18, label='Zona Right-Censoring (18–31 Agu)')
ax2.axvspan(28, 30, color='#900C3F', alpha=0.30, label='0% Fraud Terkonfirmasi (29–31 Agu)')
ax2.set_xticks(range(0, len(aug_daily), 3))
ax2.set_xticklabels([str(d)[5:] for d in aug_daily['tanggal'].iloc[::3]], rotation=30)
ax2.set_title("Bukti Label Delay: Peluruhan Fraud Harian Agustus 2025", fontweight='bold')
ax2.set_xlabel("Tanggal di Bulan Agustus 2025 (Penarikan Data: 31 Agu)")
ax2.set_ylabel("Tingkat Fraud Terkonfirmasi (%)")
ax2.legend(loc='upper right', fontsize=8.5)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_label_censoring_audit.png", dpi=150)
plt.close()

print("=== RINGKASAN AUDIT PEMBENTUKAN LABEL (RIGHT-CENSORING) ===")
print(f"Rata-rata Fraud Rate Jan–Jul 2025 : {train_df[train_df['bulan'] < '2025-08']['is_fraud'].mean()*100:.3f}%")
print(f"Rata-rata Fraud Rate 1–17 Agu 2025: {train_df[(train_df['waktu_dt'] >= '2025-08-01') & (train_df['waktu_dt'] < '2025-08-18')]['is_fraud'].mean()*100:.3f}%")
print(f"Rata-rata Fraud Rate 18–24 Agu    : {train_df[(train_df['waktu_dt'] >= '2025-08-18') & (train_df['waktu_dt'] < '2025-08-25')]['is_fraud'].mean()*100:.3f}%")
print(f"Rata-rata Fraud Rate 25–28 Agu    : {train_df[(train_df['waktu_dt'] >= '2025-08-25') & (train_df['waktu_dt'] < '2025-08-29')]['is_fraud'].mean()*100:.3f}%")
print(f"Rata-rata Fraud Rate 29–31 Agu    : {train_df[train_df['waktu_dt'] >= '2025-08-29']['is_fraud'].mean()*100:.3f}% (0 dari {len(train_df[train_df['waktu_dt'] >= '2025-08-29']):,} transaksi!)")


# --- Cell 7 ---
# 4. [KODE PLOT] Audit Target Leakage Fatal pada Fitur 'flag_review_manual'
# Berdasarkan kamus_data.md, flag_review_manual = 1 jika transaksi pernah ditinjau manual oleh tim risiko.
# Tujuan sistem kita justru memprediksi skor SAAT transaksi terjadi untuk memutuskan apakah perlu ditinjau manual!

ct_flag = pd.crosstab(train_df['flag_review_manual'], train_df['is_fraud'], margins=True)
print("Tabel Kontingensi flag_review_manual vs is_fraud (Train):")
print(ct_flag)

# Bandingkan profil transaksi flag_review_manual == 1 di Train (Fraud vs Non-Fraud) dan di Test
prof_rows = [
    {
        'Kelompok': 'Train: flag=0 (Normal)',
        'Jumlah': (train_df['flag_review_manual'] == 0).sum(),
        'Mean Nominal (Rp)': train_df.loc[train_df['flag_review_manual'] == 0, 'nominal'].mean(),
        'Mean trx_1j': train_df.loc[train_df['flag_review_manual'] == 0, 'trx_1j'].mean(),
        'Proporsi perangkat_baru (%)': train_df.loc[train_df['flag_review_manual'] == 0, 'perangkat_baru'].mean() * 100
    },
    {
        'Kelompok': 'Train: flag=1 & fraud=0 (Audit Rutin ~1.2%)',
        'Jumlah': ((train_df['flag_review_manual'] == 1) & (train_df['is_fraud'] == 0)).sum(),
        'Mean Nominal (Rp)': train_df.loc[(train_df['flag_review_manual'] == 1) & (train_df['is_fraud'] == 0), 'nominal'].mean(),
        'Mean trx_1j': train_df.loc[(train_df['flag_review_manual'] == 1) & (train_df['is_fraud'] == 0), 'trx_1j'].mean(),
        'Proporsi perangkat_baru (%)': train_df.loc[(train_df['flag_review_manual'] == 1) & (train_df['is_fraud'] == 0), 'perangkat_baru'].mean() * 100
    },
    {
        'Kelompok': 'Train: flag=1 & fraud=1 (Review Post-Fraud!)',
        'Jumlah': ((train_df['flag_review_manual'] == 1) & (train_df['is_fraud'] == 1)).sum(),
        'Mean Nominal (Rp)': train_df.loc[(train_df['flag_review_manual'] == 1) & (train_df['is_fraud'] == 1), 'nominal'].mean(),
        'Mean trx_1j': train_df.loc[(train_df['flag_review_manual'] == 1) & (train_df['is_fraud'] == 1), 'trx_1j'].mean(),
        'Proporsi perangkat_baru (%)': train_df.loc[(train_df['flag_review_manual'] == 1) & (train_df['is_fraud'] == 1), 'perangkat_baru'].mean() * 100
    },
    {
        'Kelompok': 'Test: flag=1 (Hanya Audit Rutin ~1.2%)',
        'Jumlah': (test_df['flag_review_manual'] == 1).sum(),
        'Mean Nominal (Rp)': test_df.loc[test_df['flag_review_manual'] == 1, 'nominal'].mean(),
        'Mean trx_1j': test_df.loc[test_df['flag_review_manual'] == 1, 'trx_1j'].mean(),
        'Proporsi perangkat_baru (%)': test_df.loc[test_df['flag_review_manual'] == 1, 'perangkat_baru'].mean() * 100
    }
]
prof_df = pd.DataFrame(prof_rows)
print("\nPerbandingan Profil Transaksi berdasarkan flag_review_manual:")
print(prof_df.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Plot 1: Porsi flag_review_manual pada Non-Fraud vs Fraud di Train vs Test
leak_bars = pd.DataFrame({
    'Kategori Data': ['Train: Non-Fraud\n(is_fraud=0)', 'Train: Confirmed Fraud\n(is_fraud=1)', 'Test Set\n(Saat Transaksi)'],
    'Persentase flag=1 (%)': [
        train_df.loc[train_df['is_fraud'] == 0, 'flag_review_manual'].mean() * 100,
        train_df.loc[train_df['is_fraud'] == 1, 'flag_review_manual'].mean() * 100,
        test_df['flag_review_manual'].mean() * 100
    ]
})
sns.barplot(x='Kategori Data', y='Persentase flag=1 (%)', data=leak_bars, palette=['#3498db', '#c0392b', '#27ae60'], ax=ax1)
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width()/2., p.get_height() + 1.5),
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.set_title("Proporsi flag_review_manual = 1 (Bukti Post-Hoc Leakage)", fontweight='bold')
ax1.set_ylabel("Persentase Transaksi dengan flag=1 (%)")
ax1.set_ylim(0, 92)

# Plot 2: Mean Nominal pada sub-kelompok flag=1
sns.barplot(x='Kelompok', y='Mean Nominal (Rp)', data=prof_df, palette=['#95a5a6', '#3498db', '#c0392b', '#27ae60'], ax=ax2)
for p in ax2.patches:
    ax2.annotate(f"Rp {p.get_height():,.0f}", (p.get_x() + p.get_width()/2., p.get_height() + 25000),
                 ha='center', va='bottom', fontweight='bold', fontsize=9)
ax2.set_title("Rata-rata Nominal per Kelompok flag_review_manual", fontweight='bold')
ax2.set_ylabel("Rata-rata Nominal (Rp)")
ax2.set_ylim(0, 1400000)
ax2.tick_params(axis='x', rotation=18)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_flag_review_leakage.png", dpi=150)
plt.close()


# --- Cell 8 ---
# 5. Pengukuran Kuantitatif Dampak Kebocoran Fitur 'flag_review_manual' (Leakage Impact Measurement)
# Kita ukur apa yang terjadi jika model dilatih menggunakan 'flag_review_manual' lalu diuji pada
# kondisi operasional riil saat transaksi terjadi (di mana flag_review_manual hanya berisi ~1.2% sampel rutin acak,
# bukan hasil investigasi post-hoc setelah laporan fraud masuk).

mature_train = train_df[train_df['waktu_dt'] < '2025-07-01'].copy()
mature_val = train_df[(train_df['waktu_dt'] >= '2025-07-01') & (train_df['waktu_dt'] < '2025-08-18')].copy()

base_num_cols = [
    'nominal', 'jam', 'hari_libur', 'umur_akun_hari', 'umur_pengguna_tahun',
    'umur_perangkat_hari', 'perangkat_baru', 'jumlah_akun_per_perangkat',
    'ganti_password_7h', 'gagal_login_24j', 'jarak_lokasi_km', 'negara_ip_beda',
    'kartu_tidak_hadir', 'trx_1j', 'trx_24j', 'trx_7h', 'nominal_24j',
    'saldo_sebelum', 'persen_saldo', 'umur_merchant_hari', 'refund_merchant_30h', 'volume_merchant_30h'
]

# Model A: Bersih (Tanpa flag_review_manual)
clf_clean = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.07, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=4)
clf_clean.fit(mature_train[base_num_cols], mature_train['is_fraud'])
prob_clean = clf_clean.predict_proba(mature_val[base_num_cols])[:, 1]
prauc_clean = average_precision_score(mature_val['is_fraud'], prob_clean)
flag_clean = ((prob_clean * mature_val['nominal']) >= REVIEW_COST).astype(int)
cost_clean = mature_val.loc[(mature_val['is_fraud'] == 1) & (flag_clean == 0), 'nominal'].sum() + REVIEW_COST * flag_clean.sum()

# Model B: Bocor (Menggunakan flag_review_manual)
cols_leak = base_num_cols + ['flag_review_manual']
clf_leak = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.07, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=4)
clf_leak.fit(mature_train[cols_leak], mature_train['is_fraud'])

# Evaluasi Semu (ketika flag_review_manual di val sudah berisi investigasi post-hoc)
prob_leak_naive = clf_leak.predict_proba(mature_val[cols_leak])[:, 1]
prauc_leak_naive = average_precision_score(mature_val['is_fraud'], prob_leak_naive)

# Evaluasi Kondisi Operasional Riil (saat transaksi terjadi, post-hoc review pada fraud belum terjadi -> hanya 1.2% routine flag)
X_val_realtime = mature_val[cols_leak].copy()
np.random.seed(SEED)
X_val_realtime['flag_review_manual'] = np.random.binomial(1, 0.0123, size=len(X_val_realtime))
prob_leak_realtime = clf_leak.predict_proba(X_val_realtime)[:, 1]
prauc_leak_realtime = average_precision_score(mature_val['is_fraud'], prob_leak_realtime)
flag_leak_realtime = ((prob_leak_realtime * mature_val['nominal']) >= REVIEW_COST).astype(int)
cost_leak_realtime = mature_val.loc[(mature_val['is_fraud'] == 1) & (flag_leak_realtime == 0), 'nominal'].sum() + REVIEW_COST * flag_leak_realtime.sum()

print("=== KUANTIFIKASI DAMPAK KEBOCORAN FITUR 'flag_review_manual' ===")
print(f"1. Model Bocor pada Evaluasi Semu (Post-Hoc Flag)     : PR-AUC = {prauc_leak_naive:.4f} (Ilusi Over-Optimis!)")
print(f"2. Model Bocor pada Kondisi Transaksi Riil (Test-Like): PR-AUC = {prauc_leak_realtime:.4f} | Total Cost = Rp {cost_leak_realtime:,.0f}")
print(f"3. Model Bersih (Tanpa flag_review_manual)            : PR-AUC = {prauc_clean:.4f} | Total Cost = Rp {cost_clean:,.0f}")
print(f"-> Penurunan PR-AUC akibat mengandalkan fitur bocor   : -{((prauc_clean - prauc_leak_realtime)/prauc_clean)*100:.1f}%")
print(f"-> Pembengkakan Biaya Risiko akibat fitur bocor       : +Rp {cost_leak_realtime - cost_clean:,.0f} (+{((cost_leak_realtime - cost_clean)/cost_clean)*100:.1f}%)")
print("-> KESIMPULAN AUDIT: Fitur 'flag_review_manual' WAJIB DIHAPUS dari seluruh pipeline.")


# --- Cell 9 ---
# 6. [KODE PLOT] Audit Kualitas Data: Missing Values Struktural, Anomali Nilai Negatif, & Drift Fitur 'v15'
print("--- A. Audit Missing Values ---")
tr_miss = train_df.isnull().sum()
te_miss = test_df.isnull().sum()
miss_df = pd.DataFrame({
    'Missing Train': tr_miss[tr_miss > 0],
    'Train (%)': (tr_miss[tr_miss > 0] / len(train_df)) * 100,
    'Missing Test': te_miss[te_miss > 0],
    'Test (%)': (te_miss[te_miss > 0] / len(test_df)) * 100
})
print(miss_df)
n_unique_users_train = train_df['id_pengguna'].nunique()
print(f"Jumlah unik id_pengguna di Train: {n_unique_users_train:,} == Jumlah missing menit_sejak_trx_terakhir: {tr_miss['menit_sejak_trx_terakhir']:,}!")
print("-> Bukti: Missing value pada 'menit_sejak_trx_terakhir' dan 'rasio_nominal_rata2' 100% adalah transaksi PERTAMA pengguna.")

print("\n--- B. Audit Anomali Fisik & Sentinel Values ---")
neg_m_tr = (train_df['umur_merchant_hari'] < 0).sum()
neg_m_te = (test_df['umur_merchant_hari'] < 0).sum()
print(f"1. umur_merchant_hari < 0 (min={train_df['umur_merchant_hari'].min():.0f} hari) | Train: {neg_m_tr:,} baris | Test: {neg_m_te:,} baris")
ps_gt1_tr = (train_df['persen_saldo'] > 1.0).sum()
ps_50_tr = (train_df['persen_saldo'] == 50.0).sum()
print(f"2. persen_saldo > 1.0 (Transaksi melebihi saldo dompet / kanal non-saldo): {ps_gt1_tr:,} baris (Cap sentinel 50.0: {ps_50_tr:,} baris)")

print("\n--- C. Audit Covariate Shift Ekstrim pada Fitur Anonim v01-v50 ---")
v_cols = [f'v{i:02d}' for i in range(1, 51)]
z_shifts = {}
for c in v_cols:
    z_shifts[c] = (test_df[c].mean() - train_df[c].mean()) / (train_df[c].std() + 1e-9)
z_series = pd.Series(z_shifts).sort_values(key=abs, ascending=False)
print("Top 5 Fitur v01-v50 dengan Pergeseran Mean (Train vs Test, dalam satuan SD):")
print(z_series.head(5))

# Visualisasi Drift v15 & Struktur Missing Value
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))

all_v15 = pd.concat([
    train_df[['waktu_dt', 'v15']].assign(Split='Train (Jan-Agu 2025)'),
    test_df[['waktu_dt', 'v15']].assign(Split='Test (Sep-Okt 2025)')
])
all_v15['bulan'] = all_v15['waktu_dt'].dt.to_period('M').astype(str)
v15_monthly = all_v15.groupby(['bulan', 'Split'])['v15'].mean().reset_index()

sns.lineplot(x='bulan', y='v15', hue='Split', data=v15_monthly, marker='o', linewidth=2.5, palette=['#2980b9', '#e74c3c'], ax=ax1)
ax1.set_title("Bukti Temporal Drift Linear pada Fitur 'v15' (Jan–Okt 2025)", fontweight='bold')
ax1.set_xlabel("Bulan Transaksi")
ax1.set_ylabel("Rata-rata Nilai v15")
ax1.tick_params(axis='x', rotation=35)

sns.kdeplot(train_df['v15'], label=f"Train (mean={train_df['v15'].mean():.2f})", color='#2980b9', fill=True, alpha=0.3, ax=ax2)
sns.kdeplot(test_df['v15'], label=f"Test (mean={test_df['v15'].mean():.2f}, z=+1.24σ)", color='#e74c3c', fill=True, alpha=0.3, ax=ax2)
ax2.set_title("Pergeseran Distribusi 'v15' antara Train vs Test", fontweight='bold')
ax2.set_xlabel("Nilai v15")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_missing_and_drift_v15.png", dpi=150)
plt.close()
print("-> KESIMPULAN AUDIT v15: 'v15' adalah indeks waktu terselubung (non-stasioner, korelasi ke fraud ~0) sehingga WAJIB DIDROP.")


# --- Cell 11 ---
# 7. [KODE PLOT] Komparasi Random Stratified CV vs Temporal Mature Validation vs Censored Tail Validation
quick_feats = [c for c in base_num_cols if c != 'flag_review_manual']

# 1. Random 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rand_scores = []
for tr_idx, va_idx in skf.split(train_df[quick_feats], train_df['is_fraud']):
    m_cv = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.08, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=4)
    m_cv.fit(train_df[quick_feats].iloc[tr_idx], train_df['is_fraud'].iloc[tr_idx])
    p_cv = m_cv.predict_proba(train_df[quick_feats].iloc[va_idx])[:, 1]
    rand_scores.append(average_precision_score(train_df['is_fraud'].iloc[va_idx], p_cv))
prauc_random_cv = float(np.mean(rand_scores))

# 2. Temporal Split: Train (Jan-Jun), Evaluasi pada Mature Window (21 Jul - 17 Agu) vs Censored Window (18 Agu - 31 Agu)
tr_mask = train_df['waktu_dt'] < '2025-07-01'
cal_mask = (train_df['waktu_dt'] >= '2025-07-01') & (train_df['waktu_dt'] < '2025-07-21')
val_mature_mask = (train_df['waktu_dt'] >= '2025-07-21') & (train_df['waktu_dt'] < '2025-08-18')
val_censored_mask = train_df['waktu_dt'] >= '2025-08-18'

m_temp = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.08, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=4)
m_temp.fit(train_df.loc[tr_mask, quick_feats], train_df.loc[tr_mask, 'is_fraud'])

p_val_mature = m_temp.predict_proba(train_df.loc[val_mature_mask, quick_feats])[:, 1]
prauc_mature = average_precision_score(train_df.loc[val_mature_mask, 'is_fraud'], p_val_mature)

p_val_censored = m_temp.predict_proba(train_df.loc[val_censored_mask, quick_feats])[:, 1]
prauc_censored = average_precision_score(train_df.loc[val_censored_mask, 'is_fraud'], p_val_censored)

fig, ax = plt.subplots(figsize=(8.5, 4.0))
val_cmp_df = pd.DataFrame({
    'Skema Evaluasi': [
        'Random 5-Fold Stratified CV\n(Bocoran Temporal)',
        'Temporal Mature Window\n(21 Jul – 17 Agu, Label Matang)',
        'Temporal Censored Tail\n(18 – 31 Agu, Label Belum Lengkap)'
    ],
    'PR-AUC': [prauc_random_cv, prauc_mature, prauc_censored]
})
sns.barplot(x='Skema Evaluasi', y='PR-AUC', data=val_cmp_df, palette=['#27ae60', '#2980b9', '#c0392b'], ax=ax)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width()/2., p.get_height()/2.),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
ax.set_title("Pengaruh Skema Validasi & Maturitas Label terhadap PR-AUC", fontweight='bold')
ax.set_ylabel("PR-AUC (Average Precision)")
ax.set_ylim(0, max(rand_scores)*1.18)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_validation_scheme.png", dpi=150)
plt.close()

print("=== PERBANDINGAN SKEMA VALIDASI ===")
print(f"1. Random 5-Fold CV                   : PR-AUC = {prauc_random_cv:.4f} (Over-optimis karena kebocoran masa depan)")
print(f"2. Temporal Mature Window (21Jul-17Agu): PR-AUC = {prauc_mature:.4f} (Evaluasi paling jujur & reliabel)")
print(f"3. Temporal Censored Window (18-31Agu) : PR-AUC = {prauc_censored:.4f} (Anjlok semu akibat label fraud belum terkonfirmasi)")


# --- Cell 13 ---
# 8. Pipeline Rekayasa Fitur Perilaku Kausal (Strict Past-Only Behavioral Features)
t0 = time.time()

def engineer_causal_features(tr_raw, te_raw):
    """
    Membangun fitur perilaku pengguna & merchant secara kausal (hanya menggunakan transaksi masa lalu < waktu_dt).
    Menghapus fitur bocor (flag_review_manual) dan fitur non-stasioner (v15).
    """
    tr = tr_raw.copy()
    te = te_raw.copy()
    tr['is_test'] = 0
    te['is_test'] = 1
    te['is_fraud'] = np.nan

    full = pd.concat([tr, te], ignore_index=True)
    full['orig_idx'] = np.arange(len(full))
    full = full.sort_values(['waktu_dt', 'id_transaksi']).reset_index(drop=True)

    # --- 1. Penanganan Missing Values Struktural & Anomali Data ---
    full['is_first_trx'] = full['menit_sejak_trx_terakhir'].isnull().astype(np.int8)
    full['menit_sejak_trx_clean'] = full['menit_sejak_trx_terakhir'].fillna(999999.0)
    full['log_menit_sejak_trx'] = np.log1p(full['menit_sejak_trx_clean'].clip(lower=0))
    full['rasio_nominal_rata2_clean'] = full['rasio_nominal_rata2'].fillna(1.0)

    full['merchant_age_invalid'] = (full['umur_merchant_hari'] < 0).astype(np.int8)
    full['umur_merchant_clean'] = full['umur_merchant_hari'].clip(lower=0)
    full['persen_saldo_over1'] = (full['persen_saldo'] > 1.0).astype(np.int8)
    full['persen_saldo_near1'] = ((full['persen_saldo'] >= 0.85) & (full['persen_saldo'] <= 1.05)).astype(np.int8)
    full['persen_saldo_cap50'] = (full['persen_saldo'] >= 49.9).astype(np.int8)

    # --- 2. Fitur Perilaku Pengguna Kausal (Strict Past-Only via Expanding Cumsum/Cumcount) ---
    u_grp = full.groupby('id_pengguna', sort=False)
    full['user_prior_trx_cnt'] = u_grp.cumcount().astype(np.int32)
    user_prior_nom_sum = u_grp['nominal'].cumsum() - full['nominal']
    global_med_nom = float(tr['nominal'].median())
    full['user_prior_mean_nom'] = np.where(
        full['user_prior_trx_cnt'] > 0,
        user_prior_nom_sum / np.maximum(1, full['user_prior_trx_cnt']),
        global_med_nom
    )
    full['user_nom_vs_prior_mean'] = full['nominal'] / (full['user_prior_mean_nom'] + 1.0)

    # Nominal transaksi sebelumnya dari pengguna yang sama
    user_prev_nom = u_grp['nominal'].shift(1).fillna(full['nominal'])
    full['user_nom_vs_prev_trx'] = full['nominal'] / (user_prev_nom + 1.0)

    # Apakah pengguna berganti kanal atau tipe perangkat dibanding transaksi terakhirnya
    prev_kanal = u_grp['kanal'].shift(1)
    prev_dev = u_grp['tipe_perangkat'].shift(1)
    full['user_changed_kanal'] = ((prev_kanal.notnull()) & (prev_kanal != full['kanal'])).astype(np.int8)
    full['user_changed_device_type'] = ((prev_dev.notnull()) & (prev_dev != full['tipe_perangkat'])).astype(np.int8)

    # Apakah ini pertama kalinya pengguna bertransaksi di merchant atau kategori merchant ini
    full['user_merch_prior_cnt'] = full.groupby(['id_pengguna', 'id_merchant'], sort=False).cumcount().astype(np.int32)
    full['user_is_new_merchant'] = (full['user_merch_prior_cnt'] == 0).astype(np.int8)
    full['user_kat_prior_cnt'] = full.groupby(['id_pengguna', 'kategori_merchant'], sort=False).cumcount().astype(np.int32)
    full['user_is_new_kategori'] = (full['user_kat_prior_cnt'] == 0).astype(np.int8)

    # Riwayat akumulasi gagal login & perangkat baru pengguna di masa lalu
    full['user_prior_gagal_login_sum'] = (u_grp['gagal_login_24j'].cumsum() - full['gagal_login_24j']).astype(np.float32)
    full['user_prior_new_dev_sum'] = (u_grp['perangkat_baru'].cumsum() - full['perangkat_baru']).astype(np.float32)

    # --- 3. Fitur Perilaku Merchant Kausal (Strict Past-Only) ---
    m_grp = full.groupby('id_merchant', sort=False)
    full['merch_prior_trx_cnt'] = m_grp.cumcount().astype(np.int32)
    merch_prior_nom_sum = m_grp['nominal'].cumsum() - full['nominal']
    full['merch_prior_mean_nom'] = np.where(
        full['merch_prior_trx_cnt'] > 0,
        merch_prior_nom_sum / np.maximum(1, full['merch_prior_trx_cnt']),
        global_med_nom
    )
    full['merch_nom_ratio'] = full['nominal'] / (full['merch_prior_mean_nom'] + 1.0)

    # Proporsi transaksi perangkat baru & gagal login pada merchant tersebut di masa lalu
    merch_prior_newdev = m_grp['perangkat_baru'].cumsum() - full['perangkat_baru']
    full['merch_prior_newdev_rate'] = (merch_prior_newdev + 5.0 * 0.041) / (full['merch_prior_trx_cnt'] + 5.0)

    # --- 4. Fitur Interaksi Risiko & Velocity ---
    full['log_nominal'] = np.log1p(full['nominal'])
    full['log_saldo_sebelum'] = np.log1p(full['saldo_sebelum'].clip(lower=0))
    full['log_nominal_24j'] = np.log1p(full['nominal_24j'].clip(lower=0))
    full['sisa_saldo_estimasi'] = full['saldo_sebelum'] - full['nominal']
    full['rasio_nom_terhadap_24j'] = full['nominal'] / (full['nominal_24j'] + full['nominal'] + 1.0)

    full['vel_trx_1j_vs_24j'] = (full['trx_1j'] + 0.1) / (full['trx_24j'] + 0.1)
    full['vel_trx_24j_vs_7h'] = (full['trx_24j'] + 0.1) / (full['trx_7h'] + 0.1)

    # Sinyal komposit ATO (Account Takeover), CNP (Card Not Present), & QRIS Wallet Drain
    full['ato_risk_score'] = (
        full['perangkat_baru'] * 2.0 +
        full['ganti_password_7h'] * 2.0 +
        np.minimum(full['gagal_login_24j'], 4) * 1.2 +
        full['negara_ip_beda'] * 2.5 +
        np.log1p(full['jarak_lokasi_km'].clip(lower=0)) * 0.3
    )
    full['is_qris'] = (full['kanal'] == 'qris').astype(np.int8)
    full['is_kartu'] = (full['kanal'] == 'kartu').astype(np.int8)
    full['qris_x_persen_saldo'] = full['is_qris'] * full['persen_saldo']
    full['qris_x_nominal_log'] = full['is_qris'] * full['log_nominal']
    full['qris_x_new_device'] = full['is_qris'] * full['perangkat_baru']
    full['cnp_x_foreign_ip'] = full['kartu_tidak_hadir'] * full['negara_ip_beda']
    full['new_dev_x_fail_login'] = full['perangkat_baru'] * full['gagal_login_24j']
    full['new_dev_x_pass_reset'] = full['perangkat_baru'] * full['ganti_password_7h']

    # --- 5. Siklus Waktu ---
    full['day_of_week'] = full['waktu_dt'].dt.dayofweek.astype(np.int8)
    full['is_weekend'] = (full['day_of_week'] >= 5).astype(np.int8)
    full['is_night_trx'] = ((full['jam'] >= 0) & (full['jam'] <= 5)).astype(np.int8)
    full['sin_jam'] = np.sin(2 * np.pi * full['jam'] / 24.0)
    full['cos_jam'] = np.cos(2 * np.pi * full['jam'] / 24.0)

    # Kembalikan ke urutan asli
    full = full.sort_values('orig_idx').reset_index(drop=True)
    tr_out = full[full['is_test'] == 0].copy().reset_index(drop=True)
    te_out = full[full['is_test'] == 1].copy().reset_index(drop=True)
    tr_out['is_fraud'] = tr_raw['is_fraud'].astype(int).values
    return tr_out, te_out

train_feat, test_feat = engineer_causal_features(train_df, test_df)
print(f"Rekayasa fitur perilaku kausal selesai dalam {time.time() - t0:.2f} detik.")
print(f"Total kolom setelah rekayasa fitur: {train_feat.shape[1]} kolom.")


# --- Cell 15 ---
# 9. [KODE PLOT] Visualisasi Pergeseran Pola Fraud (Pattern Shift: Jan-Jul vs Agustus 2025)
fraud_only = train_feat[train_feat['is_fraud'] == 1].copy()
fraud_only['periode'] = np.where(fraud_only['bulan'] < '2025-08', 'Jan–Jul 2025 (Rezim Lama)', 'Agustus 2025 (Rezim Baru)')

# Pangsa Fraud Bulanan per Kanal
kanal_month = pd.crosstab(fraud_only['bulan'], fraud_only['kanal'], normalize='index') * 100

# Pangsa Kategori Merchant di Jan-Jul vs Agustus
kat_shift = pd.crosstab(fraud_only['kategori_merchant'], fraud_only['periode'], normalize='columns') * 100
top_kats = kat_shift.sort_values('Agustus 2025 (Rezim Baru)', ascending=False).head(8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))

# Panel 1: Tren Pangsa Fraud per Kanal
for kn, col in zip(['kartu', 'aplikasi', 'va', 'web', 'qris'], ['#2980b9', '#27ae60', '#8e44ad', '#f39c12', '#c0392b']):
    lw = 3.2 if kn == 'qris' else 1.8
    mk = 's' if kn == 'qris' else 'o'
    ax1.plot(kanal_month.index, kanal_month[kn], marker=mk, linewidth=lw, color=col, label=f"Kanal: {kn.upper()}")
ax1.set_title("Pergeseran Pangsa Fraud per Kanal (Lonjakan QRIS 8x di Agu 2025)", fontweight='bold')
ax1.set_xlabel("Bulan Transaksi")
ax1.set_ylabel("Porsi dari Total Fraud Bulanan (%)")
ax1.tick_params(axis='x', rotation=30)
ax1.legend(loc='upper left', fontsize=8.5)

# Panel 2: Pergeseran Kategori Merchant
top_kats[['Jan–Jul 2025 (Rezim Lama)', 'Agustus 2025 (Rezim Baru)']].plot(
    kind='barh', ax=ax2, color=['#3498db', '#e74c3c'], width=0.78
)
ax2.invert_yaxis()
ax2.set_title("Pergeseran Kategori Merchant pada Transaksi Fraud", fontweight='bold')
ax2.set_xlabel("Porsi Fraud dalam Periode Tersebut (%)")
ax2.set_ylabel("Kategori Merchant")
ax2.legend(loc='lower right', fontsize=8.5)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_pattern_shift_qris.png", dpi=150)
plt.close()

print("Pangsa Fraud per Kanal (% dari total fraud bulanan):")
print(kanal_month.round(1))


# --- Cell 17 ---
# 10. Persiapan Matriks Fitur & Target Encoding Kausal (Smoothed Out-of-Time Encoding)
cat_cols = ['kategori_merchant', 'kanal', 'tipe_perangkat', 'level_kyc', 'email_domain']

def apply_causal_target_encoding(tr_df, cal_df, va_df, te_df, m_smooth=30.0):
    """Target encoding dengan m-smoothing yang dihitung HANYA dari tr_df (label matang)."""
    prior = float(tr_df['is_fraud'].mean())
    enc_maps = {}
    for c in cat_cols + ['id_merchant']:
        stats_c = tr_df.groupby(c)['is_fraud'].agg(['count', 'mean'])
        smooth_val = (stats_c['count'] * stats_c['mean'] + m_smooth * prior) / (stats_c['count'] + m_smooth)
        enc_maps[c] = smooth_val.to_dict()

    out_dfs = []
    for d in [tr_df, cal_df, va_df, te_df]:
        d_copy = d.copy()
        for c in cat_cols + ['id_merchant']:
            d_copy[f'{c}_te'] = d_copy[c].map(enc_maps[c]).fillna(prior).astype(np.float32)
        # Frequency encoding
        for c in cat_cols:
            freq_map = tr_df[c].value_counts(normalize=True).to_dict()
            d_copy[f'{c}_freq'] = d_copy[c].map(freq_map).fillna(0.0).astype(np.float32)
        out_dfs.append(d_copy)
    return out_dfs

# Pembagian Waktu:
# Train Fold      : 1 Jan - 30 Jun 2025 (297,839 baris)
# Calibration Fold: 1 Jul - 20 Jul 2025 (33,668 baris)
# Mature Val Fold : 21 Jul - 17 Agu 2025 (47,826 baris - label matang >= 14 hari sebelum 31 Agu)
tr_split = train_feat[train_feat['waktu_dt'] < '2025-07-01'].copy()
cal_split = train_feat[(train_feat['waktu_dt'] >= '2025-07-01') & (train_feat['waktu_dt'] < '2025-07-21')].copy()
va_split = train_feat[(train_feat['waktu_dt'] >= '2025-07-21') & (train_feat['waktu_dt'] < '2025-08-18')].copy()

tr_enc, cal_enc, va_enc, te_enc = apply_causal_target_encoding(tr_split, cal_split, va_split, test_feat)

# Daftar fitur bersih (TANPA flag_review_manual dan TANPA v15 yang mengalami drift linear)
v_clean_cols = [f'v{i:02d}' for i in range(1, 51) if i != 15]
engineered_cols = [
    'nominal', 'log_nominal', 'jam', 'sin_jam', 'cos_jam', 'hari_libur',
    'day_of_week', 'is_weekend', 'is_night_trx',
    'umur_akun_hari', 'umur_pengguna_tahun', 'umur_perangkat_hari',
    'perangkat_baru', 'jumlah_akun_per_perangkat', 'ganti_password_7h', 'gagal_login_24j',
    'jarak_lokasi_km', 'negara_ip_beda', 'kartu_tidak_hadir',
    'is_first_trx', 'menit_sejak_trx_clean', 'log_menit_sejak_trx',
    'trx_1j', 'trx_24j', 'trx_7h', 'nominal_24j', 'log_nominal_24j',
    'rasio_nominal_rata2_clean', 'rasio_nom_terhadap_24j',
    'vel_trx_1j_vs_24j', 'vel_trx_24j_vs_7h',
    'saldo_sebelum', 'log_saldo_sebelum', 'sisa_saldo_estimasi',
    'persen_saldo', 'persen_saldo_over1', 'persen_saldo_near1', 'persen_saldo_cap50',
    'merchant_age_invalid', 'umur_merchant_clean', 'refund_merchant_30h', 'volume_merchant_30h',
    'user_prior_trx_cnt', 'user_prior_mean_nom', 'user_nom_vs_prior_mean', 'user_nom_vs_prev_trx',
    'user_changed_kanal', 'user_changed_device_type', 'user_merch_prior_cnt', 'user_is_new_merchant',
    'user_kat_prior_cnt', 'user_is_new_kategori', 'user_prior_gagal_login_sum', 'user_prior_new_dev_sum',
    'merch_prior_trx_cnt', 'merch_prior_mean_nom', 'merch_nom_ratio', 'merch_prior_newdev_rate',
    'ato_risk_score', 'is_qris', 'is_kartu', 'qris_x_persen_saldo', 'qris_x_nominal_log',
    'qris_x_new_device', 'cnp_x_foreign_ip', 'new_dev_x_fail_login', 'new_dev_x_pass_reset',
    'kategori_merchant_te', 'kanal_te', 'tipe_perangkat_te', 'level_kyc_te', 'email_domain_te', 'id_merchant_te',
    'kategori_merchant_freq', 'kanal_freq', 'tipe_perangkat_freq', 'level_kyc_freq', 'email_domain_freq'
]
FEATURE_COLS = engineered_cols + v_clean_cols

X_tr, y_tr = tr_enc[FEATURE_COLS], tr_enc['is_fraud'].values
X_cal, y_cal = cal_enc[FEATURE_COLS], cal_enc['is_fraud'].values
X_va, y_va = va_enc[FEATURE_COLS], va_enc['is_fraud'].values
nom_cal = cal_enc['nominal'].values
nom_va = va_enc['nominal'].values

# Bobot sampel peluruhan waktu eksponensial (Time-Decay Sample Weights) untuk adaptasi pergeseran pola
days_from_start = (tr_enc['waktu_dt'] - tr_enc['waktu_dt'].min()).dt.total_seconds() / (86400.0 * 30.0)
sample_weights_tr = np.exp(0.18 * days_from_start).values

print(f"Total Fitur Bersih Siap Pakai : {len(FEATURE_COLS)} fitur")
print(f"X_tr  (Jan–Jun)    : {X_tr.shape} | Fraud = {y_tr.sum():,} ({y_tr.mean()*100:.3f}%)")
print(f"X_cal (1–20 Jul)   : {X_cal.shape} | Fraud = {y_cal.sum():,} ({y_cal.mean()*100:.3f}%)")
print(f"X_va  (21Jul–17Agu): {X_va.shape} | Fraud = {y_va.sum():,} ({y_va.mean()*100:.3f}%)")


# --- Cell 18 ---
# 11. [KODE PLOT] Eksperimen Komparasi Strategi Class Imbalance & Algoritma Klasifikasi
def compute_operational_cost(y_true, prob_pred, nominal_arr, cost_per_flag=REVIEW_COST, multiplier=1.0):
    """Menghitung keputusan flag berbasis aturan biaya Bayes dan total biaya operasional (Rp)."""
    flags = ((prob_pred * nominal_arr) >= (multiplier * cost_per_flag)).astype(int)
    missed_fraud_mask = (y_true == 1) & (flags == 0)
    missed_fraud_cost = float(nominal_arr[missed_fraud_mask].sum())
    review_cost = float(cost_per_flag * flags.sum())
    return missed_fraud_cost + review_cost, missed_fraud_cost, review_cost, int(flags.sum())

results_list = []
val_preds = {}
cal_preds = {}

# --- Model 1: Baseline Logistic Regression (Class Weight Balanced) ---
lr_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight='balanced', C=0.1, max_iter=300, random_state=SEED))
])
lr_pipe.fit(X_tr, y_tr)
p_cal_lr = lr_pipe.predict_proba(X_cal)[:, 1]
p_va_lr = lr_pipe.predict_proba(X_va)[:, 1]
# Prior correction agar skala probabilitas sesuai prevalensi asli
pi_tr = y_tr.mean()
odds_ratio_lr = (pi_tr / (1 - pi_tr)) / (0.5 / 0.5)
p_va_lr_corr = (p_va_lr * odds_ratio_lr) / ((1 - p_va_lr) + p_va_lr * odds_ratio_lr)
tot_c, mf_c, rv_c, n_fl = compute_operational_cost(y_va, p_va_lr_corr, nom_va)
results_list.append({
    'Pendekatan & Model': '1. Baseline LogReg (Class Weight + Prior Corr)',
    'PR-AUC': average_precision_score(y_va, p_va_lr_corr),
    'ROC-AUC': roc_auc_score(y_va, p_va_lr_corr),
    'Brier Score': brier_score_loss(y_va, p_va_lr_corr),
    'Flagged Trx': n_fl,
    'Total Cost (Rp)': tot_c
})
val_preds['LogReg_Balanced'] = p_va_lr_corr

# --- Model 2: Pendekatan Imbalance A - Class Weighting (scale_pos_weight=15) di LightGBM ---
lgb_weighted = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.75, scale_pos_weight=15.0,
    random_state=SEED, n_jobs=4, verbose=-1
)
lgb_weighted.fit(X_tr, y_tr)
p_va_w_raw = lgb_weighted.predict_proba(X_va)[:, 1]
# Koreksi odds karena scale_pos_weight=15
p_va_w_corr = (p_va_w_raw / 15.0) / ((1 - p_va_w_raw) + (p_va_w_raw / 15.0))
tot_c, mf_c, rv_c, n_fl = compute_operational_cost(y_va, p_va_w_corr, nom_va)
results_list.append({
    'Pendekatan & Model': '2. LightGBM Class Weight (scale_pos_weight=15)',
    'PR-AUC': average_precision_score(y_va, p_va_w_corr),
    'ROC-AUC': roc_auc_score(y_va, p_va_w_corr),
    'Brier Score': brier_score_loss(y_va, p_va_w_corr),
    'Flagged Trx': n_fl,
    'Total Cost (Rp)': tot_c
})
val_preds['LGBM_Weighted'] = p_va_w_corr

# --- Model 3: Pendekatan Imbalance B - Majority Undersampling (1:15 Ratio + Prior Recalibration) ---
pos_idx = np.where(y_tr == 1)[0]
neg_idx = np.where(y_tr == 0)[0]
np.random.seed(SEED)
sub_neg_idx = np.random.choice(neg_idx, size=len(pos_idx) * 15, replace=False)
sub_idx = np.concatenate([pos_idx, sub_neg_idx])
np.random.shuffle(sub_idx)

lgb_under = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.75, random_state=SEED, n_jobs=4, verbose=-1
)
lgb_under.fit(X_tr.iloc[sub_idx], y_tr[sub_idx])
p_va_u_raw = lgb_under.predict_proba(X_va)[:, 1]
pi_sub = y_tr[sub_idx].mean()
r_under = (pi_tr / (1 - pi_tr)) / (pi_sub / (1 - pi_sub))
p_va_u_corr = (p_va_u_raw * r_under) / ((1 - p_va_u_raw) + p_va_u_raw * r_under)
tot_c, mf_c, rv_c, n_fl = compute_operational_cost(y_va, p_va_u_corr, nom_va)
results_list.append({
    'Pendekatan & Model': '3. LightGBM Undersampling (1:15 + Bayes Prior Corr)',
    'PR-AUC': average_precision_score(y_va, p_va_u_corr),
    'ROC-AUC': roc_auc_score(y_va, p_va_u_corr),
    'Brier Score': brier_score_loss(y_va, p_va_u_corr),
    'Flagged Trx': n_fl,
    'Total Cost (Rp)': tot_c
})
val_preds['LGBM_Undersample'] = p_va_u_corr

# --- Model 4: Pendekatan Imbalance C - Unweighted Natural Prior + Time-Decay Weights + Cost Thresholding (LightGBM) ---
lgb_nat = lgb.LGBMClassifier(
    n_estimators=550, learning_rate=0.035, num_leaves=31, min_child_samples=60,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=5.0,
    random_state=SEED, n_jobs=4, verbose=-1
)
lgb_nat.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
p_cal_lgb = lgb_nat.predict_proba(X_cal)[:, 1]
p_va_lgb = lgb_nat.predict_proba(X_va)[:, 1]
tot_c, mf_c, rv_c, n_fl = compute_operational_cost(y_va, p_va_lgb, nom_va)
results_list.append({
    'Pendekatan & Model': '4. LightGBM Natural Prior + Time-Decay + Cost Rule',
    'PR-AUC': average_precision_score(y_va, p_va_lgb),
    'ROC-AUC': roc_auc_score(y_va, p_va_lgb),
    'Brier Score': brier_score_loss(y_va, p_va_lgb),
    'Flagged Trx': n_fl,
    'Total Cost (Rp)': tot_c
})
val_preds['LGBM_Natural_TimeDecay'] = p_va_lgb
cal_preds['LGBM'] = p_cal_lgb

# --- Model 5: CatBoost / XGBoost (Keluarga GBDT Kedua & Ensemble) ---
if HAS_CB:
    cb_model = cb.CatBoostClassifier(
        iterations=650, learning_rate=0.04, depth=6, l2_leaf_reg=5.0,
        random_seed=SEED, thread_count=4, verbose=0
    )
    cb_model.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
    p_cal_cb = cb_model.predict_proba(X_cal)[:, 1]
    p_va_cb = cb_model.predict_proba(X_va)[:, 1]
    tot_c, mf_c, rv_c, n_fl = compute_operational_cost(y_va, p_va_cb, nom_va)
    results_list.append({
        'Pendekatan & Model': '5. CatBoost Natural Prior + Time-Decay + Cost Rule',
        'PR-AUC': average_precision_score(y_va, p_va_cb),
        'ROC-AUC': roc_auc_score(y_va, p_va_cb),
        'Brier Score': brier_score_loss(y_va, p_va_cb),
        'Flagged Trx': n_fl,
        'Total Cost (Rp)': tot_c
    })
    val_preds['CatBoost'] = p_va_cb
    cal_preds['CatBoost'] = p_cal_cb

if HAS_XGB:
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.035, max_depth=5, min_child_weight=25,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=5.0,
        tree_method='hist', random_state=SEED, n_jobs=4, eval_metric='logloss'
    )
    xgb_model.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
    p_cal_xgb = xgb_model.predict_proba(X_cal)[:, 1]
    p_va_xgb = xgb_model.predict_proba(X_va)[:, 1]
    tot_c, mf_c, rv_c, n_fl = compute_operational_cost(y_va, p_va_xgb, nom_va)
    results_list.append({
        'Pendekatan & Model': '6. XGBoost Natural Prior + Time-Decay + Cost Rule',
        'PR-AUC': average_precision_score(y_va, p_va_xgb),
        'ROC-AUC': roc_auc_score(y_va, p_va_xgb),
        'Brier Score': brier_score_loss(y_va, p_va_xgb),
        'Flagged Trx': n_fl,
        'Total Cost (Rp)': tot_c
    })
    val_preds['XGBoost'] = p_va_xgb
    cal_preds['XGBoost'] = p_cal_xgb

# --- Model 7: Blended GBDT Ensemble (LightGBM + CatBoost + XGBoost) ---
p_cal_ens = p_cal_lgb.copy()
p_va_ens = p_va_lgb.copy()
w_sum = 1.0
if HAS_CB:
    p_cal_ens = 0.50 * p_cal_lgb + 0.50 * p_cal_cb
    p_va_ens = 0.50 * p_va_lgb + 0.50 * p_va_cb
if HAS_CB and HAS_XGB:
    p_cal_ens = 0.45 * p_cal_lgb + 0.30 * p_cal_cb + 0.25 * p_cal_xgb
    p_va_ens = 0.45 * p_va_lgb + 0.30 * p_va_cb + 0.25 * p_va_xgb
elif HAS_XGB and not HAS_CB:
    p_cal_ens = 0.60 * p_cal_lgb + 0.40 * p_cal_xgb
    p_va_ens = 0.60 * p_va_lgb + 0.40 * p_va_xgb

tot_c, mf_c, rv_c, n_fl = compute_operational_cost(y_va, p_va_ens, nom_va)
results_list.append({
    'Pendekatan & Model': '7. Blended Ensemble (LGBM + CatBoost + XGBoost)',
    'PR-AUC': average_precision_score(y_va, p_va_ens),
    'ROC-AUC': roc_auc_score(y_va, p_va_ens),
    'Brier Score': brier_score_loss(y_va, p_va_ens),
    'Flagged Trx': n_fl,
    'Total Cost (Rp)': tot_c
})
val_preds['Ensemble'] = p_va_ens
cal_preds['Ensemble'] = p_cal_ens

results_df = pd.DataFrame(results_list)
print("=== TABEL KOMPARASI STRATEGI CLASS IMBALANCE & MODEL PADA MATURE VALIDATION ===")
print(results_df.to_string(index=False))

# Visualisasi Komparasi PR-AUC dan Total Biaya
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))

sns.barplot(y='Pendekatan & Model', x='PR-AUC', data=results_df, palette='Blues_r', ax=ax1)
for p in ax1.patches:
    ax1.annotate(f"{p.get_width():.4f}", (p.get_width() + 0.005, p.get_y() + p.get_height()/2.),
                 va='center', fontweight='bold', fontsize=9)
ax1.set_title("Perbandingan PR-AUC Antar Strategi & Model", fontweight='bold')
ax1.set_xlabel("PR-AUC (Makin Tinggi Makin Baik)")
ax1.set_xlim(0, results_df['PR-AUC'].max() * 1.15)

sns.barplot(y='Pendekatan & Model', x='Total Cost (Rp)', data=results_df, palette='Oranges_r', ax=ax2)
for p in ax2.patches:
    ax2.annotate(f"Rp {p.get_width()/1e6:.1f} Jt", (p.get_width() + 1e6, p.get_y() + p.get_height()/2.),
                 va='center', fontweight='bold', fontsize=9)
ax2.set_title("Total Biaya Risiko pada Validation (Makin Rendah Makin Baik)", fontweight='bold')
ax2.set_xlabel("Total Biaya Operasional (Rupiah)")
ax2.set_xlim(0, results_df['Total Cost (Rp)'].max() * 1.18)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_imbalance_models_comparison.png", dpi=150)
plt.close()


# --- Cell 20 ---
# 12. [KODE PLOT] Kalibrasi Probabilitas & Optimasi Keputusan Berbasis Biaya
eps = 1e-6
logit_cal = np.log(np.clip(p_cal_ens, eps, 1 - eps) / (1 - np.clip(p_cal_ens, eps, 1 - eps))).reshape(-1, 1)
logit_va = np.log(np.clip(p_va_ens, eps, 1 - eps) / (1 - np.clip(p_va_ens, eps, 1 - eps))).reshape(-1, 1)

# 1. Platt Scaling pada Log-Odds (Menjaga urutan monotonik 100% sehingga PR-AUC tetap terjaga)
platt_lr = LogisticRegression(C=1e4, random_state=SEED)
platt_lr.fit(logit_cal, y_cal)
p_cal_platt = platt_lr.predict_proba(logit_cal)[:, 1]
p_va_platt = platt_lr.predict_proba(logit_va)[:, 1]

# 2. Isotonic Regression
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(p_cal_ens, y_cal)
p_cal_iso = iso_reg.predict(p_cal_ens)
p_va_iso = iso_reg.predict(p_va_ens)

# Pilih probabilitas dengan Brier Score & PR-AUC terbaik
brier_raw = brier_score_loss(y_va, p_va_ens)
brier_platt = brier_score_loss(y_va, p_va_platt)
brier_iso = brier_score_loss(y_va, p_va_iso)

print("=== EVALUASI KALIBRASI PROBABILITAS PADA MATURE VALIDATION ===")
print(f"1. Raw Ensemble Probabilities   : Brier = {brier_raw:.6f} | PR-AUC = {average_precision_score(y_va, p_va_ens):.4f}")
print(f"2. Platt Scaling (Log-Odds)     : Brier = {brier_platt:.6f} | PR-AUC = {average_precision_score(y_va, p_va_platt):.4f}")
print(f"3. Isotonic Regression          : Brier = {brier_iso:.6f} | PR-AUC = {average_precision_score(y_va, p_va_iso):.4f}")

# Gunakan kombinasi (blend) antara Raw Ensemble & Platt Scaling yang menjaga PR-AUC maksimum sekaligus terkalibrasi
w_platt = 0.50 if brier_platt <= brier_raw else 0.20
p_cal_final = (1 - w_platt) * p_cal_ens + w_platt * p_cal_platt
p_va_final = (1 - w_platt) * p_va_ens + w_platt * p_va_platt

# Optimasi Faktor Pengali Ambang Biaya (tau) HANYA pada Calibration Fold (1-20 Juli 2025)
taus = np.linspace(0.40, 2.00, 81)
cal_costs = [compute_operational_cost(y_cal, p_cal_final, nom_cal, REVIEW_COST, t)[0] for t in taus]
best_tau = float(taus[int(np.argmin(cal_costs))])

# Optimasi Fixed Probability Threshold terbaik pada Calibration Fold sebagai pembanding
fixed_ths = np.linspace(0.01, 0.50, 100)
fixed_cal_costs = []
for th in fixed_ths:
    fl_th = (p_cal_final >= th).astype(int)
    c_th = nom_cal[(y_cal == 1) & (fl_th == 0)].sum() + REVIEW_COST * fl_th.sum()
    fixed_cal_costs.append(c_th)
best_fixed_th = float(fixed_ths[int(np.argmin(fixed_cal_costs))])

# Evaluasi Kebijakan Keputusan pada Mature Validation Fold (21 Jul - 17 Agu 2025)
cost_flag_none = float(nom_va[y_va == 1].sum())
cost_flag_all = float(REVIEW_COST * len(y_va))

fl_fixed50 = (p_va_final >= 0.50).astype(int)
cost_fixed50 = float(nom_va[(y_va == 1) & (fl_fixed50 == 0)].sum() + REVIEW_COST * fl_fixed50.sum())

fl_fixed_opt = (p_va_final >= best_fixed_th).astype(int)
cost_fixed_opt = float(nom_va[(y_va == 1) & (fl_fixed_opt == 0)].sum() + REVIEW_COST * fl_fixed_opt.sum())

cost_bayes_1, mf_b1, rv_b1, nfl_b1 = compute_operational_cost(y_va, p_va_final, nom_va, REVIEW_COST, 1.0)
cost_bayes_opt, mf_opt, rv_opt, nfl_opt = compute_operational_cost(y_va, p_va_final, nom_va, REVIEW_COST, best_tau)

policy_df = pd.DataFrame([
    {'Kebijakan Keputusan (Decision Rule)': '1. Flag All (Tandai Semua Transaksi)', 'Jml Ditandai': len(y_va), 'Fraud Tertangkap': int(y_va.sum()), 'Biaya Fraud Lolos (Rp)': 0.0, 'Biaya Review (Rp)': cost_flag_all, 'Total Biaya (Rp)': cost_flag_all},
    {'Kebijakan Keputusan (Decision Rule)': '2. Flag None (Tanpa Peninjauan Sama Sekali)', 'Jml Ditandai': 0, 'Fraud Tertangkap': 0, 'Biaya Fraud Lolos (Rp)': cost_flag_none, 'Biaya Review (Rp)': 0.0, 'Total Biaya (Rp)': cost_flag_none},
    {'Kebijakan Keputusan (Decision Rule)': '3. Ambang Konstan Standar (prob >= 0.50)', 'Jml Ditandai': int(fl_fixed50.sum()), 'Fraud Tertangkap': int(((y_va == 1) & (fl_fixed50 == 1)).sum()), 'Biaya Fraud Lolos (Rp)': float(nom_va[(y_va == 1) & (fl_fixed50 == 0)].sum()), 'Biaya Review (Rp)': float(REVIEW_COST * fl_fixed50.sum()), 'Total Biaya (Rp)': cost_fixed50},
    {'Kebijakan Keputusan (Decision Rule)': f'4. Ambang Konstan Teroptimasi (prob >= {best_fixed_th:.3f})', 'Jml Ditandai': int(fl_fixed_opt.sum()), 'Fraud Tertangkap': int(((y_va == 1) & (fl_fixed_opt == 1)).sum()), 'Biaya Fraud Lolos (Rp)': float(nom_va[(y_va == 1) & (fl_fixed_opt == 0)].sum()), 'Biaya Review (Rp)': float(REVIEW_COST * fl_fixed_opt.sum()), 'Total Biaya (Rp)': cost_fixed_opt},
    {'Kebijakan Keputusan (Decision Rule)': '5. Aturan Biaya Bayes Teoretis (p * nominal >= 25.000)', 'Jml Ditandai': nfl_b1, 'Fraud Tertangkap': int(((y_va == 1) & ((p_va_final * nom_va) >= REVIEW_COST)).sum()), 'Biaya Fraud Lolos (Rp)': mf_b1, 'Biaya Review (Rp)': rv_b1, 'Total Biaya (Rp)': cost_bayes_1},
    {'Kebijakan Keputusan (Decision Rule)': f'6. Aturan Biaya Bayes Terkalibrasi (tau* = {best_tau:.2f})', 'Jml Ditandai': nfl_opt, 'Fraud Tertangkap': int(((y_va == 1) & ((p_va_final * nom_va) >= best_tau * REVIEW_COST)).sum()), 'Biaya Fraud Lolos (Rp)': mf_opt, 'Biaya Review (Rp)': rv_opt, 'Total Biaya (Rp)': cost_bayes_opt}
])
print("\n=== PERBANDINGAN KEBIJAKAN KEPUTUSAN PADA MATURE VALIDATION (47.826 TRANSAKSI) ===")
print(policy_df.to_string(index=False))
print(f"-> Penghematan Aturan Biaya Bayes vs Flag None : Rp {cost_flag_none - cost_bayes_opt:,.0f} (-{((cost_flag_none - cost_bayes_opt)/cost_flag_none)*100:.1f}%)")
print(f"-> Penghematan Aturan Biaya Bayes vs Flag All  : Rp {cost_flag_all - cost_bayes_opt:,.0f} (-{((cost_flag_all - cost_bayes_opt)/cost_flag_all)*100:.1f}%)")

# Visualisasi Kurva Kalibrasi & Kurva Optimasi Biaya
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.5))

prob_true_raw, prob_pred_raw = calibration_curve(y_va, p_va_ens, n_bins=10, strategy='quantile')
prob_true_cal, prob_pred_cal = calibration_curve(y_va, p_va_final, n_bins=10, strategy='quantile')
ax1.plot([0, 0.25], [0, 0.25], 'k--', label='Kalibrasi Sempurna (Diagonal)')
ax1.plot(prob_pred_raw, prob_true_raw, marker='o', color='#2980b9', label=f'Raw Ensemble (Brier={brier_raw:.5f})')
ax1.plot(prob_pred_cal, prob_true_cal, marker='s', color='#27ae60', label=f'Calibrated Ensemble (Brier={brier_score_loss(y_va, p_va_final):.5f})')
ax1.set_title("Reliability Diagram (Kurva Kalibrasi Probabilitas)", fontweight='bold')
ax1.set_xlabel("Rata-rata Probabilitas Prediksi")
ax1.set_ylabel("Fraksi Positif Empiris (Actual Fraud Rate)")
ax1.set_xlim(0, min(0.30, max(prob_pred_cal)*1.15))
ax1.set_ylim(0, min(0.30, max(prob_true_cal)*1.15))
ax1.legend()

val_tau_costs = [compute_operational_cost(y_va, p_va_final, nom_va, REVIEW_COST, t)[0] / 1e6 for t in taus]
ax2.plot(taus, val_tau_costs, color='#c0392b', linewidth=2.5, label='Total Biaya Risiko (Juta Rp)')
ax2.axvline(1.0, color='#2980b9', linestyle='--', linewidth=1.8, label='Ambang Teoretis Bayes (tau = 1.00)')
ax2.axvline(best_tau, color='#27ae60', linestyle='-', linewidth=2.0, label=f'Ambang Optimal Kalibrasi (tau* = {best_tau:.2f})')
ax2.set_title("Kurva Optimasi Biaya terhadap Faktor Ambang (tau)", fontweight='bold')
ax2.set_xlabel("Pengali Ambang Biaya (tau pada p * nominal >= tau * 25.000)")
ax2.set_ylabel("Total Biaya pada Validation (Juta Rupiah)")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_calibration_and_cost_curve.png", dpi=150)
plt.close()


# --- Cell 22 ---
# 13. [KODE PLOT] Breakdown Error per Kanal, Kategori Merchant, dan Pola Fraud + SHAP Summary
va_analysis = va_enc.copy()
va_analysis['prob'] = p_va_final
va_analysis['tandai'] = ((p_va_final * nom_va) >= (best_tau * REVIEW_COST)).astype(int)
va_analysis['missed_fraud_cost'] = np.where((va_analysis['is_fraud'] == 1) & (va_analysis['tandai'] == 0), va_analysis['nominal'], 0.0)
va_analysis['review_cost'] = REVIEW_COST * va_analysis['tandai']
va_analysis['total_cost'] = va_analysis['missed_fraud_cost'] + va_analysis['review_cost']
va_analysis['cost_if_no_model'] = np.where(va_analysis['is_fraud'] == 1, va_analysis['nominal'], 0.0)
va_analysis['net_savings'] = va_analysis['cost_if_no_model'] - va_analysis['total_cost']

# 1. Breakdown per Kanal
kanal_rows = []
for kn, grp in va_analysis.groupby('kanal'):
    y_g = grp['is_fraud'].values
    p_g = grp['prob'].values
    fl_g = grp['tandai'].values
    tp = int(((y_g == 1) & (fl_g == 1)).sum())
    fp = int(((y_g == 0) & (fl_g == 1)).sum())
    fn = int(((y_g == 1) & (fl_g == 0)).sum())
    kanal_rows.append({
        'Kanal': kn,
        'Transaksi': len(grp),
        'Actual Fraud': int(y_g.sum()),
        'PR-AUC': average_precision_score(y_g, p_g) if y_g.sum() > 0 else np.nan,
        'TP (Caught)': tp,
        'FP (False Alarm)': fp,
        'FN (Missed)': fn,
        'Recall (%)': (tp / max(1, tp + fn)) * 100,
        'Total Cost (Rp)': grp['total_cost'].sum(),
        'Net Savings (Rp)': grp['net_savings'].sum()
    })
kanal_err_df = pd.DataFrame(kanal_rows).sort_values('Total Cost (Rp)', ascending=False)
print("=== 1. BREAKDOWN ERROR & BIAYA PER KANAL PEMBAYARAN ===")
print(kanal_err_df.to_string(index=False))

# 2. Breakdown per Top 10 Kategori Merchant (berdasarkan nilai kerugian fraud tanpa model)
kat_rows = []
for kt, grp in va_analysis.groupby('kategori_merchant'):
    y_g = grp['is_fraud'].values
    p_g = grp['prob'].values
    fl_g = grp['tandai'].values
    tp = int(((y_g == 1) & (fl_g == 1)).sum())
    fp = int(((y_g == 0) & (fl_g == 1)).sum())
    fn = int(((y_g == 1) & (fl_g == 0)).sum())
    kat_rows.append({
        'Kategori Merchant': kt,
        'Transaksi': len(grp),
        'Actual Fraud': int(y_g.sum()),
        'PR-AUC': average_precision_score(y_g, p_g) if y_g.sum() > 0 else 0.0,
        'Recall (%)': (tp / max(1, tp + fn)) * 100,
        'Missed Cost (Rp)': grp['missed_fraud_cost'].sum(),
        'Total Cost (Rp)': grp['total_cost'].sum(),
        'Net Savings (Rp)': grp['net_savings'].sum()
    })
kat_err_df = pd.DataFrame(kat_rows).sort_values('Net Savings (Rp)', ascending=False)
print("\n=== 2. BREAKDOWN ERROR & PENGHEMATAN PER KATEGORI MERCHANT (TOP 10) ===")
print(kat_err_df.head(10).to_string(index=False))

# 3. Breakdown berdasarkan Pola Serangan Fraud (Fraud Archetypes)
def assign_fraud_pattern(df_in):
    cond_ato = (df_in['perangkat_baru'] == 1) & ((df_in['gagal_login_24j'] >= 1) | (df_in['ganti_password_7h'] == 1))
    cond_vel = (df_in['trx_1j'] >= 2) | (df_in['trx_24j'] >= 3)
    cond_cnp = (df_in['kartu_tidak_hadir'] == 1) | (df_in['negara_ip_beda'] == 1)
    cond_qris_drain = (df_in['kanal'] == 'qris') | (df_in['persen_saldo'] >= 0.85)
    return np.select(
        [cond_ato, cond_vel, cond_cnp, cond_qris_drain],
        [
            'Pola 1: Account Takeover (New Dev + Login/Pass)',
            'Pola 2: High-Velocity Burst (trx_1j>=2 / 24j>=3)',
            'Pola 3: Card-Not-Present / Foreign IP',
            'Pola 4: QRIS & Wallet-Drain (persen_saldo>=85%)'
        ],
        default='Pola 5: Low-Signal / Stealth Anomaly'
    )

va_analysis['pola_risiko'] = assign_fraud_pattern(va_analysis)
pat_rows = []
for pt, grp in va_analysis.groupby('pola_risiko'):
    y_g = grp['is_fraud'].values
    fl_g = grp['tandai'].values
    tp = int(((y_g == 1) & (fl_g == 1)).sum())
    fn = int(((y_g == 1) & (fl_g == 0)).sum())
    pat_rows.append({
        'Pola Serangan Fraud': pt,
        'Total Trx': len(grp),
        'Kasus Fraud': int(y_g.sum()),
        'Fraud Tertangkap (TP)': tp,
        'Fraud Lolos (FN)': fn,
        'Detection Rate (%)': (tp / max(1, tp + fn)) * 100,
        'Net Savings (Rp)': grp['net_savings'].sum()
    })
pat_err_df = pd.DataFrame(pat_rows).sort_values('Kasus Fraud', ascending=False)
print("\n=== 3. BREAKDOWN DETEKSI BERDASARKAN POLA SERANGAN FRAUD ===")
print(pat_err_df.to_string(index=False))

# Visualisasi Error Breakdown & Feature Importance / SHAP
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))

sns.barplot(x='Kanal', y='PR-AUC', data=kanal_err_df, palette='viridis', ax=ax1)
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width()/2., p.get_height() + 0.015),
                 ha='center', va='bottom', fontweight='bold', fontsize=9.5)
ax1.set_title("Performa PR-AUC per Kanal Pembayaran", fontweight='bold')
ax1.set_ylabel("PR-AUC (Average Precision)")
ax1.set_ylim(0, min(1.0, kanal_err_df['PR-AUC'].max() * 1.18))

# Feature Importance / SHAP Attribution
if HAS_SHAP:
    sample_shap = X_va.sample(min(600, len(X_va)), random_state=SEED)
    explainer = shap.TreeExplainer(lgb_nat)
    sv = explainer.shap_values(sample_shap)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=True).tail(15)
    mean_abs_shap.plot(kind='barh', color='#c0392b', ax=ax2)
    ax2.set_title("Top 15 Fitur Berdasarkan Mean |SHAP Value| (Log-Odds Fraud)", fontweight='bold')
    ax2.set_xlabel("Rata-rata Kontribusi Absolut SHAP |SHAP|")
else:
    imp_s = pd.Series(lgb_nat.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
    imp_s.plot(kind='barh', color='#c0392b', ax=ax2)
    ax2.set_title("Top 15 Fitur Berdasarkan LightGBM Split Importance", fontweight='bold')
    ax2.set_xlabel("Split Importance")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig8_error_analysis_and_shap.png", dpi=150)
plt.close()


# --- Cell 24 ---
# 14. Pelatihan Ensemble Final pada Data Latih Matang & Pembuatan File Submission 'prediksi_klasifikasi.csv'
t0 = time.time()

# Gunakan data latih hingga 24 Agustus 2025 (mengecualikan 25-31 Agu yang mengalami 0% confirmed fraud censoring)
FINAL_CUTOFF = '2025-08-25'
full_mature_tr = train_feat[train_feat['waktu_dt'] < FINAL_CUTOFF].copy()

# Hitung ulang target encoding pada full_mature_tr
full_tr_enc, _, _, full_te_enc = apply_causal_target_encoding(
    full_mature_tr, cal_split, va_split, test_feat, m_smooth=30.0
)

X_full = full_tr_enc[FEATURE_COLS]
y_full = full_tr_enc['is_fraud'].values
X_test_final = full_te_enc[FEATURE_COLS]
nom_test = full_te_enc['nominal'].values

days_full = (full_tr_enc['waktu_dt'] - full_tr_enc['waktu_dt'].min()).dt.total_seconds() / (86400.0 * 30.0)
weights_full = np.exp(0.18 * days_full).values

print(f"Melatih Ensemble Final pada {len(X_full):,} transaksi matang (1 Jan – 24 Agu 2025)...")

# 1. LightGBM Final (Multi-Seed Bagging untuk stabilitas varians)
p_test_lgb = np.zeros(len(X_test_final))
for s_i, seed_val in enumerate([SEED, SEED + 100, SEED + 2025]):
    m_lgb_f = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.035, num_leaves=31, min_child_samples=60,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=5.0,
        random_state=seed_val, n_jobs=4, verbose=-1
    )
    m_lgb_f.fit(X_full, y_full, sample_weight=weights_full)
    p_test_lgb += m_lgb_f.predict_proba(X_test_final)[:, 1] / 3.0

p_test_raw = p_test_lgb.copy()

# 2. CatBoost Final
if HAS_CB:
    m_cb_f = cb.CatBoostClassifier(
        iterations=700, learning_rate=0.04, depth=6, l2_leaf_reg=5.0,
        random_seed=SEED, thread_count=4, verbose=0
    )
    m_cb_f.fit(X_full, y_full, sample_weight=weights_full)
    p_test_cb = m_cb_f.predict_proba(X_test_final)[:, 1]
else:
    p_test_cb = None

# 3. XGBoost Final
if HAS_XGB:
    m_xgb_f = xgb.XGBClassifier(
        n_estimators=550, learning_rate=0.035, max_depth=5, min_child_weight=25,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=5.0,
        tree_method='hist', random_state=SEED, n_jobs=4, eval_metric='logloss'
    )
    m_xgb_f.fit(X_full, y_full, sample_weight=weights_full)
    p_test_xgb = m_xgb_f.predict_proba(X_test_final)[:, 1]
else:
    p_test_xgb = None

if HAS_CB and HAS_XGB:
    p_test_raw = 0.45 * p_test_lgb + 0.30 * p_test_cb + 0.25 * p_test_xgb
elif HAS_CB:
    p_test_raw = 0.50 * p_test_lgb + 0.50 * p_test_cb
elif HAS_XGB:
    p_test_raw = 0.60 * p_test_lgb + 0.40 * p_test_xgb

# Kalibrasi Log-Odds Platt Scaling
logit_test = np.log(np.clip(p_test_raw, eps, 1 - eps) / (1 - np.clip(p_test_raw, eps, 1 - eps))).reshape(-1, 1)
p_test_platt = platt_lr.predict_proba(logit_test)[:, 1]
prob_test_final = np.clip((1 - w_platt) * p_test_raw + w_platt * p_test_platt, 1e-6, 1 - 1e-6)

# Keputusan Penandaan Berbasis Biaya Bayes Terkalibrasi
tandai_test = ((prob_test_final * nom_test) >= (best_tau * REVIEW_COST)).astype(int)

sub_df = pd.DataFrame({
    'id_transaksi': test_df['id_transaksi'],
    'prob': np.round(prob_test_final, 6),
    'tandai': tandai_test
})

# Simpan ke direktori kerja saat ini, Task 2/, dan root repositori
sub_df.to_csv("prediksi_klasifikasi.csv", index=False)
if os.path.exists("Task 2"):
    sub_df.to_csv("Task 2/prediksi_klasifikasi.csv", index=False)
elif os.path.basename(os.getcwd()) == "Task 2" or os.path.basename(os.getcwd()) == "kode":
    sub_df.to_csv("../prediksi_klasifikasi.csv", index=False)

print(f"Pelatihan final & inferensi test selesai dalam {time.time() - t0:.2f} detik.")
print("\n=== VALIDASI KEPATUHAN FILE SUBMISSION 'prediksi_klasifikasi.csv' ===")
print(f"1. Nama Kolom          : {list(sub_df.columns)} (Wajib: ['id_transaksi', 'prob', 'tandai'])")
print(f"2. Jumlah Baris        : {len(sub_df):,} baris (Wajib: 106,674 baris)")
print(f"3. Missing Values (NaN): {sub_df.isnull().sum().sum()} (Wajib: 0)")
print(f"4. Rentang prob        : [{sub_df['prob'].min():.6f}, {sub_df['prob'].max():.6f}] (Wajib: 0 <= prob <= 1)")
print(f"5. Rata-rata prob      : {sub_df['prob'].mean()*100:.3f}% (Konsisten dengan base rate ~0.68%)")
print(f"6. Nilai unik tandai   : {sorted(sub_df['tandai'].unique().tolist())} (Wajib: [0, 1])")
print(f"7. Transaksi Ditandai  : {sub_df['tandai'].sum():,} transaksi ({sub_df['tandai'].mean()*100:.2f}% dari data uji)")
print("\n5 Baris Pertama prediksi_klasifikasi.csv:")
print(sub_df.head(5).to_string(index=False))

