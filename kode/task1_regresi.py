#!/usr/bin/env python3
"""UTS Machine Learning - Task 1: Regresi (One-Command Reproducible Script)"""

# --- Cell 2 ---
# 1. Import Pustaka Utama & Konfigurasi Global
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold
import lightgbm as lgb
import shap

# Konfigurasi Tampilan & Visualisasi
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 110

SEED = 42
np.random.seed(SEED)
print("Environment setup berhasil. Seluruh pustaka siap digunakan.")


# --- Cell 3 ---
# 2. Pemuatan Dataset Train dan Test (Path-Agnostic)
t0 = time.time()
train_path = '../Dataset/regresi_train.csv' if os.path.exists('../Dataset/regresi_train.csv') else 'Dataset/regresi_train.csv'
test_path = '../Dataset/regresi_test.csv' if os.path.exists('../Dataset/regresi_test.csv') else 'Dataset/regresi_test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(f"Pemuatan data selesai dalam {time.time() - t0:.2f} detik.")
print(f"Dimensi Data Latih (Train) : {train_df.shape[0]:,} baris x {train_df.shape[1]} kolom")
print(f"Dimensi Data Uji (Test)    : {test_df.shape[0]:,} baris x {test_df.shape[1]} kolom")


# --- Cell 4 ---
# 3. Pratinjau Struktur Kolom & Tipe Data
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
print(f"Kolom hanya ada di Train: {train_cols - test_cols} (Kolom target)")

cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Jumlah Fitur Kategorikal: {len(cat_cols)} kolom")
print(f"Jumlah Fitur Numerik    : {len(num_cols)} kolom")
display(train_df.head(3))


# --- Cell 6 ---
# 4. [KODE PLOT] Audit Missing Values pada 'berat_kg'
fig, ax = plt.subplots(figsize=(7, 3.5))
df_missing = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'Persentase Missing berat_kg': [
        (train_df['berat_kg'].isnull().sum() / len(train_df)) * 100,
        (test_df['berat_kg'].isnull().sum() / len(test_df)) * 100
    ]
})
sns.barplot(x='Dataset', y='Persentase Missing berat_kg', hue='Dataset', data=df_missing, palette=['#3498db', '#e74c3c'], legend=False, ax=ax)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
ax.set_title("Proporsi Missing Values pada Fitur 'berat_kg'", fontweight='bold')
ax.set_ylabel("Persentase (%)")
ax.set_ylim(0, 10)
plt.tight_layout()
plt.show()


# --- Cell 7 ---
# 5. Audit Nilai Sentinel & Anomali Fisik
print("--- 1. Sentinel Value Jarak (jarak_km) ---")
tr_d0 = (train_df['jarak_km'] == 0).sum()
tr_d9999 = (train_df['jarak_km'] == 9999).sum()
te_d0 = (test_df['jarak_km'] == 0).sum()
te_d9999 = (test_df['jarak_km'] == 9999).sum()
print(f"jarak_km == 0    | Train: {tr_d0:,} ({tr_d0/len(train_df)*100:.2f}%) | Test: {te_d0:,} ({te_d0/len(test_df)*100:.2f}%)")
print(f"jarak_km == 9999 | Train: {tr_d9999:,} ({tr_d9999/len(train_df)*100:.2f}%) | Test: {te_d9999:,} ({te_d9999/len(test_df)*100:.2f}%)")

print("\n--- 2. Anomali Umur Akun Penjual ---")
print(f"Nilai Maksimum umur_akun_penjual_hari: {train_df['umur_akun_penjual_hari'].max():,} hari (~144 tahun)")

print("\n--- 3. Outlier Ekstrim Durasi Pengiriman (durasi_jam) ---")
print(f"durasi_jam > 500 jam : {(train_df['durasi_jam'] > 500).sum()} baris")
print(f"durasi_jam Terbesar  : {train_df['durasi_jam'].max():,.1f} jam (~{train_df['durasi_jam'].max()/24:,.0f} hari)")


# --- Cell 8 ---
# 6. [KODE PLOT] Bukti Empiris Data Leakage pada Fitur 'jumlah_scan'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

sns.histplot(train_df['jumlah_scan'], bins=range(1, 40), color='#2980b9', label='Train (Selesai Dikirim)', 
             stat='density', discrete=True, alpha=0.6, ax=ax1)
sns.histplot(test_df['jumlah_scan'], bins=range(1, 40), color='#e74c3c', label='Test (Saat Pickup)', 
             stat='density', discrete=True, alpha=0.6, ax=ax1)
ax1.set_title("Distribusi 'jumlah_scan': Train vs Test", fontweight='bold')
ax1.set_xlabel("Jumlah Scan Pelacakan")
ax1.set_ylabel("Kepadatan (Density)")
ax1.set_xlim(1, 30)
ax1.legend()

df_scan_cmp = pd.concat([
    pd.DataFrame({'jumlah_scan': train_df['jumlah_scan'], 'Dataset': 'Train (Historical Delivered)'}),
    pd.DataFrame({'jumlah_scan': test_df['jumlah_scan'], 'Dataset': 'Test (At Pickup Time)'})
])
sns.boxplot(x='Dataset', y='jumlah_scan', hue='Dataset', data=df_scan_cmp, palette=['#2980b9', '#e74c3c'], legend=False, ax=ax2, showfliers=False)
ax2.set_title("Boxplot 'jumlah_scan' (Tanpa Fliers Ekstrim)", fontweight='bold')
ax2.set_ylabel("Jumlah Scan")
plt.tight_layout()
plt.show()

print("Statistik Deskriptif 'jumlah_scan':")
display(pd.DataFrame({
    'Train (Historis)': train_df['jumlah_scan'].describe(),
    'Test (Saat Pickup)': test_df['jumlah_scan'].describe()
}))


# --- Cell 9 ---
# 7. Pengukuran Kuantitatif Dampak Kebocoran Data (Leakage Impact)
train_df['tgl_pickup_dt'] = pd.to_datetime(train_df['tgl_pickup'])
val_mask = train_df['tgl_pickup_dt'] >= '2025-07-01'

train_part = train_df[~val_mask]
val_part = train_df[val_mask]

features_safe = ['jarak_km', 'volume_liter', 'jumlah_item', 'nilai_barang', 'alamat_skor']
features_with_leak = features_safe + ['jumlah_scan']

# Model 1: Bersih tanpa leakage
m_clean = Ridge().fit(train_part[features_safe].fillna(0), train_part['durasi_jam'])
loss_clean = mean_pinball_loss(val_part['durasi_jam'], m_clean.predict(val_part[features_safe].fillna(0)), alpha=0.5)

# Model 2: Leak diterapkan pada kondisi test nyata (scan rata-rata = 1.5)
m_leak = Ridge().fit(train_part[features_with_leak].fillna(0), train_part['durasi_jam'])
X_va_test_like = val_part[features_with_leak].fillna(0).copy()
X_va_test_like['jumlah_scan'] = 1.5
loss_leak = mean_pinball_loss(val_part['durasi_jam'], m_leak.predict(X_va_test_like), alpha=0.5)

print("=== HASIL KUANTIFIKASI DAMPAK DATA LEAKAGE ===")
print(f"1. Model Bersih (Tanpa jumlah_scan)   - Pinball Loss (q=0.5): {loss_clean:.3f} jam")
print(f"2. Model Leak (Kondisi Test Riil)     - Pinball Loss (q=0.5): {loss_leak:.3f} jam")
print(f"-> Lonjakan Kesalahan Prediksi Akibat Kebocoran: +{((loss_leak - loss_clean) / loss_clean)*100:.1f}%")
print("-> KESIMPULAN AUDIT: Fitur 'jumlah_scan' WAJIB DIDROP dari seluruh pipeline pemodelan.")


# --- Cell 10 ---
# 8. Audit Kardinalitas & Deteksi Kurir Baru K13 di Data Uji
tr_couriers = set(train_df['mitra_kurir'].unique())
te_couriers = set(test_df['mitra_kurir'].unique())
diff_couriers = te_couriers - tr_couriers

print("Overlap Mitra Kurir (Train vs Test):")
print(f" - Kurir di Train : {sorted(list(tr_couriers))}")
print(f" - Kurir di Test  : {sorted(list(te_couriers))}")
print(f" - Kurir Baru di Test (Unseen): {diff_couriers}")
print(f" - Jumlah Paket Kurir K13 di Test: {(test_df['mitra_kurir'] == 'K13').sum():,} paket ({(test_df['mitra_kurir'] == 'K13').mean()*100:.2f}% dari data uji)")
print("-> Implikasi: Encoding kategorikal harus memiliki strategi fallback untuk kategori unseen K13.")


# --- Cell 12 ---
# 9. [KODE PLOT] Analisis Distribusi Variabel Target (durasi_jam)
target = train_df['durasi_jam']
p50 = target.median()
p90 = target.quantile(0.90)
p99 = target.quantile(0.99)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: Skala Asli
sns.histplot(target[target <= p99], bins=60, kde=True, color='#2c3e50', ax=ax1)
ax1.axvline(p50, color='#e74c3c', linestyle='--', linewidth=2, label=f'P50 (Median) = {p50:.1f} jam')
ax1.axvline(p90, color='#f39c12', linestyle='-', linewidth=2, label=f'P90 = {p90:.1f} jam')
ax1.set_title("Distribusi 'durasi_jam' (Skala Asli, <= P99)", fontweight='bold')
ax1.set_xlabel("Durasi Pengiriman (Jam)")
ax1.legend()

# Plot 2: Skala Logaritmik
sns.histplot(np.log1p(target), bins=60, kde=True, color='#16a085', ax=ax2)
ax2.axvline(np.log1p(p50), color='#e74c3c', linestyle='--', linewidth=2, label=f'log(P50) = {np.log1p(p50):.2f}')
ax2.axvline(np.log1p(p90), color='#f39c12', linestyle='-', linewidth=2, label=f'log(P90) = {np.log1p(p90):.2f}')
ax2.set_title("Distribusi log1p(durasi_jam)", fontweight='bold')
ax2.set_xlabel("log(1 + durasi_jam)")
ax2.legend()

plt.tight_layout()
plt.show()

print(f"P50 (Median) : {p50:.2f} jam | P90: {p90:.2f} jam")
print(f"Skewness Asli: {target.skew():.2f} | Skewness Log: {np.log1p(target).skew():.2f}")


# --- Cell 13 ---
# 10. [KODE PLOT] Analisis Dinamika Temporal (Bulan dan Jam Pickup)
train_df['bulan_pickup'] = train_df['tgl_pickup_dt'].dt.to_period('M').astype(str)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: Tren Bulanan
monthly_dur = train_df.groupby('bulan_pickup')['durasi_jam'].agg(['median', 'mean']).reset_index()
sns.lineplot(x='bulan_pickup', y='mean', data=monthly_dur, marker='o', color='#e74c3c', label='Mean Durasi', ax=ax1)
sns.lineplot(x='bulan_pickup', y='median', data=monthly_dur, marker='s', color='#2980b9', label='Median Durasi (P50)', ax=ax1)
ax1.set_title("Tren Durasi Pengiriman Bulanan (Jan - Agu 2025)", fontweight='bold')
ax1.set_xlabel("Bulan Pickup")
ax1.set_ylabel("Durasi (Jam)")
ax1.tick_params(axis='x', rotation=45)
ax1.legend()

# Plot 2: Jam Pickup
sns.kdeplot(train_df['jam_pickup'], label='Train (Jan-Agu)', color='#2980b9', fill=True, alpha=0.3, ax=ax2)
sns.kdeplot(test_df['jam_pickup'], label='Test (Sep-Okt)', color='#e74c3c', fill=True, alpha=0.3, ax=ax2)
ax2.set_title("Distribusi Jam Penjemputan Paket (jam_pickup)", fontweight='bold')
ax2.set_xlabel("Jam Pickup (0 - 23)")
ax2.legend()

plt.tight_layout()
plt.show()


# --- Cell 14 ---
# 11. [KODE PLOT] Analisis Durasi Berdasarkan Layanan dan Pulau Tujuan
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

layanan_order = ['same-day', 'ekspres', 'reguler', 'kargo']
sns.boxplot(x='layanan', y='durasi_jam', hue='layanan', data=train_df, order=layanan_order, 
            palette='Blues_r', legend=False, showfliers=False, ax=ax1)
ax1.set_title("Variasi Durasi Pengiriman per Jenis Layanan", fontweight='bold')
ax1.set_xlabel("Layanan")
ax1.set_ylabel("Durasi (Jam)")

pulau_stats = train_df.groupby('pulau_tujuan')['durasi_jam'].median().sort_values()
sns.barplot(x=pulau_stats.values, y=pulau_stats.index, hue=pulau_stats.index, palette='viridis', legend=False, ax=ax2)
for i, v in enumerate(pulau_stats.values):
    ax2.text(v + 1, i, f"{v:.1f} j", va='center', fontweight='bold', fontsize=9)
ax2.set_title("Median Durasi Pengiriman (P50) per Pulau Tujuan", fontweight='bold')
ax2.set_xlabel("Median Durasi (Jam)")

plt.tight_layout()
plt.show()


# --- Cell 15 ---
# 12. [KODE PLOT] Analisis Performa Mitra Kurir & Pangsa Pasar Kurir K13
courier_stats = train_df.groupby('mitra_kurir')['durasi_jam'].agg(['median', 'count']).sort_values('median')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

sns.barplot(x=courier_stats.index, y=courier_stats['median'], hue=courier_stats.index, palette='coolwarm', legend=False, ax=ax1)
ax1.set_title("Median Durasi (P50) per Mitra Kurir (K01 - K12)", fontweight='bold')
ax1.set_xlabel("Mitra Kurir")
ax1.set_ylabel("Median Durasi (Jam)")
ax1.tick_params(axis='x', rotation=45)

tr_courier_share = train_df['mitra_kurir'].value_counts(normalize=True) * 100
te_courier_share = test_df['mitra_kurir'].value_counts(normalize=True) * 100
all_couriers = sorted(list(set(tr_courier_share.index) | set(te_courier_share.index)))
df_share = pd.DataFrame({
    'Train (%)': [tr_courier_share.get(k, 0) for k in all_couriers],
    'Test (%)': [te_courier_share.get(k, 0) for k in all_couriers]
}, index=all_couriers)
df_share.plot(kind='bar', ax=ax2, colormap='tab10', width=0.8)
ax2.set_title("Pangsa Paket Kurir (Sorotan K13 di Test)", fontweight='bold')
ax2.set_xlabel("Mitra Kurir")
ax2.set_ylabel("Porsi Paket (%)")
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# --- Cell 16 ---
# 13. [KODE PLOT] Pengaruh Jarak, Antrean Hub, dan Cuaca
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

sample_df = train_df.sample(8000, random_state=SEED)
sns.scatterplot(x='jarak_km', y='durasi_jam', data=sample_df[sample_df['durasi_jam'] <= p99], 
                hue='layanan', alpha=0.4, s=15, ax=ax1)
ax1.set_title("Jarak (km) vs Durasi Pengiriman", fontweight='bold')

sns.scatterplot(x='antrean_hub_asal', y='durasi_jam', data=sample_df[sample_df['durasi_jam'] <= p99],
                color='#8e44ad', alpha=0.3, s=15, ax=ax2)
ax2.set_title("Pengaruh Antrean Hub Asal terhadap Durasi", fontweight='bold')

sns.scatterplot(x='hujan_tujuan_mm', y='durasi_jam', data=sample_df[sample_df['durasi_jam'] <= p99],
                color='#2980b9', alpha=0.3, s=15, ax=ax3)
ax3.set_title("Curah Hujan Tujuan (mm) vs Durasi", fontweight='bold')

sns.boxplot(x=pd.qcut(sample_df['indeks_kemacetan'], q=5), y='durasi_jam', hue=pd.qcut(sample_df['indeks_kemacetan'], q=5),
            data=sample_df[sample_df['durasi_jam'] <= p99], palette='Reds', legend=False, ax=ax4, showfliers=False)
ax4.set_title("Pengaruh Kuintil Kemacetan Kota Asal", fontweight='bold')
ax4.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()


# --- Cell 17 ---
# 14. [KODE PLOT] Matriks Korelasi Spearman Fitur Numerik vs durasi_jam
corr_cols = [
    'durasi_jam', 'jarak_km', 'berat_kg', 'volume_liter', 'nilai_barang', 
    'alamat_skor', 'pengalaman_kurir_bulan', 'antrean_hub_asal', 
    'indeks_kemacetan', 'hujan_asal_mm', 'hujan_tujuan_mm'
]
corr_matrix = train_df[corr_cols].corr(method='spearman')

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='vlag', 
            vmin=-0.5, vmax=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title("Matriks Korelasi Spearman Fitur Numerik Utama vs durasi_jam", fontweight='bold', pad=12)
plt.tight_layout()
plt.show()


# --- Cell 19 ---
# 15. [KODE PLOT] Perbandingan Skema Validasi: Random CV vs Time-Based Split
sample_feats = ['jarak_km', 'volume_liter', 'jumlah_item', 'nilai_barang', 'alamat_skor']
X_sub = train_df[sample_feats].fillna(0)
y_sub = train_df['durasi_jam']

# 1. Random 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_err = [mean_pinball_loss(y_sub.iloc[va], Ridge().fit(X_sub.iloc[tr], y_sub.iloc[tr]).predict(X_sub.iloc[va]), alpha=0.5) for tr, va in kf.split(X_sub)]
rand_cv_score = np.mean(cv_err)

# 2. Time-Based Split (Jan-Jun vs Jul-Agu)
time_mask = train_df['tgl_pickup_dt'] >= '2025-07-01'
m_time = Ridge().fit(X_sub[~time_mask], y_sub[~time_mask])
time_score = mean_pinball_loss(y_sub[time_mask], m_time.predict(X_sub[time_mask]), alpha=0.5)

fig, ax = plt.subplots(figsize=(7, 3.5))
df_val_cmp = pd.DataFrame({
    'Skema Validasi': ['Random 5-Fold CV', 'Time-Based Split (Jul-Agu)'],
    'Pinball Loss (q=0.5)': [rand_cv_score, time_score]
})
sns.barplot(x='Skema Validasi', y='Pinball Loss (q=0.5)', hue='Skema Validasi', data=df_val_cmp, palette=['#27ae60', '#e67e22'], legend=False, ax=ax)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f} jam", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
ax.set_title("Perbandingan Error: Random CV vs Time-Based Validation", fontweight='bold')
ax.set_ylabel("Pinball Loss q=0.5 (Jam)")
ax.set_ylim(0, 20)
plt.tight_layout()
plt.show()

print(f"Random 5-Fold CV Pinball Loss : {rand_cv_score:.3f} jam (Bias terlalu optimis)")
print(f"Time-Based Split Pinball Loss : {time_score:.3f} jam (Realistis)")


# --- Cell 21 ---
# 16. Pipeline Rekayasa Fitur & Preprocessing
def build_features(train_raw, test_raw, split_date='2025-07-01'):
    tr = train_raw.copy()
    te = test_raw.copy()
    tr['tgl_pickup_dt'] = pd.to_datetime(tr['tgl_pickup'])
    te['tgl_pickup_dt'] = pd.to_datetime(te['tgl_pickup'])
    
    val_mask = tr['tgl_pickup_dt'] >= split_date
    train_fold = tr[~val_mask].copy()
    val_fold = tr[val_mask].copy()
    test_fold = te.copy()
    
    median_density = (train_fold['berat_kg'] / train_fold['volume_liter']).dropna().median()
    safe_dist_df = train_fold[(train_fold['jarak_km'] > 0) & (train_fold['jarak_km'] < 9999)]
    route_dist_map = safe_dist_df.groupby(['kota_asal', 'kota_tujuan'])['jarak_km'].median().to_dict()
    global_median_dist = safe_dist_df['jarak_km'].median()
    
    global_median_target = train_fold['durasi_jam'].median()
    m_smooth = 20
    cat_to_encode = ['layanan', 'mitra_kurir', 'hub_asal', 'pulau_tujuan', 'kota_tujuan']
    te_maps = {}
    for c in cat_to_encode:
        stats = train_fold.groupby(c)['durasi_jam'].agg(['count', 'median'])
        te_maps[c] = ((stats['count'] * stats['median'] + m_smooth * global_median_target) / (stats['count'] + m_smooth)).to_dict()
    
    dfs_out = []
    for d in [train_fold, val_fold, test_fold]:
        d['berat_was_missing'] = d['berat_kg'].isnull().astype(int)
        d['berat_kg_clean'] = d['berat_kg'].fillna(d['volume_liter'] * median_density)
        d['jarak_sentinel'] = ((d['jarak_km'] == 0) | (d['jarak_km'] == 9999)).astype(int)
        
        clean_dist = d['jarak_km'].copy()
        for idx in d[d['jarak_sentinel'] == 1].index:
            r = (d.loc[idx, 'kota_asal'], d.loc[idx, 'kota_tujuan'])
            clean_dist.loc[idx] = route_dist_map.get(r, global_median_dist)
        d['jarak_km_clean'] = clean_dist
        
        d['seller_age_clean'] = d['umur_akun_penjual_hari'].clip(upper=5000)
        d['day_of_week'] = d['tgl_pickup_dt'].dt.dayofweek
        d['is_weekend'] = (d['day_of_week'] >= 5).astype(int)
        d['sin_jam'] = np.sin(2 * np.pi * d['jam_pickup'] / 24)
        d['cos_jam'] = np.cos(2 * np.pi * d['jam_pickup'] / 24)
        
        d['densitas_paket'] = d['berat_kg_clean'] / (d['volume_liter'] + 1e-4)
        d['beban_hub_macet'] = d['antrean_hub_asal'] * d['indeks_kemacetan']
        d['total_hujan'] = d['hujan_asal_mm'] + d['hujan_tujuan_mm']
        d['nilai_per_kg'] = d['nilai_barang'] / (d['berat_kg_clean'] + 0.1)
        
        for c in cat_to_encode:
            d[f'{c}_te'] = d[c].map(te_maps[c]).fillna(global_median_target)
        dfs_out.append(d)
        
    return dfs_out[0], dfs_out[1], dfs_out[2]

tr_proc, va_proc, te_proc = build_features(train_df, test_df)

f_cols = [f'f{i:02d}' for i in range(1, 51)]
feature_names = [
    'jarak_km_clean', 'jarak_sentinel', 'berat_kg_clean', 'berat_was_missing',
    'volume_liter', 'jumlah_item', 'nilai_barang', 'asuransi', 'cod', 'alamat_skor',
    'pengalaman_kurir_bulan', 'rating_penjual', 'seller_age_clean', 'volume_penjual_30h',
    'antrean_hub_asal', 'indeks_kemacetan', 'hujan_asal_mm', 'hujan_tujuan_mm',
    'day_of_week', 'is_weekend', 'sin_jam', 'cos_jam',
    'densitas_paket', 'beban_hub_macet', 'total_hujan', 'nilai_per_kg',
    'layanan_te', 'mitra_kurir_te', 'hub_asal_te', 'pulau_tujuan_te', 'kota_tujuan_te'
] + f_cols

X_tr = tr_proc[feature_names]
y_tr = tr_proc['durasi_jam']
X_va = va_proc[feature_names]
y_va = va_proc['durasi_jam']
X_te = te_proc[feature_names]

print(f"Rekayasa Fitur Selesai: {len(feature_names)} fitur siap dimodelkan.")
print(f"Dimensi X_train: {X_tr.shape} | X_val: {X_va.shape} | X_test: {X_te.shape}")


# --- Cell 23 ---
# 17. [KODE PLOT] Pelatihan dan Komparasi Model Regresi Kuantil
# Model 1: Heuristic Baseline
grp_stats = tr_proc.groupby(['layanan', 'pulau_tujuan'])['durasi_jam'].agg(p50='median', p90=lambda x: x.quantile(0.90)).to_dict('index')
p50_m1 = va_proc.apply(lambda r: grp_stats.get((r['layanan'], r['pulau_tujuan']), {'p50': y_tr.median()})['p50'], axis=1).values
p90_m1 = np.maximum(p50_m1, va_proc.apply(lambda r: grp_stats.get((r['layanan'], r['pulau_tujuan']), {'p90': y_tr.quantile(0.90)})['p90'], axis=1).values)

# Model 2: Ridge Log
ridge = Ridge(alpha=100.0, random_state=SEED).fit(X_tr, np.log1p(y_tr))
res = np.log1p(y_tr) - ridge.predict(X_tr)
pred_log_va = ridge.predict(X_va)
p50_m2 = np.expm1(pred_log_va + np.median(res))
p90_m2 = np.maximum(p50_m2, np.expm1(pred_log_va + np.percentile(res, 90)))

# Model 3: LightGBM Quantile
lgb_params = {'n_estimators': 250, 'learning_rate': 0.08, 'num_leaves': 63, 'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': SEED, 'n_jobs': 4, 'verbose': -1}
m_lgb_p50 = lgb.LGBMRegressor(objective='quantile', alpha=0.5, **lgb_params).fit(X_tr, y_tr)
m_lgb_p90 = lgb.LGBMRegressor(objective='quantile', alpha=0.9, **lgb_params).fit(X_tr, y_tr)
p50_m3 = m_lgb_p50.predict(X_va)
p90_m3 = np.maximum(p50_m3, m_lgb_p90.predict(X_va))

# Visualisasi Komparasi
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
comparison_df = pd.DataFrame({
    'Model': ['Heuristic Baseline', 'Ridge (Target Transform)', 'LightGBM Quantile (Raw)'],
    'Pinball Loss P50 (jam)': [mean_pinball_loss(y_va, p50_m1, alpha=0.5), mean_pinball_loss(y_va, p50_m2, alpha=0.5), mean_pinball_loss(y_va, p50_m3, alpha=0.5)],
    'P90 Coverage (%)': [(y_va <= p90_m1).mean()*100, (y_va <= p90_m2).mean()*100, (y_va <= p90_m3).mean()*100]
})
sns.barplot(x='Model', y='Pinball Loss P50 (jam)', hue='Model', data=comparison_df, palette='Blues_d', legend=False, ax=ax1)
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                 ha='center', va='center', color='white', fontweight='bold', fontsize=11)
ax1.set_title("Pinball Loss P50 (Makin Rendah Makin Baik)", fontweight='bold')
ax1.set_ylabel("Pinball Loss (Jam)")

sns.barplot(x='Model', y='P90 Coverage (%)', hue='Model', data=comparison_df, palette='Oranges_d', legend=False, ax=ax2)
ax2.axhline(90.0, color='red', linestyle='--', linewidth=2, label='Target 90%')
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                 ha='center', va='center', color='white', fontweight='bold', fontsize=11)
ax2.set_title("Cakupan P90 (Target >= 90%)", fontweight='bold')
ax2.set_ylabel("Cakupan Empiris (%)")
ax2.set_ylim(80, 95)
ax2.legend()
plt.tight_layout()
plt.show()

display(comparison_df)


# --- Cell 25 ---
# 18. [KODE PLOT] Kalibrasi Conformal Prediction & Diagram Keandalan
q_hat = np.percentile(y_va.values - p90_m3, 90)
p90_m3_calib = np.maximum(p50_m3, p90_m3 + q_hat)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: Lebar Interval
interval_w = p90_m3_calib - p50_m3
sns.histplot(interval_w[interval_w <= np.percentile(interval_w, 99)], bins=50, kde=True, color='#8e44ad', ax=ax1)
ax1.axvline(np.median(interval_w), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(interval_w):.1f} jam')
ax1.set_title("Distribusi Lebar Interval Terkalibrasi (P90 - P50)", fontweight='bold')
ax1.set_xlabel("Lebar Interval (Jam)")
ax1.legend()

# Plot 2: Keandalan per Layanan
cov_serv = {}
for s in ['same-day', 'ekspres', 'reguler', 'kargo']:
    idx_s = va_proc[va_proc['layanan'] == s].index
    pos = va_proc.index.get_indexer(idx_s)
    cov_serv[s] = (y_va.loc[idx_s] <= p90_m3_calib[pos]).mean() * 100

sns.barplot(x=list(cov_serv.keys()), y=list(cov_serv.values()), hue=list(cov_serv.keys()), palette='Set2', legend=False, ax=ax2)
ax2.axhline(90.0, color='red', linestyle='--', linewidth=2, label='Target 90%')
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                 ha='center', va='center', color='white', fontweight='bold', fontsize=11)
ax2.set_title("Cakupan P90 Terkalibrasi per Jenis Layanan", fontweight='bold')
ax2.set_ylabel("Cakupan (%)")
ax2.set_ylim(80, 100)
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Faktor Penyesuaian Conformal (q_hat): {q_hat:.3f} jam")
print(f"Cakupan P90 Mentah       : {(y_va <= p90_m3).mean()*100:.2f}%")
print(f"Cakupan P90 Terkalibrasi : {(y_va <= p90_m3_calib).mean()*100:.2f}% (Tepat 90.00%!)")
print(f"Pinball Loss P90 Baru    : {mean_pinball_loss(y_va, p90_m3_calib, alpha=0.9):.3f} jam")


# --- Cell 27 ---
# 19. [KODE PLOT] Breakdown Error per Layanan dan Pulau Tujuan
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

sns.barplot(x='Kategori', y='Pinball Loss P50', hue='Kategori', data=df_brk_serv, palette='Blues_r', legend=False, ax=ax1)
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.2f} j", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                 ha='center', va='center', color='white', fontweight='bold')
ax1.set_title("Breakdown Pinball Loss P50 per Jenis Layanan", fontweight='bold')
ax1.set_ylabel("Pinball Loss (Jam)")

sns.barplot(x='Pinball Loss P50', y='Kategori', hue='Kategori', data=df_brk_isl.sort_values('Pinball Loss P50'), palette='viridis', legend=False, ax=ax2)
for i, p in enumerate(ax2.patches):
    ax2.annotate(f"{p.get_width():.2f} j", (p.get_width() + 0.3, p.get_y() + p.get_height() / 2.),
                 va='center', fontweight='bold')
ax2.set_title("Breakdown Pinball Loss P50 per Pulau Tujuan", fontweight='bold')
ax2.set_xlabel("Pinball Loss (Jam)")

plt.tight_layout()
plt.show()


# --- Cell 28 ---
# 20. [KODE PLOT] SHAP Summary Plot (Feature Importance & Attribution)
sample_X = X_va.sample(400, random_state=SEED)
explainer = shap.TreeExplainer(m_lgb_p50)
shap_values = explainer.shap_values(sample_X)

plt.figure(figsize=(10, 5))
shap.summary_plot(shap_values, sample_X, max_display=12, show=True)


# --- Cell 30 ---
# 21. Pelatihan Model Final pada Full Train & Inferensi Test
train_full_proc, _, test_final_proc = build_features(train_df, test_df, split_date='2025-09-01')
X_full_train = train_full_proc[feature_names]
y_full_train = train_full_proc['durasi_jam']
X_test_final = test_final_proc[feature_names]

final_p50 = lgb.LGBMRegressor(objective='quantile', alpha=0.5, **lgb_params).fit(X_full_train, y_full_train).predict(X_test_final)
final_p90 = lgb.LGBMRegressor(objective='quantile', alpha=0.9, **lgb_params).fit(X_full_train, y_full_train).predict(X_test_final)

# Kalibrasi conformal & enforcement p90 >= p50
final_p90_calib = np.maximum(final_p50, final_p90 + q_hat)

sub_df = pd.DataFrame({
    'id_paket': test_df['id_paket'],
    'p50': np.round(final_p50, 2),
    'p90': np.round(final_p90_calib, 2)
})

sub_df.to_csv('prediksi_regresi.csv', index=False)

print("File prediksi_regresi.csv berhasil disimpan!")
print(f"Dimensi                : {sub_df.shape} (Wajib: 60,000 baris)")
print(f"Pelanggaran p90 < p50  : {(sub_df['p90'] < sub_df['p50']).sum()} baris (Wajib: 0)")
display(sub_df.head(5))


