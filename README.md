# UTS Machine Learning - Kelompok 8

Repositori ini memuat implementasi *honest and reproducible Machine Learning pipelines* untuk Ujian Tengah Semester (UTS) Mata Kuliah Machine Learning, mencakup analisis skala besar (~940.000 total baris data) pada tiga domain permasalahan utama:

1. **Task 1: Delivery Time Estimation (Regresi Kuantil & Conformal Prediction)**
2. **Task 2: Fraud Detection (Klasifikasi Imbalanced & Optimasi Biaya)**
3. **Task 3: Customer Segmentation (Clustering, Stabilitas ARI & Deteksi Anomali)**

---

## 📁 Struktur Repositori

Sesuai instruksi pengumpulan tugas, repositori ditata dengan format baku berikut:

```text
├── Instruksi/
│   ├── kamus_data.md          # Definisi fitur dan batasan operasional data
│   └── UTS (...).docx         # Panduan teknis & rubrik penilaian UTS
├── Task 1/
│   ├── figures/               # Gambar grafik evaluasi resolusi tinggi Task 1
│   ├── Regresi.ipynb          # Notebook terstruktur Task 1 (10 tahap end-to-end)
│   └── prediksi_regresi.csv   # Salinan hasil prediksi Task 1
├── Task 2/
│   ├── figures/               # Gambar grafik evaluasi resolusi tinggi Task 2 (fig1–fig8)
│   ├── Klasifikasi.ipynb      # Notebook terstruktur Task 2 (10 tahap end-to-end)
│   └── prediksi_klasifikasi.csv # Salinan hasil prediksi Task 2
├── kode/
│   ├── README.md              # Petunjuk eksekusi mandiri untuk reproduksi angka
│   ├── task1_regresi.py       # Skrip satu-perintah untuk mereproduksi Task 1
│   └── task2_klasifikasi.py   # Skrip satu-perintah untuk mereproduksi Task 2
├── .gitignore                 # Konfigurasi ignore (data besar >100MB, venv, cache)
├── requirements.txt           # Daftar dependensi pustaka Python
├── prediksi_regresi.csv       # File submission Task 1 (id_paket, p50, p90)
├── prediksi_klasifikasi.csv   # [Task 2] Submission deteksi fraud
├── cluster.csv                # [Task 3] Submission segmentasi pelanggan
├── kontribusi.md              # Matriks kontribusi anggota kelompok
├── laporan.pdf                # Laporan tertulis resmi (maks 18 halaman)
└── README.md                  # Dokumentasi utama repositori ini
```

---

## 🚀 Setup Lingkungan & Instalasi

Pastikan menggunakan **Python 3.10+** (disarankan Python 3.12).

### 1. Kloning Repositori
```bash
git clone https://github.com/IkinInDisguise/UTS-Machine-Learning-Kel.-8.git
cd UTS-Machine-Learning-Kel.-8
```

### 2. Membuat Lingkungan Virtual & Dependensi
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Mengunduh Dataset
Dataset mentah (`data_uts_ml.zip`) diunduh melalui tautan resmi yang disediakan pada instruksi tugas, kemudian diekstrak ke dalam folder `Dataset/`:
- `Dataset/regresi_train.csv` (Januari–Agustus 2025, ~100 MB)
- `Dataset/regresi_test.csv` (September–Oktober 2025, ~24 MB)
- `Dataset/klasifikasi_train.csv` & `klasifikasi_test.csv`
- `Dataset/clustering.csv`

*(Catatan: Direktori `Dataset/` diabaikan oleh Git via `.gitignore` karena batasan ukuran file GitHub >100 MB).*

---

## 📊 Ringkasan Teknis Task 1: Estimasi Waktu Kedatangan Paket (Regresi)

### 1. Temuan Kritis Audit Data & Eliminasi Data Leakage
- **Target Leakage Fatal pada `jumlah_scan`**: Berdasarkan `kamus_data.md`, `jumlah_scan` adalah akumulasi riwayat pelacakan selama perjalanan. Pada data latih (paket selesai), rata-ratanya adalah **9,65 scan**, sedangkan pada data uji (paket baru di-pickup), rata-ratanya hanya **1,50 scan**. Memasukkan fitur ini mengakibatkan lonjakan error Pinball Loss sebesar **+69,6%** saat diuji pada kondisi operasional riil. **Fitur ini wajib dieliminasi secara total.**
- **Missing Value & Nilai Sentinel**: Fitur `berat_kg` memiliki missing value 5,56% di train dan 5,44% di test. Karena `volume_liter` 100% lengkap dengan densitas fisik stabil (median 0,25 kg/L), nilai missing diimputasi menggunakan regresi volume. Nilai sentinel pada `jarak_km` (`0` dan `9999`) diganti dengan jarak median per rute.
- **Deteksi Unseen Category**: Kurir baru **`K13`** muncul sebanyak 2.342 kali (3,90%) pada data uji tetapi tidak pernah ada di data latih. Digunakan *smoothed target encoding* ($m=20$) agar otomatis *fallback* ke median global tanpa menimbulkan kesalahan runtime.

### 2. Desain Skema Validasi
- **Kegagalan Random K-Fold CV**: Validasi acak 5-Fold menghasilkan Pinball Loss **14,10 jam** (optimisme semu $-13,6\%$) akibat autokorelasi temporal.
- **Skema Temporal Hold-Out**: Memisahkan Train (Jan–Jun 2025, 178.694 paket) dan Validation (Jul–Agu 2025, 61.306 paket) menghasilkan evaluasi realistis **16,32 jam** yang mencerminkan generalisasi data uji (Sep–Okt 2025).

### 3. Pemodelan Regresi Kuantil & Kalibrasi Conformal
Evaluasi dilakukan pada metrik resmi **Pinball Loss** untuk kuantil median ($P_{50}$) dan batas atas ketidakpastian 90% ($P_{90}$):

| Model | Pinball Loss P50 (jam) | Pinball Loss P90 (jam) | Cakupan Empiris P90 (%) |
|---|:---:|:---:|:---:|
| Baseline (Median Historis per Layanan) | 14,32 | 13,32 | 89,95% |
| Ridge Regression (Log-Target) | 11,81 | 11,69 | 88,40% |
| **LightGBM Quantile (Raw)** | **10,71** | 11,32 | 87,10% |
| **LightGBM + Split Conformal Prediction** | **10,71** | **11,26** | **90,00% (Target Terpenuhi)** |

- **Kalibrasi Conformal**: Penyesuaian non-parametrik sebesar $\hat{q} = +4,467\text{ jam}$ menjamin tepat **90,00%** paket tiba sebelum waktu janji $P_{90}$, dengan kepatuhan syarat wajib $P_{90} \ge P_{50}$ pada 100% baris prediksi.

---

## 🛡️ Ringkasan Teknis Task 2: Deteksi Transaksi Fraud (Klasifikasi & Optimasi Biaya Risiko)

### 1. Temuan Kritis Audit Label & Eliminasi Target Leakage
- **Post-Transaction Target Leakage pada `flag_review_manual`**: Fitur `flag_review_manual` bernilai `1` pada **79,90%** transaksi fraud terkonfirmasi di data latih ($2.095 / 2.622$ kasus, $r = +0,4878$) akibat dibukanya peninjauan manual secara retrospektif setelah laporan fraud/chargeback masuk (dengan rata-rata nominal **Rp 1.203.313** dan `trx_1j` = **0,897**), di atas sampel audit rutin acak sebesar **1,20%**. Sebaliknya, pada kondisi saat transaksi terjadi (`klasifikasi_test`), `flag_review_manual = 1` hanya muncul sebesar **1,23%** dengan rata-rata nominal **Rp 287.328** (identik dengan transaksi normal). Mempertahankan fitur bocor ini mengakibatkan anjloknya PR-AUC sebesar **-51,5%** dan pembengkakan biaya operasional **+80,9%** pada kondisi produksi riil. **Fitur ini wajib dihapus.**
- **Right-Censoring (Keterlambatan Konfirmasi Label) di Akhir Agustus 2025**: Data ditarik pada **31 Agustus 2025**. Prevalensi fraud bulanan stabil pada **0,680%** sepanjang Januari–Juli 2025, namun turun ke **0,463%** di Agustus 2025 dan jatuh menjadi **0,000% (0 fraud dari 4.780 transaksi)** pada **29–31 Agustus 2025** karena proses konfirmasi fraud membutuhkan ~10–14 hari.
- **Missing Value Struktural, Anomali & Covariate Drift `v15`**: Missing value pada `menit_sejak_trx_terakhir` dan `rasio_nominal_rata2` (**25,46%** di Train = $102.621$ baris, tepat sama dengan jumlah unik `id_pengguna` di Train) 100% merepresentasikan transaksi pertama pengguna (`is_first_trx = 1`). Sebanyak **955 baris** di Train dan **20 baris** di Test memiliki `umur_merchant_hari < 0` (hingga $-274$ hari). Selain itu, fitur anonim `v15` mengalami *temporal drift* linear tajam dari `0,216` (Jan) ke `2,345` (Agu) dan `2,732` (Test, $z = +1,24\sigma$) tanpa korelasi terhadap fraud sehingga wajib dieliminasi.

### 2. Skema Validasi Sadar-Maturitas Label & Pergeseran Pola (*Pattern Shift*)
- **Kegagalan Random CV & Censored Tail**: Validasi acak 5-Fold menghasilkan PR-AUC semu **0,6532**, sedangkan evaluasi pada ekor *right-censored* (18–31 Agu) anjlok semu ke **0,1122**. Digunakan **Mature Temporal Validation Window** (`21 Juli – 17 Agustus 2025`, $47.602$ transaksi berumur $\ge 14$ hari) yang menghasilkan evaluasi jujur **PR-AUC = 0,4420** pada fitur dasar.
- **Lonjakan Serangan QRIS 8x di Agustus 2025**: Pangsa fraud pada kanal `qris` melonjak dari **2,9% – 5,0%** (Jan–Jul) menjadi **37,6%** dari seluruh fraud di Agustus 2025 ($92 / 245$ kasus), diiringi lonjakan kategori `toko_online` (**18,4%**) dan `jasa_lainnya` (**16,3%**) serta pengurasan saldo dompet (`persen_saldo >= 0.9` naik ke **46,5%**). Model diadaptasi menggunakan fitur interaksi `qris_x_persen_saldo` dan pembobotan sampel peluruhan waktu eksponensial (*exponential time-decay sample weights*).

### 3. Komparasi Strategi Class Imbalance, Kalibrasi & Aturan Keputusan Berbasis Biaya
Evaluasi kebijakan keputusan dilakukan pada *Mature Validation Window* ($47.602$ transaksi, $293$ kasus fraud dengan total nilai fraud **Rp 314.236.700**):

| Kebijakan Keputusan / Model | Jumlah Ditandai | Fraud Tertangkap | Biaya Fraud Lolos (Rp) | Biaya Review Manual (Rp) | Total Biaya Risiko (Rp) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Flag All** (Tandai Semua Transaksi) | 47.602 | 293 (100%) | Rp 0 | Rp 1.190.050.000 | Rp 1.190.050.000 |
| **Flag None** (Tanpa Peninjauan) | 0 | 0 (0%) | Rp 314.236.700 | Rp 0 | Rp 314.236.700 |
| Ambang Konstan Standar ($p \ge 0,50$) | 86 | 64 (21,8%) | Rp 211.450.200 | Rp 2.150.000 | Rp 213.600.200 |
| Ambang Konstan Teroptimasi ($p \ge 0,045$) | 1.124 | 189 (64,5%) | Rp 98.420.500 | Rp 28.100.000 | Rp 126.520.500 |
| **Aturan Biaya Bayes Teoretis** ($p \cdot \text{nominal} \ge 25.000$) | 912 | 194 (66,2%) | Rp 86.715.000 | Rp 22.800.000 | **Rp 109.515.000** |
| **Aturan Biaya Bayes Terkalibrasi** ($\tau^* = 0,48$) | **1.584** | **228 (77,8%)** | **Rp 59.985.000** | **Rp 39.600.000** | **Rp 99.585.000** |

- **Penghematan Biaya Operasional**: Aturan keputusan berbasis biaya Bayes terkalibrasi ($p_i \times \text{nominal}_i \ge \tau^* \cdot \text{Rp } 25.000$ dengan kalibrasi *Log-Odds Platt Scaling*, Brier Score = **0,00482**) menurunkan total biaya risiko menjadi **Rp 99.585.000**, menghemat **Rp 214.651.700 (-68,3%)** dibanding tanpa peninjauan (*Flag None*) dan **Rp 1.090.465.000 (-91,6%)** dibanding menandai seluruh transaksi (*Flag All*).

---

## ⚡ Reproduksi Angka Laporan (One-Command Reproduction)

Sesuai kriteria penilaian ujian lisan (reproduksi hasil dalam 5 menit), jalankan perintah berikut dari direktori utama:

```bash
# Reproduksi Task 1 (Regresi Estimasi Waktu Pengiriman)
python kode/task1_regresi.py

# Reproduksi Task 2 (Klasifikasi Deteksi Fraud & Optimasi Biaya)
python kode/task2_klasifikasi.py
```

Skrip ini akan mengeksekusi pipeline lengkap secara otomatis, menghasilkan grafik evaluasi di `Task 1/figures/` dan `Task 2/figures/`, serta mencetak seluruh metrik kinerja dan memvalidasi kepatuhan file `prediksi_regresi.csv` maupun `prediksi_klasifikasi.csv`.

---

## 👥 Tim Penyusun (Kelompok 8)
Lihat [kontribusi.md](kontribusi.md) untuk rincian pembagian kerja, kontribusi pull request, dan tanggung jawab masing-masing anggota.
