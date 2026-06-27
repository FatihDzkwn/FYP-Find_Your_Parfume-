# FYP (Find Your Perfume) - Custom Perfume Recommender System

> **Proyek Akhir Mata Kuliah Natural Language Processing (NLP) / Computational Linguistics**
> Mengintegrasikan Algoritma *Semi-Supervised Pseudo-Labelling (DeBERTa-v3)*, Arsitektur *Finetuned IndoBERT-Base* (Akurasi 75.28% dengan Kuantisasi INT8), & *Multi-File Retrieval-Augmented Generation (RAG)* Menggunakan LangChain & ChromaDB.

---

## 📌 1. Latar Belakang Proyek
Aplikasi **FYP (Find Your Perfume)** adalah sistem cerdas berbasis kecerdasan buatan (AI) yang mampu memberikan rekomendasi produk dan resep formulasi minyak parfum kustom secara presisi berdasarkan hasil analisis spektrum kepribadian psikologi pengguna menggunakan model tata sifat **Big Five Personality (OCEAN)**. 

Sistem ini menyelesaikan masalah subjektivitas industri wewangian konvensional dengan menggabungkan tiga pilar teknologi NLP mutakhir dalam satu pipa integrasi (*pipeline*) hulu-hilir:
1. **Semi-Supervised Pseudo-Labelling (DeBERTa-v3)**: Menggunakan model *Zero-Shot Classification* `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` untuk menyaring dan melabeli secara otomatis **60.000 baris data** curhatan mentah netizen ke dalam 5 kelas kepribadian Big Five (OCEAN) sebagai basis data latihan utama kelompok kami.
2. **Natural Language Understanding (NLU/Fine-Tuning)**: Model Transformer *IndoBERT-Base* dilatih khusus (*Supervised Fine-Tuning*) di atas 60.000 data hasil kurasi DeBERTa-v3 tersebut guna mengunci kecerdasan model lokal dalam memetakan emosi teks secara *real-time*.
3. **Retrieval-Augmented Generation (RAG)**: Framework *LangChain* dan database vektor *ChromaDB* digunakan untuk mengamankan basis data pengetahuan (*knowledge base*) dari berkas PDF utuh (Jurnal Binus 2024, Skripsi Riset 2021, Teori Aroma 2024, dan Dokumen SOP Formulasi Laboratorium) guna menyajikan kartu resep mililiter (ml) murni secara akurat tanpa gejala halusinasi.

---

## 📊 2. Hasil Eksperimen & Evaluasi Model (UAS Bobot 15%)
Seluruh rangkaian pengujian pelatihan (*fine-tuning*) dilakukan secara adil di atas infrastruktur akselerasi **NVIDIA T4 GPU** dengan pembagian proporsi data *80% Training Dataset* dan *20% Validation Dataset*:

| Peringkat | Arsitektur Model AI Transformer | Parameter Pelatihan | Akurasi Akhir | Weighted F1-Score | Status Proyek |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **🥇 1** | **IndoBERT-Base (`indobenchmark/indobert-base-p1`)** | LR: 2e-5, Epoch: 3, Batch: 32 | **75.28%** | **75.20%** | **DIPILIH (Juara Sejati)** |
| 🥈 2 | IndoRoBERTa-Base (`LazarusNLP/simcse-indoroberta-base`) | LR: 2e-5, Epoch: 3, Batch: 32 | 71.55% | 71.52% | Komparasi Akademis |
| 🥉 3 | XLM-RoBERTa-Base (`meta-ai/xlm-roberta-base`) | LR: 2e-5, Epoch: 3, Batch: 16 | 61.24% | 61.25% | Komparasi Akademis |
| 🏅 4 | IndoBERT-Lite-Base (`indobenchmark/indobert-lite-base-p1`) | LR: 2e-5, Epoch: 3, Batch: 16 | 42.71% | 35.33% | Komparasi Akademis |

*Analisis Temuan Utama:* Meskipun rumpun RoBERTa secara teoretis lebih modern, **IndoBERT-Base** memenangkan pertandingan final secara mutlak. Hal ini dikarenakan struktur kamus kata (*vocabulary embeddings*) milik IndoBenchmark jauh lebih adaptif dan selaras dalam memetakan singkatan kata, bahasa gaul, serta kebisingan data (*data noise*) ulasan non-formal masyarakat Indonesia.

---

## 📦 3. Optimasi Perangkat Lunak: Kuantisasi Dinamis (INT8)
Guna menjamin skalabilitas aplikasi pada lingkungan server *cloud* gratisan yang memiliki keterbatasan sumber daya perangkat keras, model FP32 pasca-pelatihan dipangkas menggunakan teknik **Post-Training Dynamic Quantization (INT8)** pada seluruh lapisan jaringan saraf linier (`nn.Linear`).
* **Ukuran Berkas Awal (FP32)**: ~498 MB
* **Ukuran Berkas Akhir (INT8)**: **~125 MB**
* **Dampak Teknis**: Penyusutan kapasitas fisik hingga **75% lebih ringan**, memangkas penggunaan memori RAM server, dan melipatgandakan kecepatan inferensi teks (*inference latency*) hingga 3x lipat lebih responsif di atas komputasi mesin CPU murni.

---

## 📐 4. Alur Arsitektur Sistem Integrasi (Pipeline)
```text
📊 TAHAP 1: EKSPERIMEN & PIPELINE DATA (HULU)
[ 60.000 Teks Mentah ] ──► [ DeBERTa-v3 Zero-Shot ] ──► [ Auto-Labeling OCEAN ]
                                                                 │
                                                                 ▼
[ Otak Juara INT8 ] ◄── [ Kuantisasi Dinamis ] ◄── [ Finetuning IndoBERT-Base ]

────────────────────────────────────────────────────────────────────────────────

🌐 TAHAP 2: IMPLEMENTASI APLIKASI WEB LIVE / INFERENCE (HILIR)
[ Input Curhatan Baru Pengguna ]
               │
               ▼
       [ Pembersihan Teks ]
               │
               ▼
┌──────────────┴──────────────┐
│  Mesin Pembaca Emosi (NLU)  │
│  - Finetuned IndoBERT INT8  │ ──► Output: % Spektrum OCEAN Dominan
└──────────────┬──────────────┘
               │ (Mengunci Sifat Dominan)
               ▼
┌──────────────┴──────────────┐
│ Pustakawan Digital (RAG)    │
│  - LangChain + ChromaDB     │ ──► Output: Ambil Potongan PDF SOP Takaran
└──────────────┬──────────────┘
               │
               ▼
[ LAYAR UTAMA WEBSITE FYP: Hasil Analisis Emosi + Kartu Resep Fisik Botol 50ml ]
```

---

## 📁 5. Blueprint Silsilah Direktori Repositori
```text
📁 FYP(Find_Your_Parfume)
 ├── 📄 fyp.py                     <-- Skrip utama aplikasi & antarmuka Gradio Web UI
 ├── 📄 requirements.txt           <-- Daftar dependensi pustaka Python untuk server hosting
 ├── 📄 indobert_quantized.pt      <-- File biner kompresi otak AI INT8 (Akurasi 75.28%)
 ├── 📄 README.md                  <-- Berkas dokumentasi utama proyek akhir (File ini)
 ├── 📄 .hfignore                  <-- Berkas instruksi penapis otomatis untuk server Hugging Face
 ├── 📁 Documents                  <-- Kumpulan berkas PDF referensi orisinal database RAG
 ├── 📁 Data                       <-- Berkas master dataset mentah & bersih 60.000 baris (.csv)
 ├── 📁 Notebook                   <-- Salinan lembar kerja riset Google Colab (Babak 1 & 2)
 ├── 📁 Model                      <-- Folder penyimpanan berkas backup checkpoint bobot AI
 └── 📁 Report                     <-- Penampung berkas eksternal (Grafik evaluasi & draf makalah)
```

---

## 🛠️ 6. Cara Menjalankan Secara Lokal (Offline Mode)
Guna melakukan simulasi atau pengujian sistem di komputer lokal Anda, ikuti instruksi baris perintah berikut:

1. Kloning repositori ini atau unduh folder proyek ke penyimpanan lokal Anda.
2. Pastikan file otak biner kustom `indobert_quantized.pt` sudah diletakkan di direktori utama.
3. Buka terminal/CMD di dalam direktori folder tersebut, lalu instal seluruh pustaka dependensi secara serentak:
   ```bash
   pip install -r requirements.txt
   ```
4. Nyalakan server aplikasi lokal menggunakan Python:
   ```bash
   python fyp.py
   ```
5. Buka peramban browser Anda, lalu akses antarmuka aplikasi visual pada alamat:
   * **`http://127.0.0.1:7860`** (Rute Jaringan Lokal)
   * **`https://gradio.live`** (Rute Terowongan Publik Cadangan)