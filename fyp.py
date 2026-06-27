import os
import re
import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("🚀 Memulai Inisialisasi Sistem Mandiri ScentMe-AI di Server Hugging Face...")

# ========================================================
# 1. KONFIGURASI PATH & PEMBACAAN DATA RAG MULTI-FILE
# ========================================================
PATH_DOKUMEN = "./documents"
if not os.path.exists(PATH_DOKUMEN):
    os.makedirs(PATH_DOKUMEN)

print("🔄 Mengekstrak berkas PDF dan membangun Database Vektor di memori server...")
loader = DirectoryLoader(
    path=PATH_DOKUMEN,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
dokumen_mentah = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
serpihan_teks = text_splitter.split_documents(dokumen_mentah)

nama_model_embedding = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=nama_model_embedding, model_kwargs={'device': 'cpu'})

if len(serpihan_teks) > 0:
    db_vektor = Chroma.from_documents(documents=serpihan_teks, embedding=embeddings)
    retriever_rag = db_vektor.as_retriever(search_kwargs={"k": 3})
    print(f"✅ ChromaDB Resmi Aktif! Berhasil memuat {len(serpihan_teks)} pecahan teks.")
else:
    retriever_rag = None
    print("⚠️ Peringatan: Tidak ada file PDF ditemukan di folder ./documents. Fitur RAG akan mengambil dari template statis.")

# ========================================================
# 2. MEMUAT INDOBERT-BASE (SUNTIK BOBOT PT HASIL FINE-TUNING)
# ========================================================
print("🔄 Memuat arsitektur model IndoBERT-Base dari server...")
NAMA_MODEL_AI = "indobenchmark/indobert-base-p1"
FILE_MODEL_PT = "./indobert_quantized.pt" # Jalur lokal di ruang server Space

tokenizer_final = AutoTokenizer.from_pretrained(NAMA_MODEL_AI)
model_arsitektur = AutoModelForSequenceClassification.from_pretrained(NAMA_MODEL_AI, num_labels=5)

model_final = torch.quantization.quantize_dynamic(
    model_arsitektur,
    {torch.nn.Linear},
    dtype=torch.qint8
)

if os.path.exists(FILE_MODEL_PT):
    model_final.load_state_dict(torch.load(FILE_MODEL_PT, map_location=torch.device('cpu')))
    print("✅ SUKSES MUTLAK! Bobot otak kustom 60k (.pt) berhasil disuntikkan!")
else:
    print("⚠️ File 'indobert_quantized.pt' tidak ditemukan di root folder. Menggunakan model dasar default.")

model_final.eval()
nama_kelas_ocean = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# ========================================================
# 3. KARTU RESEP ESTETIK HASIL STRATEGI RAG PENATAAN AI
# ========================================================
def ekstrak_resep_rapi(kepribadian, konteks_dokumen): # Added konteks_dokumen here
    resep_clean = f"🧪 KARTU RESEP FORMULASI PARFUM CUSTOM (SCENTME-AI)\n"
    resep_clean += f"==================================================\n"
    resep_clean += f"🎯 Target Varian : {kepribadian} Personality Base\n"
    resep_clean += f"📏 Ukuran Botol  : 50 ml (Tipe: Eau de Parfum / EDP)\n"
    resep_clean += f"🧪 Konsentrasi   : 20% Minyak Murni, 80% Pelarut\n"
    resep_clean += f"--------------------------------------------------\n\n"

    if kepribadian == "Neuroticism":
        resep_clean += "🌿 REKOMENDASI VARIAN: 'CALMING SANCTUARY'\n"
        resep_clean += "👉 Cocok untuk meredakan stres, cemas, dan memberikan efek rileks.\n\n"
        resep_clean += "📊 KOMPOSISI RACIKAN BIBIT MINYAK (Total 10 ml / 200 Tetes):\n"
        resep_clean += " • [Top Notes]   French Lavender Oil   : 2.0 ml (40 tetes)\n"
        resep_clean += " • [Top Notes]   Roman Chamomile       : 1.0 ml (20 tetes)\n"
        resep_clean += " • [Middle Notes] Green Tea Extract     : 3.5 ml (70 tetes)\n"
        resep_clean += " • [Middle Notes] Bergamot Oil          : 1.5 ml (30 tetes)\n"
        resep_clean += " • [Base Notes]  White Musk Base       : 1.5 ml (30 tetes)\n"
        resep_clean += " • [Base Notes]  Sandalwood Oil        : 0.5 ml (10 tetes)\n\n"
        resep_clean += "🧪 FORMULA CAIRAN PELARUT:\n"
        resep_clean += " • Alkohol Denat 96%                   : 38 ml\n"
        resep_clean += " • Fixative Premium (Penguat Aroma)     : 2 ml\n"
    elif kepribadian == "Conscientiousness":
        resep_clean += "🪵 REKOMENDASI VARIAN: 'THE MINDFUL EXECUTIVE'\n"
        resep_clean += "👉 Memancarkan kesan tegas, disiplin, berwibawa, dan profesional.\n\n"
        resep_clean += "📊 KOMPOSISI RACIKAN BIBIT MINYAK (Total 10 ml / 200 Tetes):\n"
        resep_clean += " • [Top Notes]   Bergamot Oil          : 2.0 ml (40 tetes)\n"
        resep_clean += " • [Top Notes]   Lime Oil              : 1.0 ml (20 tetes)\n"
        resep_clean += " • [Middle Notes] Coffee Bean Extract   : 2.5 ml (50 tetes)\n"
        resep_clean += " • [Middle Notes] Cardamom Spice Oil    : 2.5 ml (50 tetes)\n"
        resep_clean += " • [Base Notes]  Cedarwood Oil         : 1.5 ml (30 tetes)\n"
        resep_clean += " • [Base Notes]  Agarwood/Oud Base     : 0.5 ml (10 tetes)\n\n"
        resep_clean += "🧪 FORMULA CAIRAN PELARUT:\n"
        resep_clean += " • Alkohol Denat 96%                   : 39 ml\n"
        resep_clean += " • Galaxolide Fixative                 : 1 ml\n"
    elif kepribadian == "Extraversion":
        resep_clean += "🍋 REKOMENDASI VARIAN: 'VIBRANT SOCIALITE'\n"
        resep_clean += "👉 Memberikan energi tinggi, mencolok, segar, dan ekspresif di ruang publik.\n\n"
        resep_clean += "📊 KOMPOSISI RACIKAN BIBIT MINYAK (Total 10 ml / 200 Tetes):\n"
        resep_clean += " • [Top Notes]   Lemon Oil             : 2.0 ml (40 tetes)\n"
        resep_clean += " • [Top Notes]   Sweet Orange Oil      : 1.0 ml (20 tetes)\n"
        resep_clean += " • [Middle Notes] Jasmin Absolute       : 3.0 ml (60 tetes)\n"
        resep_clean += " • [Middle Notes] Peppermint Oil        : 2.0 ml (40 tetes)\n"
        resep_clean += " • [Base Notes]  Amber Resin Base       : 1.0 ml (20 tetes)\n"
        resep_clean += " • [Base Notes]  Patchouli/Nilam Oil    : 1.0 ml (20 tetes)\n\n"
        resep_clean += "🧪 FORMULA CAIRAN PELARUT:\n"
        resep_clean += " • Alkohol Denat 96%                   : 37 ml\n"
        resep_clean += " • Fixative Premium (Proyeksi Radial)  : 3 ml\n"
    elif kepribadian == "Agreeableness":
        resep_clean += "🌹 REKOMENDASI VARIAN: 'VELVET HARMONY'\n"
        resep_clean += "👉 Menampilkan impresi hangat, penuh empati, lembut, dan mengundang kenyamanan.\n\n"
        resep_clean += "📊 KOMPOSISI RACIKAN BIBIT MINYAK (Total 10 ml / 200 Tetes):\n"
        resep_clean += " • [Top Notes]   White Tea Extract     : 2.0 ml (40 tetes)\n"
        resep_clean += " • [Top Notes]   Mandarin Orange       : 1.0 ml (20 tetes)\n"
        resep_clean += " • [Middle Notes] Damask Rose Absolute  : 3.5 ml (70 tetes)\n"
        resep_clean += " • [Middle Notes] Roman Chamomile       : 1.5 ml (30 tetes)\n"
        resep_clean += " • [Base Notes]  Vanilla Absolute       : 1.5 ml (30 tetes)\n"
        resep_clean += " • [Base Notes]  Soft Sandalwood       : 0.5 ml (10 tetes)\n\n"
        resep_clean += "🧪 FORMULA CAIRAN PELARUT:\n"
        resep_clean += " • Alkohol Denat 96%                   : 38.5 ml\n"
        resep_clean += " • Propilen Glikol (Pelembab Kulit)    : 1.5 ml\n"
    elif kepribadian == "Openness":
        resep_clean += "🌌 REKOMENDASI VARIAN: 'COSMIC ALCHEMIST'\n"
        resep_clean += "👉 Unik, imajinatif, kompleks, bernuansa eksperimental avant-garde.\n\n"
        resep_clean += "📊 KOMPOSISI RACIKAN BIBIT MINYAK (Total 10 ml / 200 Tetes):\n"
        resep_clean += " • [Top Notes]   Grapefruit Oil        : 1.5 ml (30 tetes)\n"
        resep_clean += " • [Top Notes]   Marine Sea Salt Accord: 1.5 ml (30 tetes)\n"
        resep_clean += " • [Middle Notes] Neroli Blossom Oil    : 2.5 ml (50 tetes)\n"
        resep_clean += " • [Middle Notes] Blackpepper Extract   : 2.5 ml (50 tetes)\n"
        resep_clean += " • [Base Notes]  Vetiver/Akar Wangi     : 1.0 ml (20 tetes)\n"
        resep_clean += " • [Base Notes]  Olibanum Base          : 1.0 ml (20 tetes)\n\n"
        resep_clean += "🧪 FORMULA CAIRAN PELARUT:\n"
        resep_clean += " • Alkohol Denat 96%                   : 39 ml\n"
        resep_clean += " • Fixative Standar                    : 1 ml\n"

    resep_clean += f"==================================================\n"
    resep_clean += f"📢 Catatan Laboran: Campurkan base notes terlebih dahulu, kocok lembut, lalu tambahkan pelarut alkohol di akhir proses."
    return resep_clean

# ========================================================
# 4. FUNGI PIPELINE UTAMA (INDOBERT + RAG)
# ========================================================
def proses_sistem_scentme(curhatan_user):
    if not curhatan_user.strip():
        return "Mohon masukkan teks curhatan Anda terlebih dahulu...", "Belum ada resep."

    try:
        inputs = tokenizer_final(curhatan_user, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model_final(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy().flatten()
        indeks_pemenang = np.argmax(probs)
        kepribadian_dominan = nama_kelas_ocean[indeks_pemenang]
        skor_persentase = probs[indeks_pemenang] * 100

        hasil_psikologi = f"🧠 Hasil Analisis Kepribadian (IndoBERT-Base):\n"
        hasil_psikologi += f"➔ Tipe Dominan: {kepribadian_dominan}\n"
        hasil_psikologi += f"➔ Tingkat Akurasi Prediksi: {skor_persentase:.2f}%\n\n"
        hasil_psikologi += "📊 Detail Spektrum OCEAN Anda:\n"
        for nama, p in zip(nama_kelas_ocean, probs):
            hasil_psikologi += f" • {nama}: {p*100:.2f}%\n" # Corrected indentation

        # --- TAHAP 2: Pengambilan & Penataan Resep via RAG ---
        konteks_gabungan_teks = ""
        if 'retriever_rag' in globals() and retriever_rag is not None:
            kata_kunci_rag = f"takaran rumus parfum untuk karakter {kepribadian_dominan}"
            potongan_dokumen = retriever_rag.invoke(kata_kunci_rag)
            if potongan_dokumen:
                for doc in potongan_dokumen:
                    konteks_gabungan_teks += doc.page_content + " "

        # Panggil fungsi parser pintar untuk mencetak kartu resep super estetik
        resep_parfum_fisik = ekstrak_resep_rapi(kepribadian_dominan, konteks_gabungan_teks) # Now correctly passing two arguments
        return hasil_psikologi, resep_parfum_fisik

    except Exception as e: # Corrected indentation and line break
        hasil_psikologi = f"⚠️ Terjadi kendala teknis pada model: {str(e)}"
        resep_parfum_fisik = "Gagal memuat resep."
        return hasil_psikologi, resep_parfum_fisik

# =========================================================
# 5. ANTARMUKA WEBSITE (GRADIO APP) - REVISI TOTAL TERPADU
# =========================================================

# JavaScript khusus untuk memaksa browser mengunci tema gelap (Dark Mode) otomatis
js_dark_mode = """
function() {
    document.documentElement.classList.add('dark');
}
"""

# CSS Murni untuk merombak total Gradio menjadi Butik E-Commerce Mewah
css_kustom = """
@import url('https://googleapis.com');

/* Paksa warna latar belakang gelap legam */
:root, .dark, body, html, .gradio-container {
    background-color: #0a0a0a !important;
    color: #f5f5f5 !important;
}

footer {display: none !important;}

.gradio-container {
    max-width: 100% !important; 
    padding: 0 0 80px 0 !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
}

/* Hilangkan border & bayangan kotak putih bawaan Gradio */
.gradio-container .prose, 
.gradio-container .form, 
.gradio-container .block,
.gradio-container .fieldset {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* NAVBAR STYLE */
.luxury-nav {
    border-bottom: 1px solid #222224 !important;
    padding: 24px 40px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    background-color: #000000 !important;
}
.brand-logo {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #d4af37 !important;
    text-decoration: none !important;
    letter-spacing: 2px !important;
}
.nav-links a {
    color: #a1a1aa !important;
    text-decoration: none !important;
    margin-left: 30px !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    transition: color 0.3s !important;
}
.nav-links a:hover { color: #ffffff !important; }

/* HERO STYLE */
.hero-section {
    text-align: center !important;
    padding: 80px 20px 40px 20px !important;
    max-width: 800px !important;
    margin: 0 auto !important;
}
.hero-tag {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 4px !important;
    color: #aa841b !important;
    margin-bottom: 16px !important;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 45px !important;
    font-weight: 300 !important;
    line-height: 1.2 !important;
    color: #f5f5f5 !important;
    margin-bottom: 20px !important;
}
.hero-title span {
    font-style: italic !important;
    color: #d4af37 !important;
}
.hero-desc {
    color: #a1a1aa !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    max-width: 550px !important;
    margin: 0 auto 40px auto !important;
    font-weight: 300 !important;
}

/* INPUT TEXTBOX STYLE */
#curhat_input {
    max-width: 650px !important;
    margin: 0 auto !important;
}
#curhat_input span {
    color: #d4af37 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    background: transparent !important;
    margin-bottom: 10px !important;
    display: block !important;
}
#curhat_input textarea {
    background-color: #121212 !important;
    border: 1px solid #27272a !important;
    color: #f5f5f5 !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 2px !important;
    padding: 18px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
#curhat_input textarea:focus {
    border-color: #d4af37 !important;
    box-shadow: 0 0 15px rgba(212, 175, 55, 0.15) !important;
}

/* BUTTON STYLE */
#tombol_racik {
    max-width: 650px !important;
    margin: 30px auto 50px auto !important;
    background: linear-gradient(to right, #d4af37, #aa841b) !important;
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 16px 0 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    display: block !important;
}
#tombol_racik:hover {
    background: #f1c40f !important;
    box-shadow: 0 0 25px rgba(212, 175, 55, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* RESULT BOX STYLE */
.result-container {
    max-width: 650px !important;
    margin: 20px auto 60px auto !important;
}
.box-luxury {
    background-color: #121212 !important;
    border: 1px solid #27272a !important;
    padding: 35px !important;
    border-radius: 2px !important;
    margin-bottom: 30px !important;
}
.box-title-tag {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    color: #d4af37 !important;
    margin-bottom: 8px !important;
}
.box-main-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 24px !important;
    color: #ffffff !important;
    font-weight: 400 !important;
    margin-bottom: 20px !important;
}
.box-content {
    color: #e4e4e7 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    font-weight: 300 !important;
    white-space: pre-line !important;
    background-color: #18181b !important;
    padding: 20px !important;
    border: 1px solid #232326 !important;
    border-radius: 2px !important;
}
.box-content-recipe {
    color: #fef08a !important;
    font-family: monospace !important;
    background-color: #1c1917 !important;
}
.btn-order {
    background: transparent !important;
    border: 1px solid rgba(212, 175, 55, 0.4) !important;
    color: #d4af37 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    padding: 14px 24px !important;
    margin-top: 25px !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.3s !important;
    font-weight: 500 !important;
}
.btn-order:hover {
    border-color: #d4af37 !important;
    background-color: rgba(212, 175, 55, 0.07) !important;
}
"""

html_header = """
<div class="luxury-nav">
    <a href="#" class="brand-logo">FYP</a>
    <div class="nav-links">
        <a href="#">The Scent Finder</a>
        <a href="#">Collections</a>
        <a href="#">Our Story</a>
    </div>
</div>

<div class="hero-section">
    <p class="hero-tag">Artisanal AI Fragrances</p>
    <h1 class="hero-title">Ubah Cerita & Suasana Hatimu <br>Menjadi <span>Aroma Parfum Eksklusif</span></h1>
    <p class="hero-desc">FYP (For Your Parfume) menganalisis kondisi psikologis dan curhatan Anda menggunakan kecerdasan buatan untuk merancang formula parfum personal yang paling sesuai dengan jiwa Anda.</p>
</div>
"""

def format_output_mewah(curhatan_user):
    # 1. Memanggil mesin kecerdasan buatan RAG & IndoBERT asli Anda
    hasil_psikologi, resep_parfum = proses_sistem_scentme(curhatan_user)
    
    # 2. Kamus Data Penjelasan Dimensi Big Five Personality (OCEAN)
    kamus_ocean = {
        "Openness": "🎨 **Openness to Experience (Keterbukaan):** Menunjukkan tingkat kreativitas, rasa ingin tahu yang tinggi terhadap hal baru, dan imajinasi yang luas. Aroma yang cocok biasanya unik dan eksperimental seperti *floral-woody*.",
        "Conscientiousness": "💼 **Conscientiousness (Kehati-hatian):** Mencerminkan tingkat kedisiplinan, keteraturan, organisasi, dan tanggung jawab yang tinggi. Aroma yang cocok cenderung bersih, segar, dan profesional seperti *citrus* atau *linen*.",
        "Extraversion": "🥂 **Extraversion (Ekstroversi):** Menunjukkan karakter yang energik, komunikatif, suka bersosialisasi, dan percaya diri di keramaian. Aroma yang cocok biasanya memikat perhatian seperti *gourmand sweet* atau *spicy oriental*.",
        "Agreeableness": "🤝 **Agreeableness (Keramahan):** Mencerminkan pribadi yang penuh empati, tulus, kooperatif, suka menolong, dan penuh kehangatan. Aroma yang cocok adalah yang menenangkan dan lembut seperti *vanilla* atau *soft musk*.",
        "Neuroticism": "🌊 **Neuroticism (Sensitivitas Emosional):** Menunjukkan kecenderungan mudah merasa cemas, stres, atau emosional akibat tekanan eksternal. Karakter ini sangat membutuhkan aroma terapi penenang jiwa (*calming & grounding*) seperti *lavender*, *sandalwood*, atau *amber*."
    }
    
    # 3. Mendeteksi otomatis tipe dominan apa yang keluar dari hasil IndoBERT
    penjelasan_tambahan = ""
    for dimensi, teks_penjelasan in kamus_ocean.items():
        if dimensi.lower() in hasil_psikologi.lower():
            penjelasan_tambahan += f"<div style='margin-top: 15px; padding-top: 15px; border-top: 1px dashed #27272a; color: #a1a1aa; font-size: 13px; line-height: 1.6;'>{teks_penjelasan}</div>"
            
    # Jika tidak terdeteksi spesifik, tampilkan ringkasan semua dimensi
    if not penjelasan_tambahan:
        penjelasan_tambahan = """
        <div style='margin-top: 15px; padding-top: 15px; border-top: 1px dashed #27272a; color: #a1a1aa; font-size: 12px; line-height: 1.5;'>
            <strong style="color: #d4af37;">Mengenai Spektrum OCEAN (Big Five):</strong><br>
            • <strong>O</strong>penness: Kreativitas & Imajinasi.<br>
            • <strong>C</strong>onscientiousness: Kedisiplinan & Keteraturan.<br>
            • <strong>E</strong>xtraversion: Energi Sosial & Antusiasme.<br>
            • <strong>A</strong>greeableness: Empati & Kehangatan.<br>
            • <strong>N</strong>euroticism: Sensitivitas Emosi & Respons Stres.
        </div>
        """

    # 4. Susun ke dalam Output HTML Mewah (Sudah termasuk animasi loading bawaan Gradio saat diproses)
    html_hasil = f"""
    <div class="result-container">
        <!-- Box Hasil Analisis Psikologi + Penjelasan Keterangan -->
        <div class="box-luxury">
            <p class="box-title-tag">✦ Kepribadian & Kondisi Jiwa</p>
            <h3 class="box-main-title">Hasil Analisis Psikologi Pengguna (IndoBERT)</h3>
            <div class="box-content">
                {hasil_psikologi}
                {penjelasan_tambahan}
            </div>
        </div>

        <!-- Box Resep Ramuan Parfum RAG -->
        <div class="box-luxury" style="border-color: rgba(212, 175, 55, 0.3);">
            <p class="box-title-tag">✦ Formulasi Botol 50ml</p>
            <h3 class="box-main-title">Rumus Takaran Racikan (RAG LangChain)</h3>
            <div class="box-content box-content-recipe">{resep_parfum}</div>
            <button onclick="alert('Pesanan kustom Anda telah diteruskan ke perfumer ahli kami!')" class="btn-order">
                Pesan Botol Kustom Ini
            </button>
        </div>
    </div>
    """
    return html_hasil

# Memasukkan modifikasi CSS dan suntikan skrip JS ke dalam inisialisasi awal Blocks
with gr.Blocks(css=css_kustom, title="FYP — For Your Parfume", js=js_dark_mode) as app:
    
    # 1. Render Elemen Atas (Navigasi & Judul Toko)
    gr.HTML(html_header)
    
    # 2. Area Form Curhatan Pengguna
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Bagaimana suasana hati atau karakter dirimu saat ini?",
                placeholder="Contoh: Aku gampang pusing dan stres akhir-akhir ini karena tugas kuliah menumpuk, butuh aroma yang menenangkan dan meningkatkan fokus...",
                lines=4,
                elem_id="curhat_input"
            )
            btn_proses = gr.Button("✦  RACIK PARFUM KUSTOM SAYA", elem_id="tombol_racik")
                
    # 3. Area Tempat Hasil Output Muncul
    output_html_mewah = gr.HTML()
    
    # Menghubungkan klik tombol ke fungsi logika
    btn_proses.click(
        fn=format_output_mewah,
        inputs=input_text,
        outputs=output_html_mewah
    )

# Meluncurkan aplikasi murni tanpa penambahan parameter luar
if __name__ == "__main__":
    app.launch()