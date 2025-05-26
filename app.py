import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import librosa

# --------------------------------------------------------------------------
# 1. FUNGSI-FUNGSI DASAR FUZZY (Sama seperti sebelumnya)
# --------------------------------------------------------------------------

def fungsi_keanggotaan_segitiga(x, parameter_fungsi):
    a, b, c = parameter_fungsi
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0.0

# --------------------------------------------------------------------------
# 2. PARAMETER SISTEM
# --------------------------------------------------------------------------

def dapatkan_parameter_sistem():
    parameter = {}
    parameter['fk_input'] = {
        'tinggi_nada': {
            'Rendah': [50, 80, 150],
            'Normal': [120, 200, 280],
            'Tinggi': [250, 320, 400]
        },
        'energi': {
            'Rendah': [0, 0.15, 0.4],
            'Normal': [0.3, 0.5, 0.7],
            'Tinggi': [0.6, 0.8, 1.0]
        },
        'kecepatan_bicara': { 
            'Lambat': [0.5, 1.5, 2.5],
            'Normal': [2.0, 3.0, 4.0],
            'Cepat': [3.5, 4.5, 6.0]
        }
    }
    parameter['fk_output'] = {
        'emosi': {
            'Sedih': [0, 1.5, 3],
            'Tenang': [2, 4, 6],
            'Marah': [5, 7, 9],
            'Senang': [7.5, 9, 10]
        }
    }
    parameter['semesta_output'] = np.arange(0, 10.1, 0.1)
    parameter['daftar_aturan'] = [
        (('tinggi_nada:Rendah', 'energi:Rendah', 'kecepatan_bicara:Lambat'), 'emosi:Sedih'),
        (('tinggi_nada:Normal', 'energi:Rendah', 'kecepatan_bicara:Normal'), 'emosi:Tenang'),
        (('tinggi_nada:Rendah', 'energi:Rendah', 'kecepatan_bicara:Normal'), 'emosi:Tenang'),
        (('tinggi_nada:Tinggi', 'energi:Tinggi', 'kecepatan_bicara:Normal'), 'emosi:Marah'),
        (('tinggi_nada:Normal', 'energi:Tinggi', 'kecepatan_bicara:Cepat'), 'emosi:Marah'),
        (('tinggi_nada:Tinggi', 'energi:Tinggi', 'kecepatan_bicara:Cepat'), 'emosi:Marah'),
        (('tinggi_nada:Tinggi', 'energi:Normal', 'kecepatan_bicara:Cepat'), 'emosi:Senang'),
        (('tinggi_nada:Normal', 'energi:Normal', 'kecepatan_bicara:Cepat'), 'emosi:Senang'),
        (('tinggi_nada:Tinggi', 'energi:Tinggi', 'kecepatan_bicara:Cepat'), 'emosi:Senang'),
        (('energi:Rendah',), 'emosi:Tenang'), 
        (('kecepatan_bicara:Lambat',), 'emosi:Sedih')
    ]
    return parameter

# --------------------------------------------------------------------------
# 3. LOGIKA FUZZY
# --------------------------------------------------------------------------

def fuzzifikasi(input_numerik, fk_input):
    hasil_fuzzifikasi = {}
    for nama_variabel, nilai_numerik in input_numerik.items():
        hasil_fuzzifikasi[nama_variabel] = {}
        if nama_variabel not in fk_input: # Penanganan jika variabel tidak ada di fk_input
            print(f"Peringatan: Variabel '{nama_variabel}' tidak ditemukan dalam definisi fk_input.")
            continue
        for nama_himpunan, parameter_fungsi in fk_input[nama_variabel].items():
            hasil_fuzzifikasi[nama_variabel][nama_himpunan] = fungsi_keanggotaan_segitiga(nilai_numerik, parameter_fungsi)
    return hasil_fuzzifikasi

def terapkan_aturan(input_hasil_fuzzifikasi, daftar_aturan):
    aktivasi_aturan = []
    for daftar_kondisi, konsekuensi in daftar_aturan:
        kekuatan_kondisi = []
        valid_rule = True
        for kondisi in daftar_kondisi:
            nama_variabel, nama_himpunan = kondisi.split(':')
            if nama_variabel not in input_hasil_fuzzifikasi or nama_himpunan not in input_hasil_fuzzifikasi[nama_variabel]:
                # print(f"Peringatan: Kondisi '{kondisi}' tidak dapat dievaluasi, variabel atau himpunan tidak ada dalam hasil fuzzifikasi.")
                valid_rule = False
                break
            kekuatan = input_hasil_fuzzifikasi[nama_variabel][nama_himpunan]
            kekuatan_kondisi.append(kekuatan)
        
        if not valid_rule or not kekuatan_kondisi:
            continue

        kekuatan_aktivasi = min(kekuatan_kondisi)
        if kekuatan_aktivasi > 0:
            aktivasi_aturan.append((konsekuensi, kekuatan_aktivasi))
    return aktivasi_aturan

def agregasi_output(aktivasi_aturan):
    kekuatan_teragregasi = {}
    for (konsekuensi, kekuatan) in aktivasi_aturan:
        nama_variabel, nama_himpunan = konsekuensi.split(':')
        if nama_himpunan not in kekuatan_teragregasi:
            kekuatan_teragregasi[nama_himpunan] = 0.0
        kekuatan_teragregasi[nama_himpunan] = max(kekuatan_teragregasi[nama_himpunan], kekuatan)
    return kekuatan_teragregasi

def defuzzifikasi_centroid(kekuatan_teragregasi, fk_output, semesta):
    pembilang = 0.0
    penyebut = 0.0
    kurva_teragregasi = np.zeros_like(semesta)
    
    if 'emosi' not in fk_output or not isinstance(fk_output['emosi'], dict):
        print("Error: fk_output['emosi'] tidak terdefinisi dengan benar.")
        return 0.0, kurva_teragregasi

    for i, x in enumerate(semesta):
        keanggotaan_maksimum = 0.0
        for nama_himpunan, kekuatan_aktivasi in kekuatan_teragregasi.items():
            if nama_himpunan not in fk_output['emosi']:
                # print(f"Peringatan: Himpunan output '{nama_himpunan}' tidak ditemukan di fk_output['emosi'].")
                continue
            parameter_fungsi = fk_output['emosi'][nama_himpunan]
            keanggotaan_terpotong = min(fungsi_keanggotaan_segitiga(x, parameter_fungsi), kekuatan_aktivasi)
            if keanggotaan_terpotong > keanggotaan_maksimum:
                keanggotaan_maksimum = keanggotaan_terpotong
        kurva_teragregasi[i] = keanggotaan_maksimum

    if np.sum(kurva_teragregasi) == 0:
        return 0.0, kurva_teragregasi
        
    pembilang = np.sum(semesta * kurva_teragregasi)
    penyebut = np.sum(kurva_teragregasi)
    
    if penyebut == 0:
        return 0.0, kurva_teragregasi

    output_numerik = pembilang / penyebut
    return output_numerik, kurva_teragregasi

# --------------------------------------------------------------------------
# 4. FUNGSI EKSTRAKSI FITUR AUDIO
# --------------------------------------------------------------------------
def ekstrak_fitur_audio(path_audio):
    """Mengekstrak fitur tinggi nada, energi, dan kecepatan bicara dari file audio."""
    if path_audio is None:
        return 0, 0, 0

    try:
        y, sr = librosa.load(path_audio, sr=None)
        
        # 1. Tinggi Nada (Pitch) - menggunakan PYIN
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_valid = f0[~np.isnan(f0)]
        pitch_rata_rata = np.mean(f0_valid) if len(f0_valid) > 0 else 150 # Default jika tidak ada pitch terdeteksi

        # 2. Energi (RMS)
        rms = librosa.feature.rms(y=y)[0]
        energi_rata_rata = np.mean(rms)
        # Normalisasi energi ke rentang 0-1 (asumsi nilai RMS tipikal)
        energi_ternormalisasi = np.clip(energi_rata_rata * 5, 0, 1) # Faktor 5 adalah contoh, bisa di-tuning

        # 3. Kecepatan Bicara
        durasi_audio_detik = librosa.get_duration(y=y, sr=sr)
        if durasi_audio_detik == 0:
            kecepatan_bicara_segmen_per_detik = 1.0
        else:
            segmen_non_sunyi = librosa.effects.split(y, top_db=25) # top_db mungkin perlu di-tuning
            jumlah_segmen = len(segmen_non_sunyi)
            kecepatan_bicara_segmen_per_detik = jumlah_segmen / durasi_audio_detik if durasi_audio_detik > 0 else 1.0
            # Batasi nilai agar sesuai dengan rentang FK
            kecepatan_bicara_segmen_per_detik = np.clip(kecepatan_bicara_segmen_per_detik, 0.5, 6.0)


        return round(pitch_rata_rata,2), round(energi_ternormalisasi,2), round(kecepatan_bicara_segmen_per_detik,2)

    except Exception as e:
        print(f"Error saat memproses audio: {e}")
        return 150, 0.5, 3.0

# --------------------------------------------------------------------------
# 5. FUNGSI UNTUK GRADIO 
# --------------------------------------------------------------------------

def terjemahkan_skor_ke_emosi(skor):
    if skor <= 3.5: return "Sedih"
    elif skor <= 6.5: return "Tenang"
    elif skor < 8.0: return "Marah"
    else: return "Senang"

def jalankan_simulasi_fuzzy_untuk_gradio(input_numerik):
    parameter = dapatkan_parameter_sistem()
    fk_input = parameter['fk_input']
    fk_output = parameter['fk_output']
    daftar_aturan = parameter['daftar_aturan']
    semesta_output = parameter['semesta_output']

    hasil_fuzzifikasi = fuzzifikasi(input_numerik, fk_input)
    aktivasi_aturan = terapkan_aturan(hasil_fuzzifikasi, daftar_aturan)
    kekuatan_teragregasi = agregasi_output(aktivasi_aturan)
    output_numerik, kurva_teragregasi = defuzzifikasi_centroid(
        kekuatan_teragregasi, fk_output, semesta_output
    )
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for nama_himpunan, parameter_fk in fk_output['emosi'].items():
        ax.plot(semesta_output, [fungsi_keanggotaan_segitiga(x, parameter_fk) for x in semesta_output], label=nama_himpunan)
    ax.fill_between(semesta_output, kurva_teragregasi, color='gray', alpha=0.5, label='Area Agregasi')
    ax.vlines(output_numerik, 0, 1, color='k', linestyle='--', label=f'Centroid ({output_numerik:.2f})')
    
    ax.set_title(f"Visualisasi Inferensi Fuzzy")
    ax.set_xlabel("Skor Emosi")
    ax.set_ylabel("Derajat Keanggotaan")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    return output_numerik, fig

# Fungsi untuk Mode Manual
def prediksi_emosi_manual_gradio(tinggi_nada, energi, kecepatan_bicara):
    input_pengguna = {
        'tinggi_nada': tinggi_nada,
        'energi': energi,
        'kecepatan_bicara': kecepatan_bicara
    }
    skor, gambar_plot = jalankan_simulasi_fuzzy_untuk_gradio(input_pengguna)
    label_emosi = terjemahkan_skor_ke_emosi(skor)
    return label_emosi, round(skor, 2), gambar_plot

# Fungsi untuk Mode Coba Suara
def prediksi_emosi_audio_gradio(path_audio_input):
    if path_audio_input is None:
        default_fig, _ = plt.subplots(figsize=(8,5)) # Plot kosong
        return "Tidak ada audio", 0.0, default_fig, 0, 0, 0

    pitch_val, energi_val, kecepatan_val = ekstrak_fitur_audio(path_audio_input)
    
    input_fitur_audio = {
        'tinggi_nada': pitch_val,
        'energi': energi_val,
        'kecepatan_bicara': kecepatan_val
    }
    
    skor, gambar_plot = jalankan_simulasi_fuzzy_untuk_gradio(input_fitur_audio)
    label_emosi = terjemahkan_skor_ke_emosi(skor)
    
    return label_emosi, round(skor, 2), gambar_plot, pitch_val, energi_val, kecepatan_val

# --------------------------------------------------------------------------
# 6. ANTARMUKA GRADIO DENGAN TABS
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg') # Penting untuk Gradio agar tidak error di server

with gr.Blocks(theme=gr.themes.Soft()) as antarmuka_fuzzy:
    gr.Markdown("# 🤖 Deteksi Emosi Suara dengan Logika Fuzzy (Dari Awal)")
    gr.Markdown("Pilih mode di bawah: 'Mode Manual' untuk input slider, atau 'Mode Coba Suara' untuk menggunakan mikrofon/unggahan audio.")

    with gr.Tabs():
        with gr.TabItem("Mode Manual"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_pitch_manual = gr.Slider(minimum=50, maximum=400, value=200, step=1, label="Tinggi Nada (Pitch) dalam Hz")
                    input_energi_manual = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Energi Suara (Ternormalisasi)")
                    input_kecepatan_manual = gr.Slider(minimum=0.5, maximum=6.0, value=3.0, step=0.1, label="Kecepatan Bicara (Segmen/detik)") # Sesuaikan rentang
                    tombol_manual = gr.Button("Prediksi Emosi (Manual)", variant="primary")
                with gr.Column(scale=2):
                    output_label_manual = gr.Label(label="Prediksi Emosi")
                    output_skor_manual = gr.Number(label="Skor Emosi Final (0-10)")
                    output_plot_manual = gr.Plot(label="Grafik Proses Inferensi Fuzzy")
            
            tombol_manual.click(
                fn=prediksi_emosi_manual_gradio,
                inputs=[input_pitch_manual, input_energi_manual, input_kecepatan_manual],
                outputs=[output_label_manual, output_skor_manual, output_plot_manual]
            )
            gr.Examples(
                examples=[
                    [300, 0.9, 3.5], # Contoh input untuk marah (kecepatan normal)
                    [100, 0.2, 1.5], # Contoh input untuk sedih
                    [200, 0.3, 2.5]  # Contoh input untuk tenang
                ],
                inputs=[input_pitch_manual, input_energi_manual, input_kecepatan_manual]
            )

        with gr.TabItem("Mode Coba Suara"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_audio = gr.Audio(sources=["microphone"], type="filepath", label="Rekam atau Unggah Suara Anda (.wav, .mp3)")
                    tombol_audio = gr.Button("Prediksi Emosi dari Suara", variant="primary")
                    gr.Markdown("Setelah merekam/mengunggah, klik tombol di atas.")
                    gr.Markdown("**Catatan:** Ekstraksi fitur mungkin memerlukan beberapa detik. Kualitas prediksi sangat bergantung pada kualitas rekaman dan akurasi ekstraksi fitur.")
                with gr.Column(scale=2):
                    output_label_audio = gr.Label(label="Prediksi Emosi (dari Suara)")
                    output_skor_audio = gr.Number(label="Skor Emosi Final (dari Suara)")
                    output_plot_audio = gr.Plot(label="Grafik Proses Inferensi (dari Suara)")
                    with gr.Accordion("Lihat Fitur Audio yang Diekstrak:", open=False):
                        output_fitur_pitch = gr.Number(label="Fitur Pitch Terdeteksi (Hz)")
                        output_fitur_energi = gr.Number(label="Fitur Energi Terdeteksi (0-1)")
                        output_fitur_kecepatan = gr.Number(label="Fitur Kecepatan Bicara Terdeteksi (segmen/detik)")
            
            tombol_audio.click(
                fn=prediksi_emosi_audio_gradio,
                inputs=[input_audio],
                outputs=[output_label_audio, output_skor_audio, output_plot_audio,
                         output_fitur_pitch, output_fitur_energi, output_fitur_kecepatan]
            )

if __name__ == "__main__":
    antarmuka_fuzzy.launch()
