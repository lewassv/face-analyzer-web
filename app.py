import os
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2

app = Flask(__name__)

# Konfigurasi batas ukuran maksimal data yang dikirim dari browser (16 Megabyte)
# Ini memastikan string gambar Base64 resolusi tinggi dari kamera tidak ditolak oleh server
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Mengonfigurasi Flask agar tidak membuang performa CPU untuk mengurutkan kunci JSON secara alfabetis.
# Langkah ini membuat pengiriman data hasil analisis dari backend ke halaman web menjadi lebih instan.
app.json.sort_keys = False

# Memastikan folder untuk menyimpan aset gambar hasil potret otomatis sudah terbuat
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Menentukan file XML lokal yang ada di folder proyek VS Code Anda
XML_FILE = "haarcascade_frontalface_default.xml"

@app.route('/')
def index():
    """ Rute utama untuk memuat halaman tampilan antarmuka kamera web futuristik. """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_face():
    """ API Endpoint untuk menerima data gambar dari kamera, melakukan pemotongan 
    wajah secara lokal, dan mengekstrak data demografis menggunakan DeepFace AI. """
    try:
        # Mengambil payload data JSON yang dikirim oleh JavaScript Fetch API
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'Data gambar tidak valid atau kosong.'}), 400
            
        image_data = data.get('image')

        # Menghapus header biner skema Base64 (data:image/jpeg;base64,)
        header, encoded = image_data.split(",", 1)
        decoded = base64.b64decode(encoded)

        # Menentukan jalur penyimpanan fisik gambar hasil potret otomatis
        image_path = os.path.join(UPLOAD_FOLDER, 'capture.jpg')
        with open(image_path, "wb") as f:
            f.write(decoded)

        # --- TAHAP PROSES DETEKSI & PEMOTONGAN WAJAH LOKAL (PINTAR) ---
        # Mengonversi data biner gambar menjadi matriks array NumPy agar bisa dibaca OpenCV
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Membaca pola wajah secara lokal menggunakan file HaarCascade XML proyek Anda
        face_cascade = cv2.CascadeClassifier(XML_FILE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Jika pola wajah berhasil ditemukan oleh XML lokal, kita potong area wajahnya saja.
        # Langkah ini adalah trik optimasi agar jaringan saraf tiruan DeepFace menerima 
        # gambar wajah yang sangat fokus dan dekat, sehingga tingkat akurasi analisis meningkat tajam.
        if len(faces) > 0:
            x, y, w, h = faces[0]
            offset = 20  # Memberikan margin piksel di sekitar wajah agar potongan tampak natural
            h_img, w_img, _ = img.shape
            y1, y2 = max(0, y - offset), min(h_img, y + h + offset)
            x1, x2 = max(0, x - offset), min(w_img, x + w + offset)
            
            face_crop = img[y1:y2, x1:x2]
            cv2.imwrite(image_path, face_crop) # Menimpa file capture.jpg dengan gambar wajah yang terfokus

        # --- TAHAP ANALISIS DEEP LEARNING (DEEPFACE) ---
        # Kita set detector_backend='skip' karena prapemrosesan lokalisasi wajah 
        # sudah kita selesaikan secara cerdas menggunakan OpenCV lokal di atas.
        # Enforce_detection diatur False agar AI tidak sensitif jika posisi wajah sedikit miring.
        analysis = DeepFace.analyze(
            img_path=image_path, 
            actions=['age', 'gender', 'race', 'emotion'],
            detector_backend='skip',
            enforce_detection=False
        )

        # Menangani respons jika DeepFace mengembalikan data dalam bentuk list array
        if isinstance(analysis, list):
            result = analysis[0]
        else:
            result = analysis

        # Menyusun paket data JSON akhir yang rapi untuk dikirim kembali ke browser
        response_data = {
            'status': 'success',
            'image_url': f'/static/uploads/capture.jpg?t={os.path.getmtime(image_path)}', # t=timestamp digunakan untuk memaksa browser membuang cache foto lama
            'age': result.get('age'),
            'gender': result.get('dominant_gender'),
            'race': result.get('dominant_race'),
            'emotion': result.get('dominant_emotion')
        }
        return jsonify(response_data)

    except Exception as e:
        # Mengembalikan log kesalahan sistem secara transparan ke sisi frontend jika terjadi kendala
        return jsonify({'status': 'error', 'message': f'Analisis Gagal: {str(e)}'}), 500

if __name__ == '__main__':
    # PERBAIKAN UTAMA: use_reloader diatur ke False.
    # Ini mencegah Flask melakukan restart otomatis secara sepihak saat mendeteksi 
    # aktivitas pemuatan berkas binarisasi model deep learning milik DeepFace di folder site-packages.
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)