# Flask OpenCV Face Recognition

Project ini adalah aplikasi berbasis Flask dan OpenCV untuk verifikasi identitas pemilih menggunakan deteksi dan pengenalan wajah. Sistem ini membantu petugas mencocokkan wajah pemilih yang datang ke TPS dengan data pemilih yang telah terdaftar, sehingga proses verifikasi dapat dilakukan lebih cepat, lebih akurat, dan lebih tertib.

## Fitur Utama

- Menambahkan data pemilih ke sistem
- Mengambil dataset wajah dari kamera
- Melatih model pengenalan wajah
- Melakukan verifikasi wajah secara real-time
- Menampilkan hasil deteksi pemilih yang terverifikasi atau tidak terdaftar
- Menampilkan kategori pemilih dan informasi surat suara

## Teknologi yang Digunakan

- Python
- Flask
- OpenCV
- MySQL
- Pillow
- NumPy

## Struktur Singkat Project

- `app.py` sebagai file utama aplikasi Flask
- `templates/` untuk halaman HTML
- `static/` untuk file CSS, JavaScript, dan gambar
- `resources/` untuk file pendukung seperti haarcascade
- `dataset/` untuk menyimpan hasil dataset wajah
- `classifier.xml` untuk model hasil training

## Persyaratan

- Python 3.x
- MySQL
- Webcam

## Instalasi

1. Clone repository ini.
2. Masuk ke folder project.
3. Buat virtual environment.
4. Install dependency dari `requirements.txt`.
5. Siapkan database MySQL dengan nama `flask_db`.
6. Sesuaikan konfigurasi koneksi database di `app.py`.
7. Jalankan aplikasi Flask.

Contoh perintah:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Untuk Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Konfigurasi Database

Secara default aplikasi menggunakan konfigurasi berikut di `app.py`:

```python
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
```

Pastikan database MySQL sudah tersedia dan tabel yang dibutuhkan seperti `prs_mstr` dan `img_dataset` sudah dibuat.

## Cara Menjalankan

Jalankan aplikasi dengan perintah:

```bash
python app.py
```

Setelah itu buka browser dan akses:

```text
http://127.0.0.1:5000
```

## Alur Penggunaan

1. Tambahkan data pemilih terlebih dahulu.
2. Ambil dataset wajah untuk pemilih tersebut.
3. Lakukan training classifier.
4. Buka halaman face recognition.
5. Sistem akan mencocokkan wajah yang terdeteksi dengan data yang sudah tersimpan.

## Catatan Penting

- Jangan lupa buat folder bernama `dataset`.
- Folder `dataset` digunakan untuk menyimpan citra wajah hasil pengambilan dataset.
- Pastikan webcam aktif dan dapat diakses oleh aplikasi.
- File `classifier.xml` akan digunakan sebagai model hasil training wajah.
- Aplikasi ini saat ini menggunakan MySQL sebagai database utama.

## Dependency

Isi `requirements.txt`:

```text
Flask
mysql-connector-python
opencv-contrib-python
Pillow
numpy
```

## Saran Penggunaan

Project ini cocok digunakan sebagai prototype, tugas akhir, penelitian, atau pengembangan sistem verifikasi pemilih berbasis face recognition. Jika ingin dipakai di server production, disarankan untuk merapikan konfigurasi path, keamanan koneksi database, dan proses deployment.
