import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
import re
import os

# Fungsi untuk memastikan file ada sebelum dibaca
def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' tidak ditemukan.")

# Fungsi untuk tokenisasi
def tokenize(text):
    if isinstance(text, str):  # Memeriksa apakah text adalah string
        return re.findall(r'\w+', text)
    else:
        return []  # Mengembalikan list kosong jika bukan string

# Fungsi utama untuk menghitung frekuensi kata
def main():
    # Cek apakah file CSV ada
    file_path = 'preprocessed_text_output.csv'
    check_file_exists(file_path)
    
    # Load data dari 'preprocessed_text_output.csv'
    df = pd.read_csv(file_path)

    # Pastikan kolom 'preprocessed_text' ada di DataFrame
    if 'preprocessed_text' not in df.columns:
        raise ValueError("Kolom 'preprocessed_text' tidak ditemukan dalam file CSV.")

    # Inisialisasi stemmer dari Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Tokenisasi teks di kolom 'preprocessed_text'
    df['tokens'] = df['preprocessed_text'].apply(tokenize)

    # Gabungkan semua token menjadi satu list besar
    all_words = [word for tokens in df['tokens'] for word in tokens]

    # Hitung frekuensi kata menggunakan Counter
    word_freq = Counter(all_words)

    # Ambil kata-kata paling umum sebagai daftar tuple (kata, frekuensi)
    most_common_words = word_freq.most_common()

    # Ubah daftar tuple menjadi DataFrame pandas
    df_word_freq = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

    # Simpan DataFrame ke file CSV
    output_file = 'word_frequencies.csv'
    df_word_freq.to_csv(output_file, index=False)

    # Cetak konfirmasi
    print("Word frequencies saved to 'word_fredsisi.csv'")

# Jalankan fungsi utama
if __name__ == "__main__":
    main()
