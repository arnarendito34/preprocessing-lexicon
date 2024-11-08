import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Membuat stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Tokenisasi sederhana menggunakan spasi
def simple_tokenize(text):
    return text.split()

# Fungsi untuk normalisasi slang
def normalize_slang(text, slang_dict):
    words = simple_tokenize(text)  # Tokenisasi manual
    normalized_words = [slang_dict.get(word, word) for word in words]  # Ganti slang dengan kata yang benar
    return ' '.join(normalized_words)

# Fungsi untuk preprocessing teks
def preprocess_text(text, slang_dict, stemmer):
    # Lowercasing
    text = text.lower()

    # Normalisasi slang dan singkatan
    text = normalize_slang(text, slang_dict)

    # Hilangkan tanda baca
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenisasi manual
    tokens = simple_tokenize(text)

    # Stemming menggunakan Sastrawi
    tokens = [stemmer.stem(word) for word in tokens]

    # Gabungkan token kembali menjadi kalimat
    return ' '.join(tokens)

# Tambahkan dictionary slang yang diinginkan
def load_slang_dictionary(file_path):
    slang_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Pisahkan slang dan kata normal dengan tanda ":" hanya pada pemisahan pertama
            if ':' in line:
                slang, normal = line.strip().split(':', 1)  # maxsplit=1
                slang_dict[slang] = normal
    return slang_dict

# Path ke file slang dictionary
slang_dict_file = 'slang_abbrevations_words.txt'

# Load slang dictionary dari file
slang_dict = load_slang_dictionary(slang_dict_file)


# Load data dari 'full_text_output.csv'
path = 'narasi_output.csv'  # Ini file yang sudah Anda buat sebelumnya
df = pd.read_csv(path)

# Terapkan preprocessing pada kolom 'full_text'
df['narasi'] = df['narasi'].fillna('')  # Mengganti NaN dengan string kosong
df['preprocessed_text'] = df['narasi'].apply(lambda x: preprocess_text(x, slang_dict, stemmer))

# Simpan hanya kolom 'preprocessed_text' ke file CSV baru
df[['preprocessed_text']].to_csv('preprocessed_text_output.csv', index=False)

print("Preprocessing selesai, hasil kolom 'preprocessed_text' disimpan ke 'preprocessed_text_output.csv'")
