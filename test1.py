import pandas as pd
import nltk
# # Mendownload daftar kata yang ada (vocabulary)
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from nltk.tokenize import word_tokenize
import re


path = 'Dataset-AlfonsusAntero-1204210085.csv'
df = pd.read_csv(path, nrows=1000)

print(df['narasi'])

df['narasi'].to_csv('narasi_output.csv', index=False)

print("Kolom 'narasi' berhasil disimpan ke 'narasi_full.csv'")