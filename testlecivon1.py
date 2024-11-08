import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Melakukan load kamus senti versi bahasa indonesia
path = 'sentiwords_id.txt'
df_senti = pd.read_csv(path, sep=':', names=['word','value'])

senti_dict = {row['word']: float(row['value']) for _, row in df_senti.iterrows()}

# Buat instance analyzer dan update lexicon-nya
senti_indo = SentimentIntensityAnalyzer()
senti_indo.lexicon.update(senti_dict)

print("Setup berhasil! Analyzer siap digunakan.")


words_to_check = ["positif", "negatif", "baik", "buruk"]

for word in words_to_check:
    if word in senti_indo.lexicon:
        print(f"Kata '{word}' ditemukan dalam lexicon dengan skor: {senti_indo.lexicon[word]}")
    else:
        print(f"Kata '{word}' tidak ditemukan dalam lexicon.")
