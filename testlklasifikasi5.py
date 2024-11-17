from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Memuat data
df = pd.read_csv('data_lexicon_labeled_numbered.csv')

# Memastikan kolom yang diperlukan ada
if 'preprocessed_text' in df.columns and 'label_sentiment_numbered' in df.columns:
    texts = df['preprocessed_text'].fillna('')  # Mengisi NaN dengan string kosong
    labels = df['label_sentiment_numbered']
else:
    raise ValueError("Kolom 'preprocessed_text' atau 'label_sentiment_numbered' tidak ditemukan dalam DataFrame.")

# Vektorisasi teks dengan TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Pembagian data (gunakan stratifikasi jika diperlukan)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42, stratify=labels)

# 1. Support Vector Machine
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print("Support Vector Machine")
print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# 2. Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("\nLogistic Regression")
print("Akurasi:", accuracy_score(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# 3. Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("\nRandom Forest Classifier")
print("Akurasi:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# 4. Naive Bayes Classifier (MultinomialNB)
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)
print("\nNaive Bayes Classifier")
print("Akurasi:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# 5. K-Nearest Neighbors (KNN)
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)
print("\nK-Nearest Neighbors (KNN)")
print("Akurasi:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))

# Membuat salinan dari DataFrame asli yang berisi data uji
df_test = df.iloc[y_test.index].copy()  # Menggunakan index yang sama dengan data uji

# Menambah kolom untuk setiap hasil prediksi model
df_test['Prediksi SVM'] = y_pred
df_test['Prediksi Logistic Reggresion'] = y_pred_log_reg
df_test['Prediksi Random Forest'] = y_pred_rf
df_test['Prediksi Naive Bayes Classifier'] = y_pred_nb
df_test['Prediksi KNN'] = y_pred_knn

# Menampilkan DataFrame dengan kolom baru
print(df_test.head())

# Menyimpan DataFrame ke file CSV jika diperlukan
df_test.to_csv('hasil_prediksi_berbagai_model.csv', index=False)