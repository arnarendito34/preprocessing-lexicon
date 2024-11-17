import pandas as pd

df = pd.read_csv('data_labellexicon-AlfonsusAntero-1204210085.csv')
print(df['label_sentiment'].value_counts())