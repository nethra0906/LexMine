import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    df = pd.read_csv("data/legal_dataset.csv")
    
    df['text'] = df['text'].fillna('')
    
    return df

def clean_text(text):
    text = str(text).lower()
    
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def vectorize(df):
    df['clean'] = df['text'].apply(clean_text)
    
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english',    
        ngram_range=(1, 2)     
    )
    
    X = vectorizer.fit_transform(df['clean']).toarray()
    
    return X, vectorizer