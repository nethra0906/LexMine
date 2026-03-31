from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_text

def recommend(input_text, vectorizer, X, df, top_n=5, threshold=0.1):
    
    if not input_text.strip():
        return "Invalid input"

    input_clean = clean_text(input_text)
    
    input_vec = vectorizer.transform([input_clean])
    
    sim = cosine_similarity(input_vec, X)[0]
    
    top_indices = sim.argsort()[-top_n:][::-1]
    
    valid_indices = [i for i in top_indices if sim[i] >= threshold]
    
    if not valid_indices:
        return "No relevant cases found"
    
    results = df.iloc[valid_indices].copy()
    results['similarity'] = [sim[i] for i in valid_indices]
    
    return results