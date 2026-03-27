from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_text

def recommend(input_text, vectorizer, X, df, top_n=5, threshold=0.1):
    
    # Step 1: Handle empty input
    if not input_text.strip():
        return "Invalid input"
    
    # Step 2: Clean input
    input_clean = clean_text(input_text)
    
    # Step 3: Vectorize (NO .toarray())
    input_vec = vectorizer.transform([input_clean])
    
    # Step 4: Compute similarity
    sim = cosine_similarity(input_vec, X)[0]
    
    # Step 5: Get top matches
    top_indices = sim.argsort()[-top_n:][::-1]
    
    # Step 6: Filter low similarity results
    valid_indices = [i for i in top_indices if sim[i] >= threshold]
    
    if not valid_indices:
        return "No relevant cases found"
    
    # Step 7: Prepare results
    results = df.iloc[valid_indices].copy()
    results['similarity'] = [sim[i] for i in valid_indices]
    
    return results