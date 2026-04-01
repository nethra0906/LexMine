from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_text

def recommend(input_text, vectorizer, X, df, top_n=5, threshold=0.2):
    
    input_clean = clean_text(input_text)
    input_words = set(input_clean.split())
    
    STOP_WORDS = {"case", "involving", "with", "and"}
    input_words = input_words - STOP_WORDS
    
    input_vec = vectorizer.transform([input_clean])
    
    sim = cosine_similarity(input_vec, X)[0]
    
    scored_results = []
    
    for i, score in enumerate(sim):
        
        if score < threshold:
            continue
        
        case_text = df.iloc[i]['clean']
        case_words = set(case_text.split())
        
        case_words = case_words - STOP_WORDS
        
        common_words = input_words.intersection(case_words)
        
        if len(common_words) == 0:
            continue
        
        adjusted_score = score + (0.1 * len(common_words))
        
        scored_results.append((i, adjusted_score))
    

    scored_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
    
    results = scored_results[:top_n]
    
    if not results:
        fallback_indices = sim.argsort()[-top_n:][::-1]
        
        final = df.iloc[fallback_indices].copy()
        final['similarity'] = sim[fallback_indices]   # ✅ FIXED
        
        return final
    
    indices = [i for i, _ in results]
    scores = [s for _, s in results]
    
    final = df.iloc[indices].copy()
    final['similarity'] = scores
    
    return final