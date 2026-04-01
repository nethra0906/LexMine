from src.preprocess import load_data, vectorize, clean_text
from src.train_model import train
from src.clustering import cluster_data
from src.apriori import run_apriori
from src.recommend import recommend

# ------------------ LOAD DATA ------------------
df = load_data()

if 'outcome' in df.columns:
    df = df[df['outcome'] != "unknown"]

# ------------------ PREPROCESS ------------------
X, vectorizer = vectorize(df)

# ------------------ TRAIN MODEL ------------------
model = train(X, df['outcome'])

# ------------------ CLUSTERING ------------------
df['cluster'] = cluster_data(X)

# ------------------ APRIORI ------------------
patterns = run_apriori(df)

# ------------------ RECOMMENDATION ------------------
query = "stabbing case"

recommendations = recommend(query, vectorizer, X, df)

print("\nRecommended Cases:")

if isinstance(recommendations, str):
    print(recommendations)
else:
    cols = ['text', 'outcome']
    
    if 'similarity' in recommendations.columns:
        cols.append('similarity')
    
    print(recommendations[cols])

# ------------------ PREDICTION ------------------

input_clean = clean_text(query)

input_vec = vectorizer.transform([input_clean])

prediction = model.predict(input_vec)[0]

print("\nPredicted Outcome:", prediction)