from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train(X, y):
    
    # Step 1: Split data (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y   # handles imbalance
    )
    
    # Step 2: Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Step 3: Evaluate
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model