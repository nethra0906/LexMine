from mlxtend.frequent_patterns import apriori

def run_apriori(df):
    # Step 1: Clean sections column
    df['sections'] = df['sections'].fillna('')
    
    # Step 2: Convert to one-hot encoding
    df_sections = df['sections'].str.get_dummies(sep=',')
    
    # Step 3: Convert to boolean (IMPORTANT FIX)
    df_sections = df_sections.astype(bool)
    
    # Step 4: Run Apriori (lower support for better results)
    frequent = apriori(
        df_sections,
        min_support=0.1,   # changed from 0.2
        use_colnames=True
    )
    
    # Step 5: Sort results (better readability)
    frequent = frequent.sort_values(by='support', ascending=False)
    
    return frequent