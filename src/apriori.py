from mlxtend.frequent_patterns import apriori

def run_apriori(df):
    df['sections'] = df['sections'].fillna('')
    
    df_sections = df['sections'].str.get_dummies(sep=',')
    
    df_sections = df_sections.astype(bool)
    
    frequent = apriori(
        df_sections,
        min_support=0.1, 
        use_colnames=True
    )
    
    frequent = frequent.sort_values(by='support', ascending=False)
    
    return frequent