from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cluster_data(X):

    pca = PCA(n_components=10)
    X_reduced = pca.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    model = DBSCAN(eps=1.5, min_samples=3)
    labels = model.fit_predict(X_scaled)

    return labels