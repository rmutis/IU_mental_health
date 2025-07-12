from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import mixture

def graphic(reduced_data_sources, labels_data_sources, method, k_value):
    mds = MDS(2, random_state=0, n_init=4)
    X_reduced = mds.fit_transform(reduced_data_sources)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_data_sources, cmap="Set2")
    plt.title(f"Cluster-Visualization (MDS) for {method} and k-value = {k_value}")
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.colorbar(label="Cluster")
    plt.show()

def km_method(x, dataframe):
    model = KMeans(n_clusters=x, random_state=42)
    model.fit(dataframe)
    labels = model.predict(dataframe)
    centroids = pd.DataFrame(model.cluster_centers_, columns=dataframe.columns)
    return centroids, labels

def gmm_method(x, dataframe):
    model = mixture.GaussianMixture(n_components=x)
    model.fit(dataframe)
    labels = model.predict(dataframe)
    means = pd.DataFrame(model.means_, columns=dataframe.columns)
    bic = model.bic(dataframe)
    return means, labels, bic

def get_top_ten(input, dataframe):
    importance = input.std(axis=0).sort_values(ascending=False)
    top_importance = importance.iloc[:10].index
    reduced_data = dataframe[top_importance].copy()
    return reduced_data, top_importance

def cluster_desc(dataframe, labels):
    dataframe.loc[:, "Cluster"] = labels
    cluster = dataframe.groupby("Cluster").mean()
    return cluster