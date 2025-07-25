import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.manifold import MDS


# Function to get amount of na-values per feature
def count_na(dataframe):
    no_value = pd.DataFrame(dataframe.isna().sum(), columns=["number_na"])
    print(no_value)


# Function for cluster visualisation on 2 dimensions
def graphic(reduced_data_sources, labels_data_sources, method, k_value, name):
    mds = MDS(2, random_state=0, n_init=4)
    x_reduced = mds.fit_transform(reduced_data_sources)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=labels_data_sources, cmap="Set2")
    plt.title(f"Cluster-Visualization (MDS) for {method}, k-value = {k_value} and datasource = {name}")
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.colorbar(label="Cluster")
    plt.show()


# Function for K-Means modelling
def km_method(x, dataframe):
    model = KMeans(n_clusters=x, random_state=42)
    model.fit(dataframe)
    labels = model.predict(dataframe)
    centroids = pd.DataFrame(model.cluster_centers_, columns=dataframe.columns)
    return centroids, labels


# Function for GMM modelling
def gmm_method(x, dataframe):
    model = mixture.GaussianMixture(n_components=x)
    model.fit(dataframe)
    labels = model.predict(dataframe)
    means = pd.DataFrame(model.means_, columns=dataframe.columns)
    bic = model.bic(dataframe)
    return means, labels, bic


# Function to retrieve top ten features based on highest variation
def get_top_ten(centroids, dataframe):
    importance = centroids.std(axis=0).sort_values(ascending=False)
    top_importance = importance.iloc[:10].index
    reduced_data = dataframe[top_importance].copy()
    return reduced_data, top_importance


# Function to retrieve the mean values of the features of a cluster
def cluster_desc(dataframe, labels):
    dataframe.loc[:, "Cluster"] = labels
    cluster = dataframe.groupby("Cluster").mean()
    return cluster
