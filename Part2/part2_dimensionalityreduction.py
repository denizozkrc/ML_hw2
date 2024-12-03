import pickle

import torch
from pca import PCA
from autoencoder import AutoEncoder
import numpy as np
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))


def scatter_plot(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar()
    plt.title(title)
    plt.show()


def dimReductionFourMethods(dataset):

    # PCA
    pca = PCA(projection_dim=2)
    pca.fit(dataset)
    pca_dataset = pca.transform(dataset)
    scatter_plot(pca_dataset, KMeans(n_clusters=3).fit_predict(pca_dataset), "PCA")

    # AE
    dataset_tensor = torch.FloatTensor(dataset)
    autoencoder = AutoEncoder(input_dim=np.shape(dataset_tensor)[1], projection_dim=2, learning_rate=0.001, iteration_count=1000)
    autoencoder.fit(dataset_tensor)
    ae_dataset = autoencoder.transform(dataset_tensor)
    scatter_plot(ae_dataset, KMeans(n_clusters=3).fit_predict(ae_dataset), "Autoencoder")

    # tsne
    tsne = TSNE(n_components=2)
    tsne_dataset = tsne.fit_transform(dataset)
    scatter_plot(tsne_dataset, KMeans(n_clusters=3).fit_predict(tsne_dataset), "t-SNE")

    # umap
    umap_obj = umap.UMAP(n_components=2)
    umap_dataset = umap_obj.fit_transform(dataset)
    scatter_plot(umap_dataset, KMeans(n_clusters=3).fit_predict(umap_dataset), "UMAP: Dataset 2")



dimReductionFourMethods(dataset1)
