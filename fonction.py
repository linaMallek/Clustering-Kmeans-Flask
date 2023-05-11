from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import pandas as pd
import os

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def hierarchical_clustering(X):
    #calculer matrice des distances 
    distances = pdist(X)
    dist_intra_cluster = np.zeros(len(X) - 1)
    dist_inter_cluster = np.zeros(len(X) - 1)
    ratio = np.zeros(len(X) - 1)

    Z = linkage(X, method='ward')

    for i, max_d in enumerate(np.linspace(0, np.max(distances), len(X) - 1)):
        clusters = fcluster(Z, t=max_d, criterion='distance')
        dist_intra_cluster[i] = sum([pdist(X[clusters == j, :], metric='euclidean').mean() for j in set(clusters) if len(X[clusters == j, :]) > 1])
        dist_inter_cluster[i] = pdist(X[clusters == 1, :], metric='euclidean').mean() if len(set(clusters)) == 2 else np.nan

        ratio[i] = dist_intra_cluster[i] / dist_inter_cluster[i] if dist_inter_cluster[i] > 0 else np.nan

    idx_elbow = np.argmax(np.gradient(ratio) < np.max(np.gradient(ratio)) * 0.1)
    max_d = np.linspace(0, np.max(distances), len(X) - 1)[idx_elbow]

    
    nbr_clusters = len(set(clusters))

    for i in range(len(X)):
        plt.scatter(X[clusters==i,0], X[clusters==i,1])

    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/clusters.png')
    plt.clf()  # Clear the figure to release memory
    
    silhouette_avg = silhouette_score(X, clusters)
    return silhouette_avg, nbr_clusters


def kmeans(X, K):
    m, n = X.shape
    tol=1e-4
    
    # Initialiser les centroides aléatoirement
    centroids = X[np.random.choice(m, K), :]
    
    # Initialiser la variable d'arrêt
    diff = tol + 1
    
    while diff > tol:
        # Assigner chaque point de données à son cluster le plus proche
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Calculer les nouveaux centroides pour chaque cluster
        new_centroids = np.zeros((K, n))
        for j in range(K):
            new_centroids[j, :] = X[labels == j, :].mean(axis=0)
        
        # Calculer la différence entre les nouveaux et les anciens centroides
        diff = np.abs(new_centroids - centroids).max()
        
        centroids = new_centroids

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red')
    plt.title("K-Means Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")    
    silhouette_avg = silhouette_score(X, labels)
    
    # Créer le dossier "static" s'il n'existe pas déjà
    if not os.path.exists("static"):
        os.makedirs("static")
    
    # Enregistrer la figure dans le dossier "static"
    plt.savefig("static/plot_us.png")

    return centroids, labels, silhouette_avg


def kmeansT(X, K):
    m, n = X.shape
    tol=1e-4
    
    # Initialiser les centroides aléatoirement
    centroids = X[np.random.choice(m, K), :]
    
    # Initialiser la variable d'arrêt
    diff = tol + 1
    
    while diff > tol:
        # Assigner chaque point de données à son cluster le plus proche
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Calculer les nouveaux centroides pour chaque cluster
        new_centroids = np.zeros((K, n))
        for j in range(K):
            new_centroids[j, :] = X[labels == j, :].mean(axis=0)
        
        # Calculer la différence entre les nouveaux et les anciens centroides
        diff = np.abs(new_centroids - centroids).max()
        
        centroids = new_centroids

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red')
    plt.title("K-Means Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")    
    silhouette_avg = silhouette_score(X, labels)
    
    # Créer le dossier "static" s'il n'existe pas déjà
    if not os.path.exists("static"):
        os.makedirs("static")
    
    # Enregistrer la figure dans le dossier "static"
    plt.savefig("static/plot_them.png")

    return centroids, labels, silhouette_avg