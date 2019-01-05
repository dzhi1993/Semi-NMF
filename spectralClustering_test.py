import numpy as np
from sklearn import datasets
import scipy.io as spio
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import matplotlib.pyplot as plt

#X, y = datasets.make_blobs(n_samples=500, n_features=10, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
#print(X, X.shape)

X = np.array([[1, 1], [2, 1], [1, 0],
              [4, 7], [3, 5], [3, 6]])
clustering = SpectralClustering(n_clusters=2,
        assign_labels="discretize",
        random_state=0).fit_predict(X)
#print(clustering.labels_)

print(X.shape)
print(clustering)
