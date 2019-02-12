from sklearn import datasets
import matplotlib
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Making different type of data set
# n_samples = 1500
# X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)  # circle dataset
# X, y = datasets.make_moons(n_samples=n_samples, noise=.05)  # Moon dataset
# X, y = datasets.make_blobs(n_samples=500,
#                            n_features=10,
#                            centers=5,
#                            cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4],
#                            random_state=11)  # blob dataset

# X = np.array([[1, 1], [2, 1], [1, 0],
#               [4, 7], [3, 5], [3, 6]])
# print(X, X.shape)

mat = spio.loadmat('groupData_sc2.mat')
groupData = mat['X_C']

clustering = SpectralClustering(n_clusters=10,
                                eigen_solver='arpack',
                                affinity="nearest_neighbors").fit(groupData.transpose())
print(clustering.labels_.shape, clustering.labels_)
# spio.savemat('clusters.mat', {"Y": clustering.labels_})

# Make the clustering result bestG type
clustering_result = np.zeros((25275, 10))
for i in range(clustering_result.shape[0]):
    clustering_result[i, clustering.labels_[i]] = 0.6

spio.savemat('spec.mat', {"bestG": clustering_result})

# print(clustering, clustering.affinity_matrix_.shape, clustering.affinity_matrix_)
# print(X[:, 0], y.shape)
# print(clustering.shape)

# Plotting the clustering result
# colors = ['red', 'blue']
# plt.figure()
# # plt.scatter(X[0, 0], y[0, 1])
# plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, marker='.', cmap=matplotlib.colors.ListedColormap(colors))
# plt.title('K = 2')
# plt.show()
