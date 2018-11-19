import numpy as np
from sklearn import datasets
import scipy.io as spio
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import matplotlib.pyplot as plt

#X, y = datasets.make_blobs(n_samples=500, n_features=10, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
#print(X, X.shape)

mat = spio.loadmat('mbeta_Cerebellum_grey_all.mat')
betaValueStructure = mat['B']

betaValueMatrix = betaValueStructure[1, 1]
print(betaValueMatrix.shape)

# data = np.random.rand(13, 25)
data = betaValueMatrix.transpose()

y_pred = SpectralClustering(n_clusters=10, eigen_solver='amg').fit_predict(data)
print(y_pred, "\n", y_pred.shape)
filename = "spectralClusteringOutput_gamma=.mat"
spio.savemat(filename, mdict={'data': y_pred})
print("CH score with gamma= is: ", metrics.calinski_harabaz_score(data, y_pred))

plt.imshow(y_pred, aspect='auto')
plt.show()

print(y_pred, y_pred.shape, "Done!")

