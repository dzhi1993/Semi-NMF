##################################################################################################
# K-means clustering algorithm implementation and visualization for a 2D data points set
# Time: Nov. 24, 2018
# Author: Da Zhi
##################################################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import scipy.io as spio


# Load data from .mat file as array-like data
def loaddata(filename):
    mat = spio.loadmat(filename)
    pointdata = mat['X']
    print(pointdata.shape)
    return pointdata


if __name__ == "__main__":
    # Load data
    datapoint = loaddata('mixtureData2.mat')

    # Do k-means clustering 4 times, k = 2, 3, 4, 5

    clustering = SpectralClustering(n_clusters=3,
                                    assign_labels="discretize",
                                    random_state=0).fit_predict(datapoint)

    x = []
    y = []
    for points in datapoint:
        x.append(points[0])
        y.append(points[1])

    # k=2
    colors = ['red', 'blue', 'green']
    plt.figure()

    plt.scatter(x, y, c=clustering, marker='.', cmap=matplotlib.colors.ListedColormap(colors))
    plt.title('spectral clustering')
    plt.show()
