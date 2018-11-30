##################################################################################################
# K-means clustering algorithm implementation and visualization for a 2D data points set
# Time: Nov. 24, 2018
# Author: Da Zhi
##################################################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.io as spio


# Load data from .mat file as array-like data
def loaddata(filename):
    mat = spio.loadmat(filename)
    pointdata = mat['Y']
    print(pointdata.shape)
    return pointdata


# Initialize random k centroids at start
def createCentroids(data, k):
    rows, cols = data.shape
    centroids = np.zeros((k, cols))
    for i in range(cols):
        centroids[:, i] = np.random.uniform(data[:, i].min(), data[:, i].max(), size=k).T
    return centroids


# The main k-means methods, assigning each data point to closest cluster center.
# Input: data - the array like data points to be clustered
#        k - number of clusters
# Output: returns a 1-D array which assigned cluster labels to each of the points
def kmeans(data, k):
    centroids = createCentroids(data, k)
    print("The initial centers are: " + str(centroids))

    # Initialize the return clusters assignment
    assignments = np.zeros((data.shape[0], 1))
    updated = True
    maxiter = 100  # Maximum iteration number
    iter = 0
    while updated and iter < maxiter:
        updated = False
        # The major iteration of kmeans clustering
        # Step 1: Calculate the euclidean distances between each of the point with the current centroids
        #         If a smaller distance was found, then change the cluster label of current point to the new one
        for i in range(data.shape[0]):
            current = data[i, :]
            min_dist = np.inf
            for j in range(k):
                curr_dist = euclidean(current, centroids[j, :])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    index = j
                if assignments[i, 0] != index:
                    assignments[i, 0] = index
                    updated = True
        # Step 2: calculate the mean of current clusters and elect the new center
        for ind in range(k):
            pts = []
            for currentindex in range(len(assignments)):
                if assignments[currentindex] == ind:
                    pts.append(data[currentindex])
            centroids[ind, :] = np.mean(pts, axis=0)
        iter = iter + 1

    return assignments


# Calculate the euclidean distance between two point
def euclidean(x, y):
    return np.sqrt(np.sum((x-y)**2))


if __name__ == "__main__":
    # Load data
    datapoint = loaddata('mixtureData.mat')

    # Do k-means clustering 4 times, k = 2, 3, 4, 5
    actualabel = []
    for index in range(2, 6):
        cluster_label = kmeans(datapoint, index)
        reallabel = cluster_label.ravel()
        actualabel.append(reallabel)

        # Make comparison with standard kmeans method from sklearn lib
        #kmeans = KMeans(n_clusters=i, random_state=0).fit(datapoint)
        #actualabel.append(kmeans.labels_)

    x = []
    y = []
    for points in datapoint:
        x.append(points[0])
        y.append(points[1])

    # k=2
    colors = ['red', 'blue']
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(x, y, c=actualabel[0], marker='.', cmap=matplotlib.colors.ListedColormap(colors))
    plt.title('K = 2')

    # k=3
    colors = ['red', 'blue', 'green']
    plt.subplot(2, 2, 2)
    plt.scatter(x, y, c=actualabel[1], marker='.', cmap=matplotlib.colors.ListedColormap(colors))
    plt.title('K = 3')

    # k=4
    colors = ['red', 'blue', 'green', 'yellow']
    plt.subplot(2, 2, 3)
    plt.scatter(x, y, c=actualabel[2], marker='.', cmap=matplotlib.colors.ListedColormap(colors))
    plt.title('K = 4')

    # k=5
    colors = ['red', 'blue', 'green', 'yellow', 'pink']
    plt.subplot(2, 2, 4)
    plt.scatter(x, y, c=actualabel[3], marker='.', cmap=matplotlib.colors.ListedColormap(colors))
    plt.title('K = 5')

    plt.show()