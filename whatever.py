import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as spio

def preprocess(file):
    '''Format a teb-delimited dataset to a matrix'''
    data = list()
    for line in open(file):
        line = line.strip().split('\t')
        data.append(map(float,line))
    return np.mat(data)

def createCentroids(data, k):
    '''Generate random centroids'''
    rows, cols = data.shape
    centroids = np.zeros((k, cols))
    for i in range(cols):
        centroids[:, i] = np.random.uniform(data[:, i].min(), data[:, i].max(), size=k).T
    return centroids


def kMeans(data, k):
    '''Assign points to closest cluster'''
    centroids = createCentroids(data, k)
    assignments = np.zeros((data.shape[0], 1))
    updated = True
    while updated:
        updated = False
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
        for ind in range(k):
            if assignments == ind:
                pts = data[ind]
                centroids[ind, :] = np.mean(pts, axis=0)
    return assignments


def euclidean(x, y):
    '''Calculate euclidean distance between 2 vectors'''
    return np.sqrt(np.sum((x-y)**2))


if __name__ == "__main__":
    mat = spio.loadmat('mixtureData.mat')
    pointdata = mat['Y']
    print(pointdata.shape)

    centroids = createCentroids(pointdata, 3)
    print(centroids)
    total_iteration = 1000

    cluster_label = kMeans(pointdata, 2)
    #labellist = print_label_data([cluster_label, new_centroids])
    print()

    #kmeans = KMeans(n_clusters=2, random_state=0).fit(pointdata)
    #labellist = kmeans.labels_

    x = []
    y = []
    for points in pointdata:
        x.append(points[0])
        y.append(points[1])

    colors = ['red', 'blue', 'green']
    plt.scatter(x, y, c=cluster_label, cmap=matplotlib.colors.ListedColormap(colors))

    plt.show()