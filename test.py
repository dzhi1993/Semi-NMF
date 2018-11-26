import numpy as np
import scipy.io as spio
import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Compute the euclidean distance of centroid and each data point
def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))


def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]


def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids) / 2


def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)

    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point],
                                                                      centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            centroids[label[0]] = compute_new_centroids(label[1], centroids[label[0]])

            if iteration == (total_iteration - 1):
                cluster_label.append(label)

    return [cluster_label, centroids]


def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    labellist=[]
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
        labellist.append(data[0])

    thisarray = np.asarray(labellist)
    print(thisarray.shape, len(labellist))
    print("Last centroids position: \n {}".format(result[1]))

    return thisarray


def initialize_point(data, k):
    centroids = []
    randomnumbers = random.sample(range(0, len(data)-1), k)
    for i in randomnumbers:
        centroids.append(data[i])

    return np.array(centroids)


if __name__ == "__main__":
    mat = spio.loadmat('mixtureData.mat')
    pointdata = mat['Y']
    print(pointdata.shape)

    centroids = initialize_point(pointdata, 3)
    print(centroids)
    total_iteration = 1000

    [cluster_label, new_centroids] = iterate_k_means(pointdata, centroids, total_iteration)
    #labellist = print_label_data([cluster_label, new_centroids])
    print()

    kmeans = KMeans(n_clusters=2, random_state=0).fit(pointdata)
    labellist = kmeans.labels_

    x = []
    y = []
    for points in pointdata:
        x.append(points[0])
        y.append(points[1])

    colors = ['red', 'blue', 'green']
    plt.scatter(x, y, c=labellist, cmap=matplotlib.colors.ListedColormap(colors))

    plt.show()