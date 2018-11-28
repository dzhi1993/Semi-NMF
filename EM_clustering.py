#####################################################################################
# EM algorithm for clustering Mixture Model and visualization
#
# Date: Nov. 25, 2018
# Author: Da Zhi
#####################################################################################
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from scipy.stats import multivariate_normal

DEBUG = True  # global debugging flag


# -------------------------------
# Global debugging flag - DEBUG
# -------------------------------
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


# -------------------------------------------
# 第 k 个模型的高斯分布密度函数
# 每 i 行表示第 i 个样本在各模型中的出现概率
# 返回一维列表
# -------------------------------------------
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)


# ---------------------------------------------------------------------------------
# E - Step：Calculate the expectation of each of the model
# Input Parameters: Y - the matrix of data points with shape (400, 1)
#                   mu - the mean of each clusters
#                   cov - the covariance
#                   pi - cluster probability distribution of each point
# Return: gamma - the probability of data sample is from the k clusters
# ---------------------------------------------------------------------------------
def getExpectation(Y, mu, cov, pi):
    # number of samples (data points)
    N = Y.shape[0]
    # number of models (clusters)
    K = pi.shape[0]

    # The number of samples and mixture model are restricted equal to 1 to avoid different return types
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    # Initialize gamma, size of (400, k)
    gamma = np.mat(np.zeros((N, K)))

    # 计算各模型中所有样本出现的概率，行对应样本，列对应模型
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    # 计算每个模型对每个样本的响应度
    for k in range(K):
        gamma[:, k] = pi[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma


#########################################################################
# M 步：iteration computation of the probability distribution
#       to maximize the expectation
# Y 为样本矩阵，gamma 为响应度矩阵
#########################################################################
def maximize(Y, gamma):
    # 样本数和特征数
    N, D = Y.shape
    # 模型数
    K = gamma.shape[1]

    #初始化参数值
    mu = np.zeros((K, D))
    cov = []
    pi = np.zeros(K)

    # 更新每个模型的参数
    for k in range(K):
        # 第 k 个模型对所有样本的响应度之和
        Nk = np.sum(gamma[:, k])
        # 更新 mu
        # 对每个特征求均值
        for d in range(D):
            mu[k, d] = np.sum(np.multiply(gamma[:, k], Y[:, d])) / Nk
        # 更新 cov
        cov_k = np.mat(np.zeros((D, D)))
        for i in range(N):
            cov_k += gamma[i, k] * (Y[i] - mu[k]).T * (Y[i] - mu[k]) / Nk
        cov.append(cov_k)
        # 更新 pi
        pi[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, pi


######################################################
# Data pre-processing
# Scaling all point data within [0, 1]
######################################################
def scale_data(Y):
    # 对每一维特征分别进行缩放
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y


#################################################################
# Initialization the parameters of mixture model
# Input parameters: shape - the shape of data points (400, 2)
#                   K - the number of models (clusters)
# Return: mu - randomly generate, size of (k, 2)
#         covariance - k covariance matrices - identity
#         pi - the initial probability distribution - 1/k
#################################################################
def init_params(shape, k):
    N, D = shape
    mu = np.random.rand(k, D)
    cov = np.array([np.eye(D)] * k)
    pi = np.array([1.0 / k] * k)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "pi:", pi, sep="\n")
    return mu, cov, pi


######################################################
# 高斯混合模型 EM 算法
# 给定样本矩阵 Y，计算模型参数
# K 为模型个数
# times 为迭代次数
######################################################
def GMM_EM(datapoint, k, times):
    datapoint = scale_data(datapoint)
    mu, cov, pi = init_params(datapoint.shape, k)
    for i in range(times):
        gamma = getExpectation(datapoint, mu, cov, pi)
        mu, cov, pi = maximize(datapoint, gamma)
    debug("{sep} Result {sep}".format(sep="-" * 20))
    debug("mu:", mu, "cov:", cov, "pi:", pi, sep="\n")
    return mu, cov, pi


if __name__ == "__main__":
    # debugging mode flag
    DEBUG = True

    # Load data from .mat file as array-like data
    mat = spio.loadmat("mixtureData.mat")
    Y = mat['Y']
    print(Y.shape)
    matY = np.matrix(Y, copy=True)

    # the number of clusters
    K = 3

    # 计算 GMM 模型参数
    mu, cov, pi = GMM_EM(matY, K, 100)

    # 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
    N = Y.shape[0]
    # 求当前模型参数下，各模型对样本的响应度矩阵
    gamma = getExpectation(matY, mu, cov, pi)
    # 对每个样本，求响应度最大的模型下标，作为其类别标识
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    # 将每个样本放入对应类别的列表中
    class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
    class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
    class3 = np.array([Y[i] for i in range(N) if category[i] == 2])

    # 绘制聚类结果
    plt.plot(class1[:, 0], class1[:, 1], 'rs')
    plt.plot(class2[:, 0], class2[:, 1], 'bo')
    plt.plot(class3[:, 0], class3[:, 1], 'go')
    plt.legend(loc="best")
    plt.title("Mixture Model Clustering By EM Algorithm")
    plt.show()
