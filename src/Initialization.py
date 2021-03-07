import numpy as np
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class Initialization:

    def __init__(self):
        pass

    def kmeans_init(self, X, max_clust, plot=False):
        # cluster data into K=1..max_clust clusters
        K = range(1, max_clust)

        KM = [kmeans(X, k) for k in K]
        centroids = [cent for (cent, var) in KM]  # cluster centroids
        # avgWithinSS = [var for (cent,var) in KM] # mean within-cluster sum of squares

        # alternative: scipy.cluster.vq.vq
        # Z = [vq(X,cent) for cent in centroids]
        # avgWithinSS = [sum(dist)/X.shape[0] for (cIdx,dist) in Z]

        # alternative: scipy.spatial.distance.cdist
        D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
        cIdx = [np.argmin(D, axis=1) for D in D_k]
        dist = [np.min(D, axis=1) for D in D_k]
        avgWithinSS = [sum(d) / X.shape[0] for d in dist]

        if plot:
            kIdx = 2

            # elbow curve
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(K, avgWithinSS, 'b*-')
            ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
                    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
            plt.grid(True)
            plt.xlabel('Number of clusters')
            plt.ylabel('Average within-cluster sum of squares')
            plt.title('Elbow for KMeans clustering')

            # scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # ax.scatter(X[:,2],X[:,1], s=30, c=cIdx[k])
            clr = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i in range(K[kIdx]):
                ind = (cIdx[kIdx] == i)
                ax.scatter(X[ind, 2], X[ind, 1], s=30, c=clr[i], label='Cluster %d' % i)
            plt.xlabel('Petal Length')
            plt.ylabel('Sepal Width')
            plt.title('Iris Dataset, KMeans clustering with K=%d' % K[kIdx])
            plt.legend()

            plt.show()