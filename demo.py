import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as pt
import matplotlib
from matplotlib import colors
from sklearn.cluster import SpectralClustering
from sklearn import cluster
from sklearn.decomposition import PCA
from scipy.cluster import vq
from scipy.spatial import distance

matplotlib.style.use('ggplot')

dataset = pd.read_csv('pca_example', delim_whitespace=True, index_col=0)

print(dataset)

#Create imshow to show correlations between features
pt.imshow(dataset.corr(), cmap=pt.cm.Blues, interpolation='nearest')
pt.colorbar()
tick_marks = [i for i in range(len(dataset.columns))]
pt.xticks(tick_marks, dataset.columns, rotation='vertical')
pt.yticks(tick_marks, dataset.columns)
pt.savefig('imshow.png')

#Run PCA
pca = PCA(n_components=2)
pca.fit(dataset)
data2d = pca.transform(dataset)

dataset2d = pd.DataFrame(data2d)
dataset2d.index = dataset.index
dataset2d.columns = ['PC1', 'PC2']
print(dataset2d)

#Run K-Means
num_of_centroids = 2
centroids = vq.kmeans(dataset2d, num_of_centroids)
assignment,vqs = vq.vq(dataset2d, centroids[0])

colormap = np.array(['red', 'green', 'blue', 'yellow', 'black'])
centroids = pd.DataFrame(list(centroids[0]))
centroids.columns = ['PC1', 'PC2']
fig = pt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataset2d.PC1, dataset2d.PC2, marker='o', alpha=0.9, c=colormap[assignment])
ax2 = fig.add_subplot(111)
ax2.scatter(centroids.PC1, centroids.PC2, marker='H', c=colormap[centroids.index], alpha=0.9, linewidths=3, s=169)
for i in range (len(dataset.index)):
    ax.annotate(dataset2d.index[i], (dataset2d.iloc[i].PC1, dataset2d.iloc[i].PC2))
fig.savefig('test.png')

#Create laplacian
laplacian_matrix = np.zeros(((len(dataset2d.index)), (len(dataset2d.index))))
max_distant_between_points = 3
for i in range (len(dataset2d.index)):
    num_of_connections = 0
    for j in range (i, len(dataset2d.index)):
        if i != j:
            a = (dataset2d.iloc[[i]].PC1, dataset2d.iloc[[i]].PC2)
            b = (dataset2d.iloc[[j]].PC1, dataset2d.iloc[[j]].PC2)
            distance_between_a_b = distance.euclidean(a, b)
            if distance_between_a_b < max_distant_between_points:
                laplacian_matrix[i][j] = 1
                laplacian_matrix[j][i] = 1
                num_of_connections = num_of_connections + 1
    laplacian_matrix[i][i] = num_of_connections
print(type(laplacian_matrix))
laplacian_matrix = np.array(laplacian_matrix)

#Run spectral clustering
spectral_clust = cluster.spectral_clustering(affinity=laplacian_matrix, n_clusters=3)
print(spectral_clust)
colormap = np.array(['red', 'green', 'blue', 'yellow', 'black'])
ax = dataset2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(4,4), c=colormap[spectral_clust])
for i in range (len(dataset.index)):
    ax.annotate(dataset2d.index[i], (dataset2d.iloc[i].PC2, dataset2d.iloc[i].PC1))
fig = ax.get_figure()
fig.savefig('spectral.png')




