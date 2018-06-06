import numpy as np
import matplotlib.pyplot as plt
import data
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

train_data = np.array(data.getTuples())
#print(train_data)
#train_data = data.getScore().reshape(-1,2)

kmeans = KMeans(n_clusters =4, random_state=0).fit(train_data)
print(kmeans.labels_)

clusters = kmeans.labels_
score1 = []
score2 = []
score3 = []
score4 = []

for i in range(len(clusters)):
    if clusters[i] == 0:
        score1.append(train_data[i])
    elif clusters[i] == 1:
        score2.append(train_data[i])
    elif clusters[i] == 2:
        score3.append(train_data[i])
    elif clusters[i] == 3:
        score4.append(train_data[i])


print(score1)
print(score2)
print(score3)
print(score4)

plt.subplot(221)
plt.scatter(train_data[:, 0], train_data[:, 1], c=kmeans.labels_)
plt.show()

