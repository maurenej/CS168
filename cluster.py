import numpy as np
import matplotlib.pyplot as plt
import data
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import linear_model

train_data = data.getTuples()
#print(train_data)
#train_data = data.getScore().reshape(-1,2)

kmeans = KMeans(n_clusters =4, random_state=0).fit(train_data)
print(kmeans.labels_)

clusters = kmeans.labels_

score0 = []
index0 = []
score1 = []
index1 = []
score2 = []
index2 = []
score3 = []
index3 = []

grouping_scores = []
for i in range(len(clusters)):
    if clusters[i] == 0:
        score0.append(train_data[i])
        index0.append(i)
    elif clusters[i] == 1:
        score1.append(train_data[i])
        index1.append(i)
    elif clusters[i] == 2:
        score2.append(train_data[i])
        index2.append(i)
    elif clusters[i] == 3:
        score3.append(train_data[i])
        index3.append(i)

def bayesian():
    reg = linear_model.BayesianRidge()
    reg.fit(train_data, clusters)
    #BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300, normalize=False, tol=0.001, verbose=False)
    print("Prediction: ")
    print(reg.predict ([[60000, 0.4]]))

def lasso():
    reg = linear_model.Lasso(alpha = 0.1)
    reg.fit(train_data, clusters)
    print("Lasso Prediction: ")
    print(reg.predict ([[0, 0]]))

print('Clusters')
print(clusters)
print('Score0 and Indexes')
print(score0)
print(index0)
print('Score1 and Indexes')
print(score1)
print(index1)
print('Score2 and Indexes')
print(score2)
print(index2)
print('Score3 and Indexes')
print(score3)
print(index3)

#bayesian()
lasso()
#plt.subplot(221)
#plt.scatter(train_data[:, 0], train_data[:, 1], c=kmeans.labels_)
#plt.show()

