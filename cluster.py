import numpy as np
import matplotlib.pyplot as plt
import data
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def get_train_data():
    #return np.array(data.getTraining())
    return np.array(data.getTuples())

def get_test_data():
    #return np.array(data.getTesting())
    return np.array(data.getTuples())

def cluster_train_data(train_data, k):

    kmeans = KMeans(n_clusters =4, random_state=0).fit(train_data)
    print(kmeans.labels_)

    clusters = kmeans.labels_
    train1 = []
    train2 = []
    train3 = []
    train4 = []

    for i in range(len(clusters)):
        if clusters[i] == 0:
            train1.append(train_data[i])
        elif clusters[i] == 1:
            train2.append(train_data[i])
        elif clusters[i] == 2:
            train3.append(train_data[i])
        elif clusters[i] == 3:
            train4.append(train_data[i])


    print(train1)
    print(train2)
    print(train3)
    print(train4)

    plt.subplot(221)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=kmeans.labels_)
    #plt.show()

    return clusters

def cluster_test_data(test_data, k):
    kmeans = KMeans(n_clusters = k, random_state=0).fit(test_data)
    
    clusters = kmeans.labels_
    
    plt.subplot(221)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=kmeans.labels_)
    #plt.show()
    
def predict_test(train_data, train_cluster, test_data, test_cluster, k):
    #logistic = LogisticRegression()
    #logistic.fit(train_data, train_cluster)
    #prediction = logistic.predict(test_data)

    #print(train_cluster)
    #for i in range(len(test_data)):
        #print("X=%s, Predicted=%s" % (test_data[i], prediction[i]))
    
    clf = KNeighborsClassifier(k)
    clf.fit(train_data, train_cluster)
    prediction = clf.predict(test_data)
    print(train_cluster)
    print(prediction)

def cross_validate_hyperparameter():
    
    
def cluster_images():
    # Get the training and testing data
    train_data = get_train_data()
    test_data = get_test_data()

    # Create the initial labels for the training and test data
    # We start off with a hyperparameter of k=3 
    train_cluster = np.array(cluster_train_data(train_data, 3)) 
    test_cluster = np.array(cluster_test_data(test_data, 3))

    # Cross validate for the hyperparameter k to run Nearest Neighbors with
    

    # Use Nearest Neighbors classifier to predict the clusters of test data and score the prediction
    prediction_score = predict_test(train_data, train_cluster, test_data, test_cluster)

if __name__== "__main__":
    cluster_images()
