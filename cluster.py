import sys
import numpy as np
import matplotlib.pyplot as plt
import data
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def get_data():
    #return np.array(data.getTraining())
    train_data, test_data = data.getTrainTestX()
    return train_data, test_data

def get_test_data():
    #return np.array(data.getTesting())
    return np.array(data.getTuples())

def cluster_data(train_data, test_data, k, silent=True, plot=False):

    if silent == False:
        print(("We will be fitting the training data to {k} clusters and use the clusters to make predictions on the test data.").format(k=k))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(train_data)
    #print(kmeans.labels_)

    clusters = kmeans.labels_
    predictions = kmeans.predict(test_data)
    if plot == True:
        plt.subplot(221)
        plt.scatter(train_data[:, 0], train_data[:, 1], c=kmeans.labels_)
        plt.show()
        
        plt.subplot(221)
        plt.scatter(test_data[:,0], test_data[:, 1], c=predictions)
        plt.show()
    

    return clusters, predictions

#def predict_test_data(test_data, k, silent=True, plot=False):
    
    #if silent == False:
        #print(("We will be fitting the test data to {k} clusters.").format(k=k))

    #kmeans = KMeans(n_clusters = k, random_state=0).fit(test_data)
    
    #clusters = kmeans.labels_
    
    #if plot == True:
        #plt.subplot(221)
        #plt.scatter(test_data[:, 0], test_data[:, 1], c=kmeans.labels_)
        #plt.show()
    #return clusters

def predict_test(train_data, train_cluster, test_data, test_cluster, k):
    
    clf = KNeighborsClassifier(k)
    clf.fit(train_data, train_cluster)
    prediction = clf.predict(test_data)
    #print(train_cluster)
    #print(prediction)

    return prediction

def cross_validate_hyperparameter():
    train, test = get_data()
    print("Cross Validation Scores")
    for i in range (1, 10):
        clusters, predictions = cluster_data(train, test, i,plot=False)
        clf = KNeighborsClassifier(i)
        scores = cross_val_score(clf, train, clusters, cv=3)
        print(scores)
    print("End of Cross Validation Scores")
    return 3

def score_cluster(cluster_prediction, kkn_prediction):

    misclassification_error = 1- np.mean(cluster_prediction==kkn_prediction)
    return misclassification_error

def cluster_images(silent=True, plot=False):
    # Get the training and testing data
    train_data, test_data = get_data()
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # Create the initial labels for the training and test data
    # We start off with a hyperparameter of k=3 
    train_cluster, test_cluster = cluster_data(train_data, test_data, 3, silent, plot) 
    #test_cluster = np.array(cluster_test_data(test_data, 3, silent, plot))

    # Cross validate for the hyperparameter k to run Nearest Neighbors with
    k = cross_validate_hyperparameter()

    # Use Nearest Neighbors classifier to predict the clusters of test data and score the prediction
    kkn_prediction = predict_test(train_data, train_cluster, test_data, test_cluster, k)
    
    # Compare test cluster with the KNN
    # This cluster score is how much the predictions made by KNN and clustering
    cluster_score = score_cluster(test_cluster, kkn_prediction)
    print(cluster_score)

if __name__== "__main__":
    s = sys.argv[1] == 'y'
    p = sys.argv[2] == 'y'
    cluster_images(silent=s, plot=p)
