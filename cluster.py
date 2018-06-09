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

# cluster.py is the bulk of our project learning algorithm
# The training data images of prostate cancer are grouped using the KMeans clustering algorithm on two features: 
# 1. the size of the prostate 
# 2. the ratio of the central gland to the whole prostate.  
# Taking this same clustering rule, we group the test data as well, using the cluster assignments as labels.
# The training data is fit to the K Nearest Neighbors algorithm using a cross-validated hyperparameter, k, in order to make grouping predictions on the test data.
# Based on the test data’s group predictions made by KNN, we find a number of clusters that minimizes the prediction error between KMeans clustering and KNN. 
# This creates the best grouping of the data. 


# the default training set of tuple data
def_train = [[52022, 0.736], [39996, 0.512], [42188, 0.782], [59223, 0.414], [21031, 0.57], [31791, 0.883], [68893, 0.495], [41470, 0.492], [25277, 0.39], [32027, 0.669], [53596, 0.66], [132946, 0.531], [33661, 0.505], [66978, 0.846], [57416, 0.773], [47204, 0.427], [34911, 0.611], [66670, 0.859], [47111, 1.0], [91391, 0.958], [63192, 0.446], [52717, 0.498], [51173, 0.637], [185080, 0.915], [212707, 0.791], [36140, 0.662], [232686, 0.783], [11929, 0.611], [60284, 0.449], [40763, 0.786]]

# the default testing set of tuple data
def_test = [[67135, 0.556], [68074, 0.765], [114652, 0.579], [32253, 0.82], [48491, 0.702], [46770, 0.765], [77672, 0.822], [66649, 0.504], [74520, 0.49], [75667, 0.951], [70296, 0.593], [73844, 0.621], [84664, 0.673], [80799, 0.348], [33886, 0.887], [201945, 0.737], [140342, 0.679], [50194, 0.701], [46712, 0.862], [17066, 0.529], [41811, 0.467], [27516, 0.44], [96176, 0.734], [64459, 0.715], [64539, 0.408], [69547, 0.331], [123213, 0.543], [52066, 0.81], [61977, 0.497], [222554, 0.963]]

# Returns a split set of training and testing data.
# defdata=False: returns a new combination of training and testing data from the main set
# defdata=True: returns the default training and testing data declared above
def get_data(defdata=False):
    #return np.array(data.getTraining())
    if defdata == False:
        train_data, test_data = data.getTrainTestX()
        return train_data, test_data
    elif defdata == True:
        return def_train, def_test

# Returns the entire set of data tuples
def get_test_data():
    #return np.array(data.getTesting())
    return np.array(data.getTuples())

# Clusters the training data into k groups based on the K-Means algorithm
# Returns: 
#     clusters: list of cluster group number that corresponds to each train_data tuple
#     predictions: list of predicted group number that corresponds to each test_data tuple
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

# With the given cluster group numbers, it predicts the cluster group number for test_data
# Returns: 
#     prediction: list of predicted group number that corresponds to each test_data tuple
def predict_test(train_data, train_cluster, test_data, test_cluster, k):
    print(("K = {k}").format(k=k))
    clf = KNeighborsClassifier(k)
    clf.fit(train_data, train_cluster)
    prediction = clf.predict(test_data)
    #print(train_cluster)
    #print(prediction)

    return prediction


# Repeatedly runs the KNeighborsClassifiers with varying number of folds and different data combinations.
# Finds the number of folds that ensues in the best cross validation prediction score. 
def cross_validate_hyperparameter():
    train, test = get_data()
    avg_scores = []
    for i in range (2, 10):
        clusters, predictions = cluster_data(train, test, i,plot=False)
        clf = KNeighborsClassifier(i)
        scores = cross_val_score(clf, train, clusters, cv=3)
        avg_scores.append(np.mean(scores))
        
    print("Average Cross Validation Scores")
    print(avg_scores)
    max_index = avg_scores.index(max(avg_scores)) + 2
    print(max_index)
    return max_index 

# Calculates the number of times that a cluster was mispredicted. 
# Returns:
#     1. the fraction of the cluster numbers that were misclassified
def score_cluster(cluster_prediction, kkn_prediction):

    misclassification_error = 1- np.mean(cluster_prediction==kkn_prediction)
    return misclassification_error

def cluster_images(silent=True, plot=False, defdata=False):
    # Get the training and testing data
    train_data, test_data = get_data(defdata)
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    minimum_error = sys.maxsize
    minimum_k = 0
    best_cluster = []
    # Try KMeans clustering with 2, 3, 4, and 5 clusters
    for i in np.arange(3,5):
        train_cluster, test_cluster = cluster_data(train_data, test_data, i, silent, plot) 
        # Cross validate for the hyperparameter k to run Nearest Neighbors with
        k = cross_validate_hyperparameter()
        # Use Nearest Neighbors classifier to predict the clusters of test data and score the prediction
        kkn_prediction = predict_test(train_data, train_cluster, test_data, test_cluster, k)
        # Compare test cluster with the KNN
        # This cluster score is how much the predictions made by KNN and clustering
        cluster_score = score_cluster(test_cluster, kkn_prediction)
        if cluster_score < minimum_error:
            minimum_error = cluster_score
            print("MIN ERROR")
            print(minimum_error)
            minimum_k = i
            best_cluster = test_cluster

    if silent == False:
        print(("The calculated grouping with a minimized error of {x} involves {k} clusters").format(x = minimum_error, k=minimum_k))

    plt.subplot(221)
    plt.scatter(test_data[:,0], test_data[:, 1], c=best_cluster)
    plt.show()



if __name__== "__main__":
    s = True
    p = False
    d = False
    if len(sys.argv) > 1:
        s = sys.argv[1] == 'y'
        p = sys.argv[2] == 'y'
        d = sys.argv[3] == 'y'
    cluster_images(silent=s, plot=p, defdata=d)
