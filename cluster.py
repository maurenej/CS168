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

def_train = [[52022, 0.736], [39996, 0.512], [42188, 0.782], [59223, 0.414], [21031, 0.57], [31791, 0.883], [68893, 0.495], [41470, 0.492], [25277, 0.39], [32027, 0.669], [53596, 0.66], [132946, 0.531], [33661, 0.505], [66978, 0.846], [57416, 0.773], [47204, 0.427], [34911, 0.611], [66670, 0.859], [47111, 1.0], [91391, 0.958], [63192, 0.446], [52717, 0.498], [51173, 0.637], [185080, 0.915], [212707, 0.791], [36140, 0.662], [232686, 0.783], [11929, 0.611], [60284, 0.449], [40763, 0.786]]

def_test = [[67135, 0.556], [68074, 0.765], [114652, 0.579], [32253, 0.82], [48491, 0.702], [46770, 0.765], [77672, 0.822], [66649, 0.504], [74520, 0.49], [75667, 0.951], [70296, 0.593], [73844, 0.621], [84664, 0.673], [80799, 0.348], [33886, 0.887], [201945, 0.737], [140342, 0.679], [50194, 0.701], [46712, 0.862], [17066, 0.529], [41811, 0.467], [27516, 0.44], [96176, 0.734], [64459, 0.715], [64539, 0.408], [69547, 0.331], [123213, 0.543], [52066, 0.81], [61977, 0.497], [222554, 0.963]]

def get_data(defdata=False):
    #return np.array(data.getTraining())
    if defdata == False:
        train_data, test_data = data.getTrainTestX()
        return train_data, test_data
    elif defdata == True:
        return def_train, def_test

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

def cluster_images(silent=True, plot=False, defdata=False):
    # Get the training and testing data
    train_data, test_data = get_data(defdata)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    

    # # Create the initial labels for the training and test data
    # # We start off with a hyperparameter of k=3 
    # train_cluster, test_cluster = cluster_data(train_data, test_data, 3, silent, plot) 
    # #test_cluster = np.array(cluster_test_data(test_data, 3, silent, plot))

    # # Cross validate for the hyperparameter k to run Nearest Neighbors with
    # k = cross_validate_hyperparameter()

    # # Use Nearest Neighbors classifier to predict the clusters of test data and score the prediction
    # kkn_prediction = predict_test(train_data, train_cluster, test_data, test_cluster, k)
    
    # # Compare test cluster with the KNN
    # # This cluster score is how much the predictions made by KNN and clustering
    # cluster_score = score_cluster(test_cluster, kkn_prediction)
    # print(cluster_score)

    minimum_error = sys.maxsize
    minimum_k = 0
    best_cluster = []
    # Try KMeans clustering with 2, 3, 4, and 5 clusters
    for i in np.arange(2,5):
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
            minimum_k = i
            best_cluster = test_cluster
            print("BEST CLUSTER: ")
            print(test_cluster)
            print("MIN K: ")
            print(minimum_k)

    if not silent:
        print(("The calculated best grouping of the data involves {k} clusters.").format(k=minimum_k))

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
