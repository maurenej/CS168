import os
import nrrd #pip install pynrrd, if pynrrd is not already installed
import numpy as np
from sklearn.model_selection import train_test_split

# data.py handles all of the data processing needed for cluster.py
# the images are loaded from nrrd3T and nrrdDx directories

#total volume scores from nrrdDx
scoresDx = [64539, 63192, 84664, 61977, 68893, 201945, 41470, 66649, 69547, 59223, 123213, 114652, 67135, 39996, 80799, 52717, 41811, 185080, 51173, 132946, 64459, 212707, 96176, 60284, 73844, 47204, 140342, 74520, 70296, 232686]

#central zone scores from nrrdDx
cz_scoresDx = [26323, 28175, 56945, 30785, 34122, 148846, 20414, 33568, 23019, 24493, 66861, 66418, 37318, 20464, 28082, 26245, 19511, 169363, 32595, 70533, 46099, 168267, 70607, 27046, 45861, 20173, 95230, 36517, 41718, 182232]

#ratios from nrrdDx
ratiosDx = [0.408, 0.446, 0.673, 0.497, 0.495, 0.737, 0.492, 0.504, 0.331, 0.414, 0.543, 0.579, 0.556, 0.512, 0.348, 0.498, 0.467, 0.915, 0.637, 0.531, 0.715, 0.791, 0.734, 0.449, 0.621, 0.427, 0.679, 0.49, 0.593, 0.783]

#tuples from nrrdDx
tuplesDx = [[64539, 0.408], [63192, 0.446], [84664, 0.673], [61977, 0.497], [68893, 0.495], [201945, 0.737], [41470, 0.492], [66649, 0.504], [69547, 0.331], [59223, 0.414], [123213, 0.543], [114652, 0.579], [67135, 0.556], [39996, 0.512], [80799, 0.348], [52717, 0.498], [41811, 0.467], [185080, 0.915], [51173, 0.637], [132946, 0.531], [64459, 0.715], [212707, 0.791], [96176, 0.734], [60284, 0.449], [73844, 0.621], [47204, 0.427], [140342, 0.679], [74520, 0.490], [70296, 0.593], [232686, 0.783]]

#total volume scores from nrrd3T
scores3T = [32253, 52022, 50194, 57416, 32027, 40763, 11929, 33886, 91391, 33661, 31791, 48491, 77672, 21031, 34911, 42188, 222554, 25277, 47111, 17066, 66978, 36140, 52066, 75667, 46712, 66670, 53596, 68074, 46770, 27516]

#central zone scores from nrrd3T
cz_scores3T = [26451, 38265, 35187, 44365, 21413, 32050, 7294, 30071, 87558, 17010, 28073, 34054, 63883, 11984, 21346, 33003, 214263, 9850, 47111, 9020, 56663, 23940, 42161, 71961, 40269, 57302, 35365, 52073, 35761, 12110]

#ratios from nrrd3T
ratios3T = [0.82, 0.736, 0.701, 0.773, 0.669, 0.786, 0.611, 0.887, 0.958, 0.505, 0.883, 0.702, 0.822, 0.57, 0.611, 0.782, 0.963, 0.39, 1.0, 0.529, 0.846, 0.662, 0.81, 0.951, 0.862, 0.859, 0.66, 0.765, 0.765, 0.44]

#tuples from nrrd3T
tuples3T = [[32253, 0.820], [52022, 0.736], [50194, 0.701], [57416, 0.773], [32027, 0.669], [40763, 0.786], [11929, 0.611], [33886, 0.887], [91391, 0.958], [33661, 0.505], [31791, 0.883], [48491, 0.702], [77672, 0.822], [21031, 0.570], [34911, 0.611], [42188, 0.782], [222554, 0.963], [25277, 0.390], [47111, 1.000], [17066, 0.529], [66978, 0.846], [36140, 0.662], [52066, 0.810], [75667, 0.951], [46712, 0.862], [66670, 0.859], [53596, 0.660], [68074, 0.765], [46770, 0.765], [27516, 0.440]] 


#######      datatype == 0 if data from nrrdDx
#######      datatype == 1 if data from nrrd3T 


#load all the .nrrd files into one list
def getData(datatype):
    
    images = [ ]
    if datatype == 0:
        for filen in os.listdir('nrrdDx'):
            if filen.endswith('.nrrd'):
                print(filen)
                data, header = nrrd.read(os.path.join('nrrdDx/'+ filen))
                images.append(data)
    elif datatype == 1:
        for filen in os.listdir('nrrd3T'):
            if filen.endswith('.nrrd'):
                print(filen)
                data, header = nrrd.read(os.path.join('nrrd3T/'+ filen))
                images.append(data)
    return images

#returns the list of total volume scores
def getScore(datatype):
    if datatype == 0:
        return scoresDx
    elif datatype == 1:
        return scores3T

#returns the list of Central Zone volume scores
def getCZScores(datatype):
    if datatype == 0:
        return cz_scoresDx
    elif datatype == 1:
        return cz_scores3T

#returns the list of ratios
def getRatios(datatype):
    if datatype == 0:
        return ratiosDx
    elif datatype == 1:
        return ratios3T

#retunrs the list of tuples
def getTuples(datatype):
    if datatype == 0:
        return tuplesDx
    elif datatype == 1:
        return tuples3T

#combines the two tuple lists together and randomly selects 
#1/2 of data for training and 1/2 of data for testing
def getTrainTestX():
    X = tuplesDx + tuples3T
    X_train, X_test = train_test_split(X, test_size=0.5)
    return [X_train, X_test]

#calculates the total volume scores by iterating through each 3D-image
#the data then saved in data structures at top of file
def calculateScores(datatype, start, end):
    images = getData(datatype)
    image_scores = [ ]
    score = 0
    for a in range (start, end):
        img = images[a]
        for i in range (0, len(img)):
            for j in range (0, len(img)):
                for k in range (0, len(img[0][0])):
                    if img[i][j][k] > 0:
                        score = score + 1
        image_scores.append(score)
        score = 0
    print(image_scores)

#calculates the central zone volume scores by iterating through each 3D-image
#the data then saved in data structures at top of file
def calculateCZ(datatype, start, end):
    images = getData(datatype)
    image_scores = [ ]
    score = 0
    for a in range (start, end):
        img = images[a]
        for i in range (0, len(img)):
            for j in range (0, len(img)):
                for k in range (0, len(img[0][0])):
                    if img[i][j][k] > 1:
                        score = score + 1
        image_scores.append(score)
        score = 0
    print(image_scores)

#calculates the ratios by iterating through the score and cz_score lists
#the data then saved in data structures at top of file
def calculateRatios(datatype):
    ratios = []
    if datatype == 0:
        for i in range (0, len(cz_scoresDx)):
            ratios.append(round(cz_scoresDx[i]/scoresDx[i],3))
    elif datatype == 1:
        for i in range (0, len(cz_scores3T)):
            ratios.append(round(cz_scores3T[i]/scores3T[i],3))
            
    print(ratios)

#writes tuples to an external file for easy access
#the data then saved in data structures at top of file
def createTuples(datatype):
    if datatype == 0:
        f= open("dataDx.txt","w+")
        for i in range(len(scores)):
            f.write("[%d, " % scoresDx[i]) 
            f.write("%.3f], " % ratiosDx[i]) 
        f.write("\r\n")
        f.close
    elif datatype == 1:
        f= open("data3T.txt","w+")
        for i in range(len(scores3T)):
            f.write("[%d, " % scores3T[i]) 
            f.write("%.3f], " % ratios3T[i]) 
        f.write("\r\n")
        f.close


