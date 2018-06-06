import os
import nrrd #pip install pynrrd, if pynrrd is not already installed
import numpy as np

#scores from nrrdDx
scores = [64539, 63192, 84664, 61977, 68893, 201945, 41470, 66649, 69547, 59223, 123213, 114652, 67135, 39996, 80799, 52717, 41811, 185080, 51173, 132946, 64459, 212707, 96176, 60284, 73844, 47204, 140342, 74520, 70296, 232686]

#central zone scores from nrrdDx
cz_scores = [26323, 28175, 56945, 30785, 34122, 148846, 20414, 33568, 23019, 24493, 66861, 66418, 37318, 20464, 28082, 26245, 19511, 169363, 32595, 70533, 46099, 168267, 70607, 27046, 45861, 20173, 95230, 36517, 41718, 182232]

#ratios from nrrdDx
ratios = [0.408, 0.446, 0.673, 0.497, 0.495, 0.737, 0.492, 0.504, 0.331, 0.414, 0.543, 0.579, 0.556, 0.512, 0.348, 0.498, 0.467, 0.915, 0.637, 0.531, 0.715, 0.791, 0.734, 0.449, 0.621, 0.427, 0.679, 0.49, 0.593, 0.783]

#tuples from nrrdDx
tuples = [[64539, 0.408], [63192, 0.446], [84664, 0.673], [61977, 0.497], [68893, 0.495], [201945, 0.737], [41470, 0.492], [66649, 0.504], [69547, 0.331], [59223, 0.414], [123213, 0.543], [114652, 0.579], [67135, 0.556], [39996, 0.512], [80799, 0.348], [52717, 0.498], [41811, 0.467], [185080, 0.915], [51173, 0.637], [132946, 0.531], [64459, 0.715], [212707, 0.791], [96176, 0.734], [60284, 0.449], [73844, 0.621], [47204, 0.427], [140342, 0.679], [74520, 0.490], [70296, 0.593], [232686, 0.783]]

scores3T = [32253, 52022, 50194, 57416, 32027, 40763, 11929, 33886, 91391, 33661]

cz_scores3T = [26451, 38265, 35187, 44365, 21413, 32050, 7294, 30071, 87558, 17010]

ratios3T = [0.82, 0.736, 0.701, 0.773, 0.669, 0.786, 0.611, 0.887, 0.958, 0.505]
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

def getScore(datatype):
    if datatype == 0:
        return scores
    elif datatype == 1:
        return scores3T
    
def getCZScores(datatype):
    if datatype == 0:
        return cz_scores
    elif datatype == 1:
        return cz_scores3T
    
def getRatios():
    return ratios

def getTuples():
    return tuples

def calculateScore(datatype, start, end):
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

def calculateRatios(datatype):
    ratios = []
    if datatype == 0:
        for i in range (0, len(cz_scores)):
            ratios.append(round(cz_scores[i]/scores[i],3))
    elif datatype == 1:
        for i in range (0, len(cz_scores3T)):
            ratios.append(round(cz_scores3T[i]/scores3T[i],3))
            
    print(ratios)

def createData():
    f= open("data.txt","w+")
    for i in range(len(scores)):
        f.write("[%d, " % scores[i]) 
        f.write("%.3f], " % ratios[i]) 
    f.write("\r\n")
    f.close

print('Number of Total Volume Scores')
print(len(scores))
print('Number of CZ Volume Scores')
print(len(cz_scores))
calculateRatios(1)
print('Number of Score Ratios')
print(len(ratios))
print('Number of Tuples')
print(len(tuples))
#createData()
<<<<<<< HEAD

=======
>>>>>>> cba7314411e42ee4b5b3acd1219400bcde45dfd1
