import os
import nrrd #pip install pynrrd, if pynrrd is not already installed
import numpy as np


scores = [64539, 63192, 84664, 61977, 68893, 201945, 41470, 66649, 69547, 59223, 123213, 114652, 67135, 39996, 80799, 52717, 41811, 185080, 51173, 132946, 64459, 212707, 96176, 60284, 73844, 47204, 140342, 74520, 70296, 232686]

cz_scores = [26323, 28175, 56945, 30785, 34122, 148846, 20414, 33568, 23019, 24493, 66861, 66418, 37318, 20464, 28082, 26245, 19511, 169363, 32595, 70533, 46099, 168267, 70607, 27046, 45861, 20173, 95230, 36517, 41718, 182232]

ratios = [0.408, 0.446, 0.673, 0.497, 0.495, 0.737, 0.492, 0.504, 0.331, 0.414, 0.543, 0.579, 0.556, 0.512, 0.348, 0.498, 0.467, 0.915, 0.637, 0.531, 0.715, 0.791, 0.734, 0.449, 0.621, 0.427, 0.679, 0.49, 0.593, 0.783]

tuples = [[64539, 0.408], [63192, 0.446], [84664, 0.673], [61977, 0.497], [68893, 0.495], [201945, 0.737], [41470, 0.492], [66649, 0.504], [69547, 0.331], [59223, 0.414], [123213, 0.543], [114652, 0.579], [67135, 0.556], [39996, 0.512], [80799, 0.348], [52717, 0.498], [41811, 0.467], [185080, 0.915], [51173, 0.637], [132946, 0.531], [64459, 0.715], [212707, 0.791], [96176, 0.734], [60284, 0.449], [73844, 0.621], [47204, 0.427], [140342, 0.679], [74520, 0.490], [70296, 0.593], [232686, 0.783]]

#load all the .nrrd files into one list
def getData():
    images = [ ]
    for filen in os.listdir('nrrd'):
        if filen.endswith('.nrrd'):
            print(filen)
            data, header = nrrd.read(os.path.join('nrrd/'+ filen))
            images.append(data)
    return images

def getScore():
    return scores

def getCZScores():
    return cz_scores

def getRatios():
    return ratios

def getTuples():
    return tuples

def calculateScore():
    images = getData()
    image_scores = [ ]
    score = 0
    for a in range (0,len(scores)):
        img = images[a]
        for i in range (0, len(img)):
            for j in range (0, len(img)):
                for k in range (0, len(img[0][0])):
                    if img[i][j][k] > 0:
                        score = score + 1
        image_scores.append(score)
        score = 0
    print(image_scores)
    
def calculateCZ(start, end):
    images = getData()
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

def calculateRatios():
    for i in range (0, len(cz_scores)):
        ratios.append(round(cz_scores[i]/scores[i],3))
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
calculateRatios()
print('Number of Score Ratios')
print(len(ratios))
print('Number of Tuples')
print(len(tuples))
#createData()
