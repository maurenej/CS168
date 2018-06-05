import os
import nrrd #pip install pynrrd, if pynrrd is not already installed
import numpy as np


scores = [64539, 63192, 84664, 61977, 68893, 201945, 41470, 66649, 69547, 59223, 123213, 114652, 67135, 39996, 80799, 52717, 41811, 185080, 51173, 132946, 64459, 212707, 96176, 60284, 73844, 47204, 140342, 74520, 70296, 232686]

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

