import matplotlib
matplotlib.use('Agg')
import os
import nrrd #pip install pynrrd, if pynrrd is not already installed
import numpy as np
import matplotlib.pyplot as plt
from data import getData
    
#load nrrd
#data, header = nrrd.read('nrrd/ProstateDx-01-0005.nrrd')

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

#testing images[14]
imglist = getData()
img_14 = images[14]
slice_0 = img_14[200, :, :]
slice_1 = img_14[:, 30, :]
slice_2 = img_14[:, :, 10]
#np.savetxt('OutputFiles/out15.txt', slice_0, fmt='%d', delimiter=' ')
#show_slices([slice_0, slice_1, slice_2])
print('Imported list length:')
print(len(imglist))


