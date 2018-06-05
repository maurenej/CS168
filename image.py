import matplotlib
matplotlib.use('Agg')
import os
import nrrd #pip install pynrrd, if pynrrd is not already installed
import numpy as np
import matplotlib.pyplot as plt

#load all the .nrrd files into one list
images = [ ]
for filen in os.listdir('nrrd'):
    if filen.endswith('.nrrd'):
        print(filen)
        data, header = nrrd.read(os.path.join('nrrd/'+ filen))
        images.append(data)
    
#load nrrd
#data, header = nrrd.read('nrrd/ProstateDx-01-0005.nrrd')

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

#testing images[14]
img_14 = images[14]
slice_0 = img_14[200, :, :]
slice_1 = img_14[:, 30, :]
slice_2 = img_14[:, :, 10]
np.savetxt('OutputFiles/out15.txt', slice_0, fmt='%d', delimiter=' ')
show_slices([slice_0, slice_1, slice_2])
