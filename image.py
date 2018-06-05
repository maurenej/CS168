import matplotlib
matplotlib.use('Agg')

import nrrd #pip install pynrrd, if pynrrd is not already installed
import numpy as np
import matplotlib.pyplot as plt

#load nrrd
data, header = nrrd.read('nrrd/ProstateDx-01-0005.nrrd')
#print the data dimensions
print('Data Shape:')
print(img_data.shape)

#print the data header
print(header)
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = data[200, :, :]
slice_1 = img_data[:, 30, :]
slice_2 = img_data[:, :, 10]
np.savetxt('out2.txt', slice_0, fmt='%d', delimiter=' ')
show_slices([slice_0, slice_1, slice_2])
