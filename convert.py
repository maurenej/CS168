import matplotlib
matplotlib.use('Agg')
import os

import nrrd #pip install pynrrd, if pynrrd is not already installed
import nibabel as nib #pip install nibabel, if nibabel is not already installed
import numpy as np
import matplotlib.pyplot as plt

#load nrrd
data, header = nrrd.read('nrrd/ProstateDx-01-0005.nrrd')
#save nifti
img = nib.load('someones_epi.nii.gz')
#img = nib.Nifti1Image(data, np.eye(4))
img_data = img.get_data()
print('Data Shape:')
print(img_data.shape)

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


slice_0 = img_data[26, :, :]
slice_1 = img_data[:, 30, :]
slice_2 = img_data[:, :, 16]
np.savetxt('out1.txt', slice_2, fmt='%.4e', delimiter=' ')
show_slices([slice_0, slice_1, slice_2])
