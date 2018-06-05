
import nrrd #pip install pynrrd, if pynrrd is not already installed
import nibabel as nib #pip install nibabel, if nibabel is not already installed
import numpy as np


_nrrd = nrrd.read('nrrd/ProstateDx-01-0021.nrrd')
data = _nrrd[0]
header = _nrrd[1]

#save nifti
img = nib.Nifti1Image(data, np.eye(4))
nib.save(img,'one.nii.gz')
