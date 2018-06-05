import SimpleITK as sitk
import nrrd
import matplotlib.pyplot as plt
import numpy as np
# some sample numpy data
filename1 = 'nrrd/ProstateDx-01-0001.nrrd'
filename5 = 'nrrd/ProstateDx-01-0005.nrrd'
filename11= 'nrrd/ProstateDx-01-0011.nrrd'
filename13= 'ProstateDx-01-0013.nrrd'
filename14= 'ProstateDx-01-0014.nrrd'
filename19= 'ProstateDx-01-0019.nrrd'
filename21= 'ProstateDx-01-0021.nrrd'
filename23= 'ProstateDx-01-0023.nrrd'
filename28= 'ProstateDx-01-0028.nrrd'




# read the data back from file
readdata, options = nrrd.read(filename11)
#plt.imshow(readdata[100], interpolation='nearest')
#plt.show()
print('Total shape:')
print(readdata.shape)
#print(options)

slice10 =readdata[:,:,10]
print('Slice10 Shape')
print(slice10.shape)

np.savetxt('out1.txt', slice10, fmt='%.4f',delimiter=' ')
#f = open("dataout.txt","w+")
#f.write(readdata[100])
#f.close()
#print(slice10)
