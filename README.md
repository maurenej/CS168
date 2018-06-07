## Automated Prostate Tumor Classification from MRI Segmentations

Our code for CS 168: Medical Imaging taught by Dr. Scalzo. Our project aimed to automatically classify numerous cases of prostate cancer into few clusters based on the ratio of the central gland and the peripheral zone. The 3D segmentation data was retrieved from https://wiki.cancerimagingarchive.net/display/Public/PROSTATE-DIAGNOSIS 

### Approach

The main points that we are concerned with for each 3D image are the total volume of the prostate and ratio of central gland to total volume. From that, we aim to classify them into numerical clusters. Our system will then learn to correlate each [volume, ratio] pair with the clusters and then be able to predict the cluster numbers for further test data. 

### File Structure
```shell
CS168/
---nrrd3T/        (.nrrd segmentation files 1)
---nrrdDx/        (.nrrd segmentation files 2)
---cluster.py     (code to cluster data)
---data.py        (code to generate and retrieve data from .nrrd files)
---data3T.txt     (holds tuple data from nrrd3T)
---dataDx.txt     (holds tuple data from nrrdDx)
```
### Installation

To be able to run the Python scripts, you will need to have a version of Python 3 (to generate plots) and install several libraries. 
These include matplotlib, scikit, numpy, pynrrd
You can do so with the following:

` $ [sudo] pip3 install matplotlib scikit numpy pynrrd `

Once you have all of them installed, you are good to go!

### Usage

You can see how each of our data points are clustered and the performances of the learning algorithm by running `cluster.py`. This script uses sklearn as a basis to train and test the segmentation data, which are produced from `data.py`.

The algorithm that we use is KNN (K-Nearest-Neightbors) which is a form of supervised learning to relate X to Y, in our case, [total volume, ratio] to cluster number. More info can be found here: http://scikit-learn.org/stable/modules/neighbors.html

```bash
$ python3 cluster.py <n or y> <n or y> <n or y>
```

The first argument is the silent option. `n` will mute status info from the console while `y` will display to the console.  
The second argument is the plot option. `n` will not display plots. `y` will display the plots that show color coded cluster groups of each data point.  
The third argument is the default train/test data option. `n` will generate a new combination of test and training data everytime it is run. `y` will use the default test and training data.
