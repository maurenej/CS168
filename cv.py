import numpy as np
import matplotlib.pyplot as plt
import data
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn import linear_model

train_data = data.getTuples()