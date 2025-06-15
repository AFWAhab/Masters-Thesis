import pandas as pd
from scipy.stats import spearmanr
import statistics
import h5py
import os
import numpy as np


datadir = "paperData\\pM10Kb_1KTest"
testfile = h5py.File(os.path.join(datadir, 'test.h5'), 'r')
valfile = h5py.File(os.path.join(datadir, 'valid.h5'), 'r')
trainfile = h5py.File(os.path.join(datadir, 'train.h5'), 'r')
X = np.concatenate((testfile['data'], valfile['data'], trainfile['data']))

correlations = spearmanr(X)
statistic = correlations.statistic
pvalue = correlations.pvalue

statistic_list = []
pvalue_list = []
for i in range(0, statistic.shape[1]):
    for j in range(0, statistic.shape[1]):
        if i == j:
            continue

        statistic_list.append(statistic[i, j])
        pvalue_list.append(pvalue[i, j])

print(statistic_list)
print("mean of Spearman correlations: ", statistics.mean(statistic_list))