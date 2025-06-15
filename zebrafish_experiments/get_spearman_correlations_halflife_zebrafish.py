import pandas as pd
from scipy.stats import spearmanr
import statistics
import h5py
import os
import numpy as np


file = "training/zebrafish.hdf5"
X = h5py.File(file, 'r')
data = X['halflifeData']
#label = X['label']
embryo = X['embryoLabels']

fiveutrlen = data[:,0]
threeutrlen = data[:,1]
orflen = data[:,2]
fiveutrgc = data[:,3]
threeutrgc = data[:,4]
orfgc = data[:,5]
intronlen = data[:,6]
exonjunctiondensity = data[:,7]

correlations = spearmanr(fiveutrlen, embryo)
statistic = correlations.statistic
print("5utrlen", statistic)

correlations = spearmanr(threeutrlen, embryo)
statistic = correlations.statistic
print("3utrlen", statistic)

correlations = spearmanr(orflen, embryo)
statistic = correlations.statistic
print("orflen", statistic)

correlations = spearmanr(fiveutrgc, embryo)
print("fiveutrgc", correlations.statistic)

correlations = spearmanr(threeutrgc, embryo)
print("threeutrgc", correlations.statistic)

correlations = spearmanr(orfgc, embryo)
print("orfgc", correlations.statistic)

correlations = spearmanr(intronlen, embryo)
print("intronlen", correlations.statistic)

correlations = spearmanr(exonjunctiondensity, embryo)
print("exonjunctiondensity", correlations.statistic)