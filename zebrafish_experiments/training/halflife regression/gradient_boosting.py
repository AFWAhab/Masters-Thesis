import h5py
import numpy as np
import os
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

datadir = "..\\..\\paperData\\pM10Kb_1KTest"
testfile = h5py.File(os.path.join(datadir, 'test.h5'), 'r')
valfile = h5py.File(os.path.join(datadir, 'valid.h5'), 'r')
trainfile = h5py.File(os.path.join(datadir, 'train.h5'), 'r')
X = np.concatenate((testfile['data'], valfile['data'], trainfile['data']))

# Get truth data
y = np.concatenate((testfile['label'][:], valfile['label'][:], trainfile['label'][:]), axis=None)
print("y.shape", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Gradient boosting regressor
clf = HistGradientBoostingRegressor(random_state=0, l2_regularization=1, max_features=0.6)
clf.fit(X_train, y_train)

test_pred = clf.predict(X_test)
print("FOR Gradient boosting regressor")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

