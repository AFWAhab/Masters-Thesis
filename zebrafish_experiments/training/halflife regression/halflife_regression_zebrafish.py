import h5py
import numpy as np
import os
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

datadir = "..\\"
file = h5py.File(os.path.join(datadir, 'zebrafish.hdf5'), 'r')
X = np.array(file["halflifeData"])

# Get truth data
y = np.array(file['label'][:])
print("y.shape", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Decision tree
clf = tree.DecisionTreeRegressor(random_state=0)
clf.fit(X_train, y_train)

test_pred = clf.predict(X_test)
print("FOR decision tree ")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# Gradient boosting regressor
clf = HistGradientBoostingRegressor(random_state=0)
clf.fit(X_train, y_train)

test_pred = clf.predict(X_test)
print("FOR Gradient boosting regressor")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# Random forest
clf = RandomForestRegressor(random_state=0)
clf.fit(X_train, y_train)

test_pred = clf.predict(X_test)
print("FOR Random forest")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)