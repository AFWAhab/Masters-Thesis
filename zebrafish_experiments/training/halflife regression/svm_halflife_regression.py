import h5py
import numpy as np
import os
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

datadir = "..\\..\\paperData\\pM10Kb_1KTest"
testfile = h5py.File(os.path.join(datadir, 'test.h5'), 'r')
valfile = h5py.File(os.path.join(datadir, 'valid.h5'), 'r')
trainfile = h5py.File(os.path.join(datadir, 'train.h5'), 'r')
X = np.concatenate((testfile['data'], valfile['data'], trainfile['data']))

# Get truth data
y = np.concatenate((testfile['label'][:], valfile['label'][:], trainfile['label'][:]), axis=None)
print("y.shape", y.shape)

# rbf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = svm.SVR(kernel="rbf", C=0.1, gamma='scale', epsilon=0.1)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
print("FOR SVR w. rbf")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# poly deg 3
model = svm.SVR(kernel="rbf", C=0.1, gamma='scale', epsilon=0.1)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
print("FOR SVR w. poly deg 3")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# poly deg 5
model = svm.SVR(kernel="rbf", C=0.1, gamma='scale', epsilon=0.1, degree=5)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
print("FOR SVR w. poly deg 5")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# lin reg
model = LinearRegression()
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
print("FOR lin reg")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# ridge reg
model = Ridge()
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
print("FOR ridge reg")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# SGDRegressor
model = SGDRegressor()
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
print("FOR SGDRegressor")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

