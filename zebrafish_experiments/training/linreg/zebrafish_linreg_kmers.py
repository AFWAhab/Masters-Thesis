import os

import h5py
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from training.nb_util import tokenize_genes_substring

datadir = "..\\"

print("Starting tokenization")
tokenized_sequences = tokenize_genes_substring("../zebrafish_promoter_sequences_CORRECTED.fasta", 0, 20000, 5)

# Use CountVectorizer to count k-mers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tokenized_sequences)

# You now have a sparse matrix of k-mer counts
print("X.shape", X.shape)  # (num_sequences, num_unique_kmers)

# Get truth data
file = h5py.File(os.path.join(datadir, 'zebrafish.hdf5'), 'r')
y = np.array(file['label'][:])
one_hot_stuff = np.array(file["promoter"])
print("y.shape", y.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit linreg
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR linreg")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# Fit ridge
model = Ridge()
model.fit(X_train, y_train)

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR ridge")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# Fit lasso
model = Lasso()
model.fit(X_train, y_train)

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR lasso")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)

# Fit ElasticNet
model = ElasticNet()
model.fit(X_train, y_train)

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR ElasticNet")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)
