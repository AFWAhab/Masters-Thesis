import h5py
import numpy as np
import os
import pandas as pd
from scipy import stats

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA

from training.nb_util import tokenize_genes_substring

datadir = "..\\..\\paperData\\pM10Kb_1KTest"



# Get truth data
testfile = h5py.File(os.path.join(datadir, 'test.h5'), 'r')
valfile = h5py.File(os.path.join(datadir, 'valid.h5'), 'r')
trainfile = h5py.File(os.path.join(datadir, 'train.h5'), 'r')
y = np.concatenate((testfile['label'][:], valfile['label'][:], trainfile['label'][:]), axis=None)
print("y.shape", y.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = svm.SVR(kernel="rbf", C=100, gamma='scale', epsilon=0.1)
model.fit(X_train[:4000], y_train[:4000])

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR CountVectorizer")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)
df = pd.DataFrame(np.column_stack((test_pred, y_test)), columns=['Pred', 'Actual'])

print('Rows & Cols:', df.shape)
#df.to_csv('svm_human_count_4000_6mers.txt', index=False, header=True, sep='\t')

# REPEATED FOR TFIDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_sequences)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = svm.SVR(kernel="rbf", C=100, gamma='scale', epsilon=0.1)
model.fit(X_train[:4000], y_train[:4000])

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR TfidfVectorizer")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)
df = pd.DataFrame(np.column_stack((test_pred, y_test)), columns=['Pred', 'Actual'])

print('Rows & Cols:', df.shape)
#df.to_csv('svm_human_tfidf_4000_6mers.txt', index=False, header=True, sep='\t')