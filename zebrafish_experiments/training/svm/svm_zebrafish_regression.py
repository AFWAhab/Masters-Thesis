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

datadir = "..\\zebrafish_training"

print("Starting tokenization")
#tokenized_sequences_test = tokenize_genes("../zebrafish_promoter_sequences_test.fasta")
#tokenized_sequences_val = tokenize_genes("../zebrafish_promoter_sequences_validation.fasta")
#tokenized_sequences_training = tokenize_genes("../zebrafish_promoter_sequences_training.fasta")
tokenized_sequences_test = tokenize_genes_substring("../zebrafish_promoter_sequences_test.fasta", 8500, 11501, 10)
tokenized_sequences_val = tokenize_genes_substring("../zebrafish_promoter_sequences_validation.fasta", 8500, 11501, 10)
tokenized_sequences_training = tokenize_genes_substring("../zebrafish_promoter_sequences_training.fasta", 8500, 11501, 10)
tokenized_sequences = np.concatenate((tokenized_sequences_test, tokenized_sequences_val, tokenized_sequences_training), axis=None)

# Use CountVectorizer to count k-mers
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_sequences)

# You now have a sparse matrix of k-mer counts
print("X.shape", X.shape)  # (num_sequences, num_unique_kmers)

# Get truth data
testfile = h5py.File(os.path.join(datadir, 'zebrafish_test.hdf5'), 'r')
valfile = h5py.File(os.path.join(datadir, 'zebrafish_val.hdf5'), 'r')
trainfile = h5py.File(os.path.join(datadir, 'zebrafish_train.hdf5'), 'r')
y = np.concatenate((testfile['label'][:], valfile['label'][:], trainfile['label'][:]), axis=None)
y = stats.zscore(y)
print("y.shape", y.shape)

#names = np.concatenate((testfile['geneName'][:], valfile['geneName'][:], trainfile['geneName'][:]), axis=None)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit Naive Bayes
model = svm.LinearSVR(random_state=0, tol=1e-5)
model.fit(X_train[:2000], y_train[:2000])

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR CountVectorizer")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)
df = pd.DataFrame(np.column_stack((test_pred, y_test)), columns=['Pred', 'Actual'])

print('Rows & Cols:', df.shape)
df.to_csv('svm_zebrafish_count_2000_10mers.txt', index=False, header=True, sep='\t')

# REPEATED FOR TFIDF
# Use CountVectorizer to count k-mers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_sequences)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit Naive Bayes
model = svm.LinearSVR(random_state=0, tol=1e-5)
model.fit(X_train[:500], y_train[:500])

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR TfidfVectorizer")
slope, intercept, r_value, p_value, std_err = stats.linregress(test_pred, y_test)
print('Test R^2 = %.3f' % r_value ** 2)
df = pd.DataFrame(np.column_stack((test_pred, y_test)), columns=['Pred', 'Actual'])

print('Rows & Cols:', df.shape)
df.to_csv('svm_zebrafish_tfidf_2000_10mers.txt', index=False, header=True, sep='\t')