import h5py
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from training.nb_util import tokenize_genes_substring

datadir = "..\\"

print("Starting tokenization")
tokenized_sequences = tokenize_genes_substring("../zebrafish_promoter_sequences_CORRECTED.fasta", 8500, 11501, 12)

# Use CountVectorizer to count k-mers
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_sequences)

# You now have a sparse matrix of k-mer counts
print("X.shape", X.shape)  # (num_sequences, num_unique_kmers)

# Get truth data
#testfile = h5py.File(os.path.join(datadir, 'zebrafish_test.hdf5'), 'r')
#valfile = h5py.File(os.path.join(datadir, 'zebrafish_val.hdf5'), 'r')
#trainfile = h5py.File(os.path.join(datadir, 'zebrafish_train.hdf5'), 'r')
#y = np.concatenate((testfile['detectionFlagInt'][:], valfile['detectionFlagInt'][:], trainfile['detectionFlagInt'][:]), axis=None)
file = h5py.File(os.path.join(datadir, 'zebrafish.hdf5'), 'r')
y = np.array(file['detectionFlagInt'][:])
print("y.shape", y.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

test_number_of_present = len([label for label in y_test if label == 1])
test_number_of_absent = len([label for label in y_test if label == 0])
print("y stats: Number of present (positives):", test_number_of_present," and number of absent (negatives):", test_number_of_absent)
print("Percentage of present:", test_number_of_present / len(y_test) * 100)

# Fit Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR CountVectorizer")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred))
print("Recall:", recall_score(y_test, test_pred))
print("F1-score:", f1_score(y_test, test_pred))
tf = sum([1 for idx, label in enumerate(test_pred) if test_pred[idx] == y_test[idx] and y_test[idx] == 1])
tn = sum([1 for idx, label in enumerate(test_pred) if test_pred[idx] == y_test[idx] and y_test[idx] == 0])
fp = sum([1 for idx, label in enumerate(test_pred) if test_pred[idx] != y_test[idx] and y_test[idx] == 0])
fn = sum([1 for idx, label in enumerate(test_pred) if test_pred[idx] != y_test[idx] and y_test[idx] == 1])
print("Number of true positives:", tf)
print("Number of true negatives:", tn)
print("Number of false positives:", fp)
print("Number of false negatives:", fn)
print("Negative precision (Out of all the negative predictions we made, how many were actually negative?):", tn/(tn+fn))
print("Negative recall (Out of all the sequences that should be predicted as unexpressed, how many did we correctly predict as unexpressed?):", tn/(tn+fp))

# REPEATED FOR TFIDF
# Use CountVectorizer to count k-mers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_sequences)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
test_pred = model.predict(X_test)
print("FOR TfidfVectorizer")
print("Accuracy (tf/(tf+fP)):", accuracy_score(y_test, test_pred))
print("Precision (tf/(tf+nP)):", precision_score(y_test, test_pred))
print("Recall:", recall_score(y_test, test_pred))
print("F1-score:", f1_score(y_test, test_pred))
tf = sum([1 for idx, label in enumerate(test_pred) if test_pred[idx] == y_test[idx] and y_test[idx] == 1])
tn = sum([1 for idx, label in enumerate(test_pred) if test_pred[idx] == y_test[idx] and y_test[idx] == 0])
fp = sum([1 for idx, label in enumerate(test_pred) if test_pred[idx] != y_test[idx] and y_test[idx] == 0])
fn = sum([1 for idx, label in enumerate(test_pred) if test_pred[idx] != y_test[idx] and y_test[idx] == 1])
print("Number of true positives:", tf)
print("Number of true negatives:", tn)
print("Number of false positives:", fp)
print("Number of false negatives:", fn)
print("Negative precision (Out of all the negative predictions we made, how many were actually negative?):", tn/(tn+fn))
print("Negative recall (Out of all the sequences that should be predicted as unexpressed, how many did we correctly predict as unexpressed?):", tn/(tn+fp))