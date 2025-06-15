import h5py, os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from Bio import SeqIO
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from training.naive_bayes.naive_bayes import tokenize_genes_substring

datadir = "..\\..\\paperData\\pM10Kb_1KTest"

print("Starting tokenization")
#tokenized_sequences_test = tokenize_genes("../../paperData/pM10Kb_1KTest/human_promoter_sequences_test.fasta")
#tokenized_sequences_val = tokenize_genes("../../paperData/pM10Kb_1KTest/human_promoter_sequences_valid.fasta")
#tokenized_sequences_training = tokenize_genes("../../paperData/pM10Kb_1KTest/human_promoter_sequences_train.fasta")
tokenized_sequences_test = tokenize_genes_substring("../../paperData/pM10Kb_1KTest/human_promoter_sequences_test.fasta", 3000, 13501, 10)
tokenized_sequences_val = tokenize_genes_substring("../../paperData/pM10Kb_1KTest/human_promoter_sequences_valid.fasta", 3000, 13501, 10)
tokenized_sequences_training = tokenize_genes_substring("../../paperData/pM10Kb_1KTest/human_promoter_sequences_train.fasta", 3000, 13501, 10)
tokenized_sequences = np.concatenate((tokenized_sequences_test, tokenized_sequences_val, tokenized_sequences_training), axis=None)

# Get truth data
testfile = h5py.File(os.path.join(datadir, 'test.h5'), 'r')
valfile = h5py.File(os.path.join(datadir, 'valid.h5'), 'r')
trainfile = h5py.File(os.path.join(datadir, 'train.h5'), 'r')
y = np.concatenate((testfile['detectionFlagInt'][:], valfile['detectionFlagInt'][:], trainfile['detectionFlagInt'][:]), axis=None)
print("y.shape", y.shape)

# exclude genes if expression is unknown i.e., 2
excluded_indices = [i for i in range(len(y)) if y[i] == 2]
y = [label for (idx, label) in enumerate(y) if idx not in excluded_indices]
tokenized_sequences = [seq for (idx, seq) in enumerate(tokenized_sequences) if idx not in excluded_indices]

# Use CountVectorizer to count k-mers
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_sequences)

# You now have a sparse matrix of k-mer counts
print("X.shape", X.shape)  # (num_sequences, num_unique_kmers)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

test_number_of_present = len([label for label in y_test if label == 1])
test_number_of_absent = len([label for label in y_test if label == 0])
print("Number of test genes:", len(y_test))
print("y stats: Number of present:", test_number_of_present," and number of absent:", test_number_of_absent)
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