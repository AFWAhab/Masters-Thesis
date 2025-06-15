import h5py
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from training.nb_util import tokenize_genes, tokenize_genes_substring
import math
from Bio import SeqIO
import pandas as pd

print("Starting tokenization")
tokenized_sequences = tokenize_genes_substring("../pigs_fasta_corrected.fa", 5500, 8501, 10)

ids = []
counter = 0
for record in SeqIO.parse("../pigs_fasta_corrected.fa", "fasta"):
    gene_name = record.id.split("|")[0]
    ids.append(gene_name)

dict_tokens = {'tokens': tokenized_sequences, 'ids': ids}
df_tokens = pd.DataFrame(data = dict_tokens)


# Get truth data
h5_file = h5py.File("../pig.hdf5", 'r')
detection_flag_int = h5_file['detectionFlagIntSScrofa11']
h5_genes = h5_file['geneName']
h5_genes = [str(gene)[2:-1] for gene in h5_genes]
dict_truth = {'detection_flag_int': detection_flag_int, 'ids': h5_genes}
dict_truth_filtered = {'detection_flag_int': [], 'ids': []}
for idx, flag in enumerate(dict_truth['detection_flag_int']):
    if flag != 0 and flag != 1:
        #print("sus", idx)
        continue
    else:
        dict_truth_filtered['detection_flag_int'].append(flag)
        dict_truth_filtered['ids'].append(dict_truth['ids'][idx])

df_truth = pd.DataFrame(data = dict_truth_filtered)

df_join = df_tokens.set_index('ids').join(df_truth.set_index('ids'), how='inner')
tokenized_sequences = df_join['tokens'].tolist()
y = df_join['detection_flag_int'].tolist()

# Use CountVectorizer to count k-mers
vectorizer = CountVectorizer()
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