import h5py
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from training.nb_util import tokenize_genes_substring

datadir = "..\\zebrafish_training"

print("Starting tokenization")
#tokenized_sequences_test = tokenize_genes("../zebrafish_promoter_sequences_test.fasta")
#tokenized_sequences_val = tokenize_genes("../zebrafish_promoter_sequences_validation.fasta")
#tokenized_sequences_training = tokenize_genes("../zebrafish_promoter_sequences_training.fasta")
tokenized_sequences_test = tokenize_genes_substring("../zebrafish_promoter_sequences_test.fasta", 8500, 11501, 12)
tokenized_sequences_val = tokenize_genes_substring("../zebrafish_promoter_sequences_validation.fasta", 8500, 11501, 12)
tokenized_sequences_training = tokenize_genes_substring("../zebrafish_promoter_sequences_training.fasta", 8500, 11501, 12)
tokenized_sequences = np.concatenate((tokenized_sequences_test, tokenized_sequences_val, tokenized_sequences_training), axis=None)

# Use CountVectorizer to count k-mers
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_sequences[:500])

# You now have a sparse matrix of k-mer counts
print("X.shape", X.shape)  # (num_sequences, num_unique_kmers)

# Get truth data
testfile = h5py.File(os.path.join(datadir, 'zebrafish_test.hdf5'), 'r')
valfile = h5py.File(os.path.join(datadir, 'zebrafish_val.hdf5'), 'r')
trainfile = h5py.File(os.path.join(datadir, 'zebrafish_train.hdf5'), 'r')
y = np.concatenate((testfile['detectionFlagInt'][:], valfile['detectionFlagInt'][:], trainfile['detectionFlagInt'][:]), axis=None)
print("y.shape", y.shape)
pca = PCA(n_components=2)
components = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(components[:, 0], components[:, 1], c=y[:500], cmap='bwr', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Projection Colored by Labels')
plt.colorbar(scatter, label='Label')
plt.grid(True)
plt.show()