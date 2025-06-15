import h5py, os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from training.nb_util import tokenize_genes_substring

datadir = "..\\..\\paperData\\pM10Kb_1KTest"

print("Starting tokenization")
#tokenized_sequences_test = tokenize_genes("../../paperData/pM10Kb_1KTest/human_promoter_sequences_test.fasta")
#tokenized_sequences_val = tokenize_genes("../../paperData/pM10Kb_1KTest/human_promoter_sequences_valid.fasta")
#tokenized_sequences_training = tokenize_genes("../../paperData/pM10Kb_1KTest/human_promoter_sequences_train.fasta")
tokenized_sequences_test = tokenize_genes_substring("../../paperData/pM10Kb_1KTest/human_promoter_sequences_test.fasta", 8500, 11501)
tokenized_sequences_val = tokenize_genes_substring("../../paperData/pM10Kb_1KTest/human_promoter_sequences_valid.fasta", 8500, 11501)
tokenized_sequences_training = tokenize_genes_substring("../../paperData/pM10Kb_1KTest/human_promoter_sequences_train.fasta", 8500, 11501)
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
X = vectorizer.fit_transform(tokenized_sequences[:500])

# You now have a sparse matrix of k-mer counts
print("X.shape", X.shape)  # (num_sequences, num_unique_kmers)
pca = TruncatedSVD(n_components=2, n_iter=5, random_state=0)
components = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(components[:, 0], components[:, 1], c=y[:500], cmap='bwr', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Projection Colored by Labels')
plt.colorbar(scatter, label='Label')
plt.grid(True)
plt.show()