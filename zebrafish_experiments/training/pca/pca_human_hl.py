import h5py, os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from training.nb_util import tokenize_genes_substring

datadir = "..\\..\\paperData\\pM10Kb_1KTest"

# Get truth data
testfile = h5py.File(os.path.join(datadir, 'test.h5'), 'r')
valfile = h5py.File(os.path.join(datadir, 'valid.h5'), 'r')
trainfile = h5py.File(os.path.join(datadir, 'train.h5'), 'r')
y = np.concatenate((testfile['detectionFlagInt'][:], valfile['detectionFlagInt'][:], trainfile['detectionFlagInt'][:]), axis=None)
X = np.concatenate((testfile['data'], valfile['data'], trainfile['data']))
print("y.shape", y.shape)

# exclude genes if expression is unknown i.e., 2
excluded_indices = [i for i in range(len(y)) if y[i] == 2]
y = [label for (idx, label) in enumerate(y) if idx not in excluded_indices]
X = [np.array(data) for (idx, data) in enumerate(X) if idx not in excluded_indices]

#pca = TruncatedSVD(n_components=2, n_iter=5, random_state=0)
pca = KernelPCA(n_components=2, kernel="poly")
components = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(components[:500, 0], components[:500, 1], c=y[:500], cmap='bwr', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Projection Colored by Labels')
plt.colorbar(scatter, label='Label')
plt.grid(True)
plt.show()