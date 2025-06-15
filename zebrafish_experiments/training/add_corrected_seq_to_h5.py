import h5py
import pandas as pd
import numpy as np
from scipy import stats
from Bio import SeqIO

# Define one-hot encoding function
def one_hot_encode(seq, seq_length=20000):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}  # 'N' represents unknown bases
    encoded = np.zeros((seq_length, 4), dtype=bool)
    for i, nucleotide in enumerate(seq[:seq_length]):  # Truncate if too long
        encoded[i] = mapping.get(nucleotide, [0, 0, 0, 0])
    return encoded

promoter_fasta_file_path = "zebrafish_promoter_sequences_CORRECTED.fasta"
promoters = {}
for record in SeqIO.parse(promoter_fasta_file_path, "fasta"):
    gene_name = record.id.split("|")[0]
    promoters[gene_name] = one_hot_encode(str(record.seq))

promoters_gene_names = promoters.keys()

f_train = h5py.File("zebrafish_training/zebrafish_train.hdf5", "a")
f_val = h5py.File("zebrafish_training/zebrafish_val.hdf5", "a")
f_test = h5py.File("zebrafish_training/zebrafish_test.hdf5", "a")

gene_name_dataset_train = f_train["geneName"]
gene_name_dataset_val = f_val["geneName"]
gene_name_dataset_test = f_test["geneName"]
promoter_array_train = np.zeros((len(gene_name_dataset_train), 20000, 4), dtype=bool)
promoter_array_val = np.zeros((len(gene_name_dataset_val), 20000, 4), dtype=bool)
promoter_array_test = np.zeros((len(gene_name_dataset_test), 20000, 4), dtype=bool)

print("Test dataset")
for i, gene in enumerate(gene_name_dataset_test):
    if i % 100 == 0:
        print(i)

    gene_string = str(gene)[2:-1]
    if gene_string in promoters_gene_names:
        seq = promoters[gene_string]
        promoter_array_test[i] = seq
    else:
        promoter_array_test[i] = []

f_test.create_dataset("promoterCorrected", data=promoter_array_test)

print("Val dataset")
for i, gene in enumerate(gene_name_dataset_val):
    if i % 100 == 0:
        print(i)

    gene_string = str(gene)[2:-1]
    if gene_string in promoters_gene_names:
        seq = promoters[gene_string]
        promoter_array_val[i] = seq
    else:
        promoter_array_val[i] = []

f_val.create_dataset("promoterCorrected", data=promoter_array_val)

print("Train dataset")
for i, gene in enumerate(gene_name_dataset_train):
    if i % 100 == 0:
        print(i)

    gene_string = str(gene)[2:-1]
    if gene_string in promoters_gene_names:
        seq = promoters[gene_string]
        promoter_array_train[i] = seq
    else:
        promoter_array_train[i] = []

f_train.create_dataset("promoterCorrected", data=promoter_array_train)