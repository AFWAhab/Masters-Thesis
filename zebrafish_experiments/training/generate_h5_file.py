import h5py
import numpy as np
import pandas as pd
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

# Load labels from CSV
df_estradiol = pd.read_csv("../data/Estradiol_CountMatrix_RPKM.tsv", sep="\t")
df_estradiol = df_estradiol.set_index('gene_id')
#df = pd.read_csv("../data/all_values_pig.txt")
df = pd.read_csv("../data/all_values.txt")
df = df.set_index("Gene ID")  # Ensure we can match gene names easily
#df_detection_flag = pd.read_csv("../data/all_values_detection_flag.txt")
#df_detection_flag = df_detection_flag.set_index("Gene ID")

df_tss = pd.read_csv("..\\data\\zebrafish_tss_CORRECTED_w_strands.csv", sep="\t")

# Load promoter sequences and gene names from FASTA
promoter_fasta_file_path = "zebrafish_promoter_sequences_CORRECTED.fasta"
#promoter_fasta_file_path = "10kpigs.fa"
promoters = {}
for record in SeqIO.parse(promoter_fasta_file_path, "fasta"):
    gene_name = record.id.split("|")[0]
    gene_names = df_tss[df_tss["Gene ID"] == gene_name]
    if gene_names.empty:
        continue

    strand = gene_names.iloc[0]["strand"]
    if strand == "-":
        promoters[gene_name] = one_hot_encode(str(record.seq.complement())) # the negative have already been reverse
    else:
        promoters[gene_name] = one_hot_encode(str(record.seq))

common_genes = sorted(set(promoters.keys()) & set(df.index) & set(df_estradiol.index))  # Ensure same order

# Create arrays to store data
num_samples = len(common_genes)
promoter_array = np.zeros((num_samples, 20000, 4), dtype=bool)
gene_names_array = np.array(common_genes, dtype="S18")  # Convert to bytes
labels_array = np.zeros(num_samples, dtype="float64")
halflife_data_array = np.zeros((num_samples, 8), dtype="float64") # for now, we ignore this
estradiol_values = np.zeros(num_samples, dtype="float64")
#detection_flags_int = np.empty([num_samples], dtype="uint8")


# Fill arrays with data
for i, gene in enumerate(common_genes):
    if i % 100 == 0:
        print(i)

    promoter_array[i] = promoters[gene]
    labels_array[i] = df.loc[gene, "Median log RPKM"]  # Retrieve label from CSV
    estradiol_val = df_estradiol.loc[gene, "median"]
    estradiol_values[i] = estradiol_val
    #majority_detection_flag = df_detection_flag.loc[gene, "Majority detection flag"]
    #detection_flags_int[i] = 1 if majority_detection_flag == "present" else 0

#f = h5py.File("zebrafish_train.hdf5", "w")
#f = h5py.File("zebrafish_val.hdf5", "w")
f = h5py.File("zebrafish.hdf5", "w")
#f = h5py.File("pig.hdf5", "w")
f.create_dataset("data", data=halflife_data_array)
f.create_dataset("geneName", data=gene_names_array)
f.create_dataset("label", data=labels_array)
f.create_dataset("promoter", data=promoter_array)
f.create_dataset("estradiolLabels", data=estradiol_values)
#f.create_dataset("detectionFlagInt", data=detection_flags_int)