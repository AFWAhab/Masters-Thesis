import h5py
import numpy as np

def get_nucleotide(array):
    if np.array_equiv(array, np.array([True, False, False, False])):
        return 'A'
    if np.array_equiv(array, np.array([False, True, False, False])):
        return 'C'
    if np.array_equiv(array, np.array([False, False, True, False])):
        return 'G'
    if np.array_equiv(array, np.array([False, False, False, True])):
        return 'T'
    return 'N'

def one_hot_to_string(matrix):
    promoter_string = ''
    len_of_seq = matrix.shape[0]
    for j in range(len_of_seq):
        promoter_string += get_nucleotide(matrix[j])

    return promoter_string

# Open the file in read mode
#file_path = "pM10Kb_1KTest/train.h5"
file_path = "..\\training\\zebrafish.hdf5"
with h5py.File(file_path, "r") as f:
    promoter_dataset = f["promoter"]
    gene_name_dataset = f["geneName"]
    #output_fasta_test = "pM10Kb_1KTest/human_promoter_sequences_train.fasta"
    output_fasta_test = "zebrafish_test_promoters_CORRECTED.fasta"
    with open(output_fasta_test, "w") as fasta_out:
        # n = gene_name_dataset.shape[0]
        n = 1000 # experiment on 1000 zebrafish genes
        print("length:", n)
        counter = 0
        for i in range(n):
            gene_id = gene_name_dataset[i]
            promoter_matrix = promoter_dataset[i]
            seq = one_hot_to_string(promoter_matrix)
            fasta_out.write(f">{gene_id}\n{seq}\n")

            if counter % 100 == 0:
                print(counter)
            counter += 1
