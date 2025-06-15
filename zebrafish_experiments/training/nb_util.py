from Bio import SeqIO
import numpy as np

# Convert to k-mer representation
def get_kmers(sequence, k=6):
    return ' '.join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

# Tokenize every gene into string of kmers
def tokenize_genes(path):
    print("tokenize_genes CALLED")
    sequences = []
    counter = 0
    for record in SeqIO.parse(path, "fasta"):
        if counter % 100 == 0:
            print(counter)

        #gene_name = record.id.split("|")[0]
        tokenized_seq = get_kmers(str(record.seq))
        sequences.append(tokenized_seq)
        counter += 1

    return np.array(sequences)

def tokenize_genes_substring(path, start, end, k=6):
    print("tokenize_genes_substring CALLED")
    sequences = []
    counter = 0
    for record in SeqIO.parse(path, "fasta"):
        #if counter % 100 == 0:
            #print(counter)

        #gene_name = record.id.split("|")[0]
        tokenized_seq = get_kmers(str(record.seq)[start:end], k)
        sequences.append(tokenized_seq)
        counter += 1

    return np.array(sequences)