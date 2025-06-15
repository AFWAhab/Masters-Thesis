import numpy as np

def one_hot(seq):
    num_seqs = len(seq)
    seq_len = len(seq[0])
    seqindex = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}
    seq_vec = np.zeros((num_seqs,seq_len,4), dtype='bool')
    for i in range(num_seqs):
        thisseq = seq[i]
        for j in range(seq_len):
            try:
                seq_vec[i,j,seqindex[thisseq[j]]] = 1
            except:
                pass
    return seq_vec

