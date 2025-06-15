import gzip
import sys
from functools import partial
from mimetypes import guess_type

import keras
import numpy as np
import pandas as pd
from Bio import SeqIO


# code from
# https://colab.research.google.com/gist/vagarwal87/bdd33e66fa2c59c41409ca47e7132e61/xpresso.ipynb

# the pretrained models need to have 'lr' replaced by 'learning_rate' this can be done with HDFView

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

def generate_predictions(model_file, input_file, output_file):
    model = keras.models.load_model(model_file) #or use one of several pre-trained models
    encoding = guess_type(input_file)[1]  # uses file extension to guess zipped or unzipped

    if encoding is None:
        _open = open
    elif encoding == 'gzip':
        _open = partial(gzip.open, mode='rt')
    else:
        raise ValueError('Unknown file encoding: "{}"'.format(encoding))

    i, bs, names, predictions, sequences = 0, 32, [], [], []
    hlfeatures=6
    halflifedata = np.zeros((bs,hlfeatures), dtype='float32')

    with _open(input_file) as f:
        for fasta in SeqIO.parse(f, 'fasta'):
            name, sequence = fasta.id, str(fasta.seq[3000:13500])
            sequences.append(sequence)
            names.append(name)
            i += 1
            if (len(sequence) != 10500):
                sys.exit( "Error in sequence %s, length is not equal to the required 10,500 nts. Please fix or pad with Ns if necessary." % name )
            if i % bs == 0:
                seq = one_hot(sequences)
                predictions.extend( model.predict([seq, halflifedata], batch_size=bs).flatten().tolist() )
                sequences = []

        remain = i % bs
        if remain > 0:
            halflifedata = np.zeros((remain,hlfeatures), dtype='float32')
            seq = one_hot(sequences)
            predictions.extend( model.predict([seq, halflifedata], batch_size=remain).flatten().tolist() )

        df = pd.DataFrame(np.column_stack((names, predictions)), columns=['ID','SCORE'])
        print(df[1:10]) #print first 10 entries
        df.to_csv(output_file, index=False, header=True, sep='\t')

#generate_predictions(model_file="pretrained_models/GM12878_trainepoch.06-0.5062.h5",
#                     input_file="input_fasta/testinput.fa.gz",
#                     output_file="test_predictions.txt")
generate_predictions(model_file="pretrained_models/humanMedian_trainepoch.11-0.426.h5",
                     input_file="input_fasta/zebrafish_test_promoters_CORRECTED.fasta",
                     output_file="zebrafish_test_human_model_NEW_predictions.txt")