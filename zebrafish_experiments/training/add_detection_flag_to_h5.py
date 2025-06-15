import h5py
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

# Load labels from CSV
df = pd.read_csv("../data/all_values_detection_flag.txt", sep=",")
#df = pd.read_csv("../data/all_values_detection_flag_human.txt", sep=",")
#df = pd.read_csv("../data/all_values_detection_flag_pig.txt", sep=",")
#df = pd.read_csv("../data/all_values_detection_flag_pig_SScrofa11.txt", sep=",")
#df = pd.read_csv("../data/all_values.txt", sep=",")
#df = pd.read_csv("../hic/zebrafish_halflife_data.csv")
df = df.set_index("Gene ID")  # Ensure we can match gene names easily
#five_utr_len = df["5' UTR len"]
#three_utr_len = df["3' UTR len"]
#orf_len = df["ORF len"]
#intron_len = df["Intron len"]

#for idx, row in df.iterrows():
#    five_utr_len[idx] = np.log10(five_utr_len[idx] + 0.1)
#    three_utr_len[idx] = np.log10(three_utr_len[idx] + 0.1)
#    orf_len[idx] = np.log10(orf_len[idx] + 0.1)
#    intron_len[idx] = np.log10(intron_len[idx] + 0.1)

#df["5' UTR len"] = five_utr_len
#df["3' UTR len"] = three_utr_len
#df["ORF len"] = orf_len
#df["Intron len"] = intron_len
#numeric_cols = df.select_dtypes(include=[np.number]).columns
#df = df[numeric_cols].apply(zscore)

#f = h5py.File("zebrafish_training/zebrafish_train.hdf5", "a")
#f = h5py.File("zebrafish_training/zebrafish_val.hdf5", "a")
#f = h5py.File("zebrafish_training/zebrafish_test.hdf5", "a")
#f = h5py.File("../paperData/pM10Kb_1KTest/train.h5", "a")
#f = h5py.File("../paperData/pM10Kb_1KTest/valid.h5", "a")
#f = h5py.File("pig.hdf5", "a")
f = h5py.File("zebrafish.hdf5", "a")

gene_name_dataset = f["geneName"]
detection_flags_int = np.empty([len(gene_name_dataset)], dtype="uint8")
#embryo_values = np.empty([len(gene_name_dataset)], dtype="float64")
#halflife_data_array = np.zeros((len(gene_name_dataset), 8), dtype="float64") # for now, we ignore this


for i, gene in enumerate(gene_name_dataset):
    if i % 100 == 0:
        print(i)

    gene_string = str(gene)[2:-1]

    if gene_string in df.index:
        #values = df.loc[gene_string]
        #values_list = values.to_numpy()
        #halflife_data_array[i] = values_list
        #embryo_val = df.loc[gene_string]
        #embryo_values[i] = embryo_val
        majority_detection_flag = df.loc[gene_string, "Majority detection flag"]
        detection_flags_int[i] = 1 if majority_detection_flag == "present" else 0
    else: # we use value 2 to indicate that we don't know if the gene is expressed
        detection_flags_int[i] = 2
        print("oh god oh fuck:", gene_string)

#f.create_dataset("halflifeData", data=halflife_data_array)
f.create_dataset("detectionFlagInt", data=detection_flags_int)