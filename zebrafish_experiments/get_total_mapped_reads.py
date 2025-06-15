import pandas as pd

# Load the TSV file
#df = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_SRP044781.tsv", sep="\t")
#df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_SRP069860_RPKM.tsv", sep="\t")
#df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_E-MTAB-5895.tsv", sep="\t")
df = pd.read_csv("data/tsvFiles/Estradiol_CountMatrix.txt", sep="\t")


read_count_dict = {}

for index, row in df.iterrows():
    for col in row.keys():
        if col not in read_count_dict:
            read_count_dict[col] = row[col]
        else:
            read_count_dict[col] = read_count_dict[col] + row[col]

# for index, row in df.iterrows():
#     lib_id = row["Library ID"]
#     aen = row["Anatomical entity name"]
#     idx = lib_id + "," + aen
#
#     if idx not in read_count_dict:
#         read_count_dict[idx] = row["Read count"]
#     else:
#         read_count_dict[idx] = read_count_dict[idx] + row["Read count"]


# Write to a new file
with open("data/datasetsExtractedData/Estradiol_total_mapped_reads.txt", "w") as f:
    f.write("Experiment ID,Read count\n")
    for k,v in read_count_dict.items():
        f.write(k + "," + str(v) + "\n")