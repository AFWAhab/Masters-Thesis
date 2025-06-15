import pandas as pd

#df = pd.read_csv("data/tsvFiles/Homo_sapiens_RNA-Seq_read_counts_TPM_ERP006650.tsv", sep="\t")
#df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_SRP069860_RPKM.tsv", sep="\t")
df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_E-MTAB-5895.tsv", sep="\t")

print("Lenght of dataframe:", len(df))

unique_gene_ids = df["Gene ID"].unique()
df_unique_library_ids = df["Library ID"].unique()

f = open("data/all_values_detection_flag_pig_SScrofa11.txt", "w")
f.write("Gene ID,Majority detection flag\n")

counter = 0
for gene_id in unique_gene_ids:
    counter += 1
    if counter % 100 == 0:
        print("counter:", counter)

    rows = df[df["Gene ID"] == gene_id]
    rows_detection_flag = rows["Detection flag"]

    number_of_present = len([flag for flag in rows_detection_flag if flag == "present"])
    number_of_absent = len([flag for flag in rows_detection_flag if flag == "absent"])

    if number_of_absent > number_of_present:
        majority_flag = "absent"
    else:
        majority_flag = "present"

    f.write(gene_id + "," + str(majority_flag) + "\n")