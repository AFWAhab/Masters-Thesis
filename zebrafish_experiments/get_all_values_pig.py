import pandas as pd

#df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_SRP069860_RPKM_final.tsv", sep="\t")
df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_E-MTAB-5895_RPKM.tsv", sep="\t")
print("Lenght of dataframe:", len(df))

unique_gene_ids = df["Gene ID"].unique()

unique_library_ids = df["Library ID"].unique()

f = open("data/all_values_pig_SScrofa11.txt", "w")
f.write("Gene ID,Median log RPKM,")

for lib_id in unique_library_ids:
    f.write(lib_id + ",")

f.write("\n")

counter = 0
for gene_id in unique_gene_ids:
    counter += 1
    if counter % 100 == 0:
        print("counter:", counter)

    rows = df[df["Gene ID"] == gene_id]
    rows_log_rpkm = rows["Log RPKM"]

    #if len(rows_log_rpkm) != 35:
    #    continue

    median_log = rows_log_rpkm.median()
    f.write(gene_id + "," + str(median_log) + ",")

    for val in rows_log_rpkm:
        f.write(str(val) + ",")

    f.write("\n")