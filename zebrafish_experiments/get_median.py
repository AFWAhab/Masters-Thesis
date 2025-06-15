import pandas as pd

df1 = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_ERP000447_RPKM.tsv", sep="\t")
df2 = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_ERP121186_RPKM.tsv", sep="\t")
df3 = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_SRP044781_RPKM.tsv", sep="\t")

frames = [df1, df2, df3]
df = pd.concat(frames)
print("Lenght of dataframe:", len(df))

unique_gene_ids = df["Gene ID"].unique()

f = open("data/median_values_new.txt", "w")
f.write("Gene ID,Median log RPKM\n")
counter = 0
for gene_id in unique_gene_ids:
    counter += 1
    if counter % 100 == 0:
        print("counter:", counter)

    rows = df[df["Gene ID"] == gene_id]
    rows_log_rpkm = rows["Log RPKM"]
    median_log = rows_log_rpkm.median()
    f.write(gene_id + "," + str(median_log) + "\n")
