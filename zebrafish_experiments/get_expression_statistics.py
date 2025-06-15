import pandas as pd
import statistics

df1 = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_ERP000447_RPKM.tsv", sep="\t")
df2 = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_ERP121186_RPKM.tsv", sep="\t")
df3 = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_SRP044781_RPKM.tsv", sep="\t")

frames = [df1, df2, df3]
df = pd.concat(frames)
print("Lenght of dataframe:", len(df))

gene_predictions = pd.read_csv("data/zebrafish_tss.csv", sep=",")

f = open("data/zebrafish_expression_statistics.txt", "w")
f.write("Gene ID,Median RPKM,Mean RPKM,Sample variance RPKM,Difference RPKM,Median log RPKM,Mean log RPKM,Sample variance log RPKM,Difference log RPKM\n")
counter = 0
for gene_id in gene_predictions["Gene ID"]:
    counter += 1
    if counter % 100 == 0:
        print("counter:", counter)

    rows = df[df["Gene ID"] == gene_id]

    rows_rpkm = rows["RPKM"]
    median = rows_rpkm.median()
    mean_val = statistics.mean(rows_rpkm)
    sample_variance = statistics.variance(rows_rpkm)
    difference = max(rows_rpkm) - min(rows_rpkm)

    rows_log_rpkm = rows["Log RPKM"]
    median_log = rows_log_rpkm.median()
    mean_val_log = statistics.mean(rows_log_rpkm)
    sample_variance_log = statistics.variance(rows_log_rpkm)
    difference_log = max(rows_log_rpkm) - min(rows_log_rpkm)

    f.write(gene_id + "," + str(median) + "," + str(mean_val) + "," + str(sample_variance) + "," + str(difference) + "," + str(median_log) + "," + str(mean_val_log) + "," + str(sample_variance_log) + "," + str(difference_log) + "\n")

print("done")