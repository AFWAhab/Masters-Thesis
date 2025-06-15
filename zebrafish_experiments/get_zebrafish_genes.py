import pandas as pd

# Load the TSV file
#df = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_SRP044781.tsv", sep="\t")
#df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_SRP069860_RPKM.tsv", sep="\t")
df = pd.read_csv("data/tsvFiles/Estradiol_CountMatrix.txt", sep="\t")

# Extract unique Gene IDs
#unique_gene_ids = df["Gene ID"].drop_duplicates()

# Write to a new file
with open("data/datasetsExtractedData/zebrafish_gene_name_estradiol.txt", "w") as f:
    #for gene_id in unique_gene_ids:
    #    f.write(str(gene_id) + "\n")
    for idx, row in df.iterrows():
        f.write(str(idx) + "\n")

print("Unique Gene IDs written to file")