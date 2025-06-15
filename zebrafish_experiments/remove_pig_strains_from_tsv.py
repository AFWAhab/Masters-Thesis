import pandas as pd

df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_SRP069860.tsv", sep="\t")
new_df = df[df["Strain"] == "Hampshire"]
result = new_df.to_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_SRP069860_RPKM.tsv", sep='\t', index=False)