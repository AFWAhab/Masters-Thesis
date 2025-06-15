import pandas as pd

# DOESN'T WORK

# Load the TSV file
df = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_ERP000447.tsv", sep="\t")
df_tmr = pd.read_csv("data/datasetsExtractedData/ERP000447_total_mapped_reads.txt", sep=",")

rpkm = []

for index, row in df.iterrows():
    tpm = row["TPM"]
    lib_id = row["Library ID"]
    aen = row["Anatomical entity name"]
    tmr_in_mil = df_tmr[(df_tmr["Library ID"] == lib_id) & (df_tmr["Anatomical entity name"] == aen)]['Read count'].iloc[0] / 10**6
    numerator = tpm * tmr_in_mil
    denominator = df_tmr[(df_tmr["Library ID"] == lib_id) & (df_tmr["Anatomical entity name"] == aen)]['Denominator'].iloc[0]
    rpkm.append(numerator / denominator)

df["RPKM"] = rpkm
df.to_csv("data/Danio_rerio_RNA-Seq_read_counts_TPM_ERP000447_RPKM.tsv", sep="\t")