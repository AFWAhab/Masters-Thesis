import numpy as np
import pandas as pd

# Load the TSV file
#df = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_SRP044781.tsv", sep="\t")
df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_E-MTAB-5895.tsv", sep="\t")
#df_tmr = pd.read_csv("data/datasetsExtractedData/SRP044781_total_mapped_reads.txt", sep=",")
df_tmr = pd.read_csv("data/datasetsExtractedData/E-MTAB-5895_total_mapped_reads.txt", sep=",")
#gene_lengths = pd.read_csv("data/datasetsExtractedData/zebrafish_gene_lengths_SRP044781.txt", sep="\t")
gene_lengths = pd.read_csv("data/datasetsExtractedData/pig_gene_lengths_SRP069860.txt", sep="\t")

lib_ids = ['ERX1886666', 'ERX1886697', 'ERX1886673', 'ERX1886669', 'ERX1886694', "ERX1886708", "ERX1886713", 'ERX1886668', 'ERX1886674', 'ERX1886710']
df = df[df["Library ID"].isin(lib_ids)]

new_df = pd.DataFrame(columns=df.columns)
new_df.insert(len(new_df.columns), "Log RPKM", np.empty(0))

#rpkm_col = []
#log_rpkm_col = []

counter = 0
for index, row in df.iterrows():
    if counter % 1000 == 0:
        print(counter)

    lib_id = row["Library ID"]
    aen = row["Anatomical entity name"]
    gene_id = row["Gene ID"]
    test = gene_lengths[gene_lengths["Gene"] == gene_id]['length']
    if len(test) == 0:
        continue

    gene_length = test.iloc[0]

    tmr_in_mil = df_tmr[(df_tmr["Library ID"] == lib_id) & (df_tmr["Anatomical entity name"] == aen)]['Read count'].iloc[0] / 10**6
    rpm = row["Read count"] / tmr_in_mil
    rpkm = rpm / (gene_length / 1000)
    log_rpkm = np.log10(rpkm + 0.1)

    row["Log RPKM"] = log_rpkm
    #rpkm_col.append(rpkm)
    #log_rpkm_col.append(log_rpkm)
    new_df.loc[counter] = row
    counter += 1


#df["RPKM"] = rpkm_col
#df["Log RPKM"] = log_rpkm_col
new_df.to_csv("data/Sus_scrofa_RNA-Seq_read_counts_TPM_E-MTAB-5895_RPKM.tsv", sep="\t")