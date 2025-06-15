import numpy as np
import pandas as pd
from scipy.stats import zscore
import statistics

def get_rpkm(read_count, gene_length, tmr_in_mil):
    rpm = read_count / tmr_in_mil
    rpkm = rpm / (gene_length / 1000)
    log_rpkm = np.log10(rpkm + 0.1)
    return log_rpkm

# Load the TSV file
#df = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_SRP044781.tsv", sep="\t")
#df = pd.read_csv("data/tsvFiles/Sus_scrofa_RNA-Seq_read_counts_TPM_E-MTAB-5895.tsv", sep="\t")
df = pd.read_csv("data/tsvFiles/Estradiol_CountMatrix.txt", sep="\t")
#df_tmr = pd.read_csv("data/datasetsExtractedData/SRP044781_total_mapped_reads.txt", sep=",")
df_tmr = pd.read_csv("data/datasetsExtractedData/Estradiol_total_mapped_reads.txt", sep=",")
df_tmr.set_index("Experiment ID", inplace=True)
#gene_lengths = pd.read_csv("data/datasetsExtractedData/zebrafish_gene_lengths_SRP044781.txt", sep="\t")
#gene_lengths = pd.read_csv("data/datasetsExtractedData/pig_gene_lengths_SRP069860.txt", sep="\t")
gene_lengths = pd.read_csv("data/datasetsExtractedData/zebrafish_gene_lengths_estradiol.txt", sep="\t")

lib_ids = ["gene_id", "R2123", "R2128", "R2133", "R2123_log_RPKM", "R2128_log_RPKM", "R2133_log_RPKM", "median"]
new_df = pd.DataFrame(columns=lib_ids)

#rpkm_col = []
#log_rpkm_col = []

R2123_tmr_in_mil = df_tmr.loc["R2123", "Read count"] / 10**6
R2128_tmr_in_mil = df_tmr.loc["R2128", "Read count"] / 10**6
R2133_tmr_in_mil = df_tmr.loc["R2133", "Read count"] / 10**6
counter = 0
for index, row in df.iterrows():
    if counter % 1000 == 0:
        print(counter)

    #lib_id = row["Library ID"]
    #aen = row["Anatomical entity name"]
    #gene_id = row["Gene ID"]
    test = gene_lengths[gene_lengths["Gene"] == index]['length']
    if len(test) == 0:
        continue

    row["gene_id"] = index
    gene_length = test.iloc[0]
    R2123_log_RPKM, R2128_log_RPKM, R2133_log_RPKM = get_rpkm(row["R2123"], gene_length, R2123_tmr_in_mil), get_rpkm(row["R2128"], gene_length, R2128_tmr_in_mil), get_rpkm(row["R2133"], gene_length, R2133_tmr_in_mil)
    row["R2123_log_RPKM"] = R2123_log_RPKM
    row["R2128_log_RPKM"] = R2128_log_RPKM
    row["R2133_log_RPKM"] = R2133_log_RPKM
    row["median"] = statistics.median([R2123_log_RPKM, R2128_log_RPKM, R2133_log_RPKM])
    new_df.loc[counter] = row
    counter += 1

median_values = new_df["median"].tolist()
new_df["median"] = zscore(median_values)

#df["RPKM"] = rpkm_col
#df["Log RPKM"] = log_rpkm_col
new_df.to_csv("data/Estradiol_CountMatrix_RPKM.tsv", sep="\t", index=False)