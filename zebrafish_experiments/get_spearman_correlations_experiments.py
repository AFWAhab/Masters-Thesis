import pandas as pd
from scipy.stats import spearmanr
import statistics

df = pd.read_csv("data/all_values.txt", sep=",")

lib_ids_sanger = ["ERX009445","ERX009447","ERX013539","ERX009448","ERX009449","ERX013540","ERX009444","ERX009450","ERX013538","ERX009443","ERX009446","ERX013537"]
expression_sanger = df[lib_ids_sanger]
lib_ids_ENA = ["ERX4030549","ERX4030545","ERX4030546","ERX4030547","ERX4030552","ERX4030536","ERX4030550","ERX4030531","ERX4030532","ERX4030551","ERX4030548"]
expression_ENA = df[lib_ids_ENA]
lib_ids_SRA = ["SRX661008","SRX661010","SRX661004","SRX661014","SRX661011","SRX661005","SRX661003","SRX661013","SRX661009","SRX661007","SRX661006","SRX661012"]
expression_SRA = df[lib_ids_SRA]

sra_correlations = spearmanr(expression_SRA)
sra_statistic_list = []
sra_pvalue_list = []
for i in range(0,  sra_correlations.statistic.shape[1]):
    for j in range(0, sra_correlations.statistic.shape[1]):
        if i == j:
            continue

        sra_statistic_list.append(sra_correlations.statistic[i, j])
        sra_pvalue_list.append(sra_correlations.pvalue[i, j])

print("mean of SRA Spearman correlations: ", statistics.mean(sra_statistic_list))

sanger_correlations = spearmanr(expression_sanger)
sanger_statistic_list = []
sanger_pvalue_list = []
for i in range(0,  sanger_correlations.statistic.shape[1]):
    for j in range(0, sanger_correlations.statistic.shape[1]):
        if i == j:
            continue

        sanger_statistic_list.append(sanger_correlations.statistic[i, j])
        sanger_pvalue_list.append(sanger_correlations.pvalue[i, j])

print("mean of sanger Spearman correlations: ", statistics.mean(sanger_statistic_list))

ENA_correlations = spearmanr(expression_ENA)
ENA_statistic_list = []
ENA_pvalue_list = []
for i in range(0,  ENA_correlations.statistic.shape[1]):
    for j in range(0, ENA_correlations.statistic.shape[1]):
        if i == j:
            continue

        ENA_statistic_list.append(ENA_correlations.statistic[i, j])
        ENA_pvalue_list.append(ENA_correlations.pvalue[i, j])

print("mean of ENA Spearman correlations: ", statistics.mean(ENA_statistic_list))