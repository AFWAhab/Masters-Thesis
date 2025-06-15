import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

zebrafish = pd.read_csv('data/all_values.txt', sep=',')
human = pd.read_excel("paperData/416685-1.xlsx", sheet_name="Human", header=3)
mouse = pd.read_excel("paperData/416685-1.xlsx", sheet_name="Mouse", header=3)
# add data for pigs
with h5py.File("training/pig.hdf5", "r") as f:
    dataset = f["label"]
    pig = dataset[:]

human_zebrafish_orthologs = pd.read_csv("data/zebrafish_human_orthologs_median.csv", sep=',')

zebrafish_median_expression = zebrafish['Median log RPKM']
zebrafish_embryo_expression = zebrafish['SRX661011']
human_zebrafish_orthologs_expression = human_zebrafish_orthologs['Median RPKM']
human_median_expression = human['Median expression']
mouse_median_expression = mouse[mouse['Median expression'] != "NA"]['Median expression']

# Compute cumulative distribution
def compute_cdf(values):
    sorted_values = np.sort(values)
    cdf = np.arange(1, len(values) + 1) / len(values)
    return sorted_values, cdf

zebrafish_sorted, zebrafish_cdf = compute_cdf(zebrafish_median_expression)
zebrafish_embryo_sorted, zebrafish_embryo_cdf = compute_cdf(zebrafish_embryo_expression)
human_zebrafish_orthologs_sorted, human_zebrafish_orthologs_cdf = compute_cdf(human_zebrafish_orthologs_expression)
human_sorted, human_cdf = compute_cdf(human_median_expression)
mouse_sorted, mouse_cdf = compute_cdf(mouse_median_expression)
pig_sorted, pig_cdf = compute_cdf(pig)

# Plot CDFs
plt.figure(figsize=(8, 6))
plt.plot(zebrafish_sorted, zebrafish_cdf, label="Zebrafish median (25,403)", color="blue")
plt.plot(zebrafish_embryo_sorted, zebrafish_embryo_cdf, label="Zebrafish embryo tissue (25,403)", color="orange")
plt.plot(human_zebrafish_orthologs_sorted, human_zebrafish_orthologs_cdf, label="Zebrafish median, 1:1 orthologous to human genes (1,471)", color="teal")
plt.plot(human_sorted, human_cdf, label="Human median (18,377)", color="purple")
plt.plot(mouse_sorted, mouse_cdf, label="Mouse median (21,856)", color="red")
plt.plot(pig_sorted, pig_cdf, label="Pig median (16,349)", color="black")
plt.xlabel("mRNA expression level (log10)")
plt.ylabel("Cumulative fraction")
plt.title("Cumulative distributions of mRNA expression levels")
plt.legend()
plt.grid()
plt.show()