import pandas as pd
from sklearn.metrics import r2_score

median_truth = pd.read_csv("data/median_values_new.txt", sep=",")
median_predicted = pd.read_csv("data/zebrafish_test_predictions_human_model.txt", sep="\t")

predicted_identifiers = median_predicted["ID"]
predicted_gene_IDs = [gene_id.split("|")[0] for gene_id in predicted_identifiers]

median_truth_filtered = [row["Median log RPKM"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_gene_IDs]
with open("data/zebrafish_human_orthologs_median.csv", "w") as f:
    f.write("Median RPKM\n")
    for val in median_truth_filtered:
        f.write(str(val) + "\n")