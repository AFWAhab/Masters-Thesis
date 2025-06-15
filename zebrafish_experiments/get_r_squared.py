import pandas as pd
from sklearn.metrics import r2_score
from scipy import stats

median_truth = pd.read_csv("data/median_values_new.txt", sep=",")
#median_truth = pd.read_csv("data/Estradiol_CountMatrix_RPKM.tsv", sep="\t")
median_predicted = pd.read_csv("paperData/zebrafish_test_human_model_NEW_predictions.txt", sep="\t")

predicted_identifiers = median_predicted["ID"].tolist()
#predicted_gene_IDs = [gene_id.split("|")[0] for gene_id in predicted_identifiers]

#median_truth_filtered = median_truth[median_truth["Gene ID"] in predicted_gene_IDs]
median_truth_filtered = [row["Z-score"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_identifiers]
#median_truth_filtered = [row["R2133_log_RPKM"] for idx, row in median_truth.iterrows() if row["gene_id"] in predicted_identifiers]


r_squared = r2_score(median_truth_filtered, median_predicted["SCORE"])
print("r squared:", r_squared)
slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted["SCORE"], median_truth_filtered)
print('Test R^2 = %.3f' % r_value ** 2)