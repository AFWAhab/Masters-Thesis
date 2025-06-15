import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import zscore
from scipy import stats

median_truth = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_ERP000447_RPKM.tsv", sep="\t")
median_predicted = pd.read_csv("paperData/zebrafish_test_human_model_NEW_predictions.txt", sep="\t")

predicted_identifiers = median_predicted["ID"].tolist()
#predicted_gene_IDs = [gene_id.split("|")[0] for gene_id in predicted_identifiers]

# heart
median_truth_heart = [row["Log RPKM"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_identifiers
                         and row["Library ID"] == "ERX009445"]

#r_squared_heart = r2_score(zscore(median_truth_heart), median_predicted["SCORE"])
slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted["SCORE"], zscore(median_truth_heart))
print('Test R^2 = %.3f' % r_value ** 2)
#print("r squared for heart:", r_squared_heart)

# brain
median_truth_brain = [row["Log RPKM"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_identifiers
                         and row["Library ID"] == "ERX009448"]

#r_squared_brain = r2_score(zscore(median_truth_brain), median_predicted["SCORE"])
#print("r squared for brain:", r_squared_brain)
slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted["SCORE"], zscore(median_truth_brain))
print('Test R^2 = %.3f' % r_value ** 2)

# swim bladder
median_truth_swim = [row["Log RPKM"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_identifiers
                         and row["Library ID"] == "ERX009444"]

#r_squared_swim = r2_score(zscore(median_truth_swim), median_predicted["SCORE"])
#print("r squared for swim bladder:", r_squared_swim)
slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted["SCORE"], zscore(median_truth_swim))
print('Test R^2 = %.3f' % r_value ** 2)

# head kidney
median_truth_kidney = [row["Log RPKM"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_identifiers
                         and row["Library ID"] == "ERX009443"]

#r_squared_kidney = r2_score(zscore(median_truth_kidney), median_predicted["SCORE"])
#print("r squared for head kidney:", r_squared_kidney)
slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted["SCORE"], zscore(median_truth_kidney))
print('Test R^2 = %.3f' % r_value ** 2)