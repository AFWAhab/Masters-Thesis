import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import zscore
from scipy import stats

median_truth = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_SRP044781_RPKM.tsv", sep="\t")
median_predicted = pd.read_csv("paperData/zebrafish_test_human_model_NEW_predictions.txt", sep="\t")

predicted_identifiers = median_predicted["ID"].tolist()
#predicted_gene_IDs = [gene_id.split("|")[0] for gene_id in predicted_identifiers]

# heart
median_truth_heart = [row["Log RPKM"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_identifiers
                         and row["Anatomical entity name"] == "heart"]

#r_squared_heart = r2_score(zscore(median_truth_heart), median_predicted["SCORE"])
#print("r squared for heart:", r_squared_heart)
slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted["SCORE"], zscore(median_truth_heart))
print('Test R^2 = %.3f' % r_value ** 2)

# testis
median_truth_testis = [row["Log RPKM"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_identifiers
                         and row["Anatomical entity name"] == "testis"]

#r_squared_testis = r2_score(zscore(median_truth_testis), median_predicted["SCORE"])
#print("r squared for testis:", r_squared_testis)
slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted["SCORE"], zscore(median_truth_testis))
print('Test R^2 = %.3f' % r_value ** 2)

# ovary
median_truth_ovary = [row["Log RPKM"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_identifiers
                         and row["Anatomical entity name"] == "ovary"]

#r_squared_ovary = r2_score(zscore(median_truth_ovary), median_predicted["SCORE"])
#print("r squared for ovary:", r_squared_ovary)
slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted["SCORE"], zscore(median_truth_ovary))
print('Test R^2 = %.3f' % r_value ** 2)

# embryo
median_truth_embryo = [row["Log RPKM"] for idx, row in median_truth.iterrows() if row["Gene ID"] in predicted_identifiers
                         and row["Anatomical entity name"] == "embryo"]

#r_squared_embryo = r2_score(zscore(median_truth_embryo), median_predicted["SCORE"])
#print("r squared for embryo:", r_squared_embryo)
slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted["SCORE"], zscore(median_truth_embryo))
print('Test R^2 = %.3f' % r_value ** 2)