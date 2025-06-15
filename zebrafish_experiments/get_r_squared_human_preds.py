import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
from scipy import stats

median_predicted = pd.read_csv("paperData/human_predictions.txt", sep="\t")
median_predicted_ids = median_predicted["ID"]
median_truth = pd.read_excel("paperData/416685-1.xlsx", sheet_name="Human", header=3)
median_truth_ids = median_truth["Ensembl Gene ID"]

id_intersection = np.intersect1d(median_predicted_ids, median_truth_ids)

median_predicted_filtered = median_predicted[median_predicted["ID"].isin(id_intersection)]
median_truth_filtered = median_truth[median_truth["Ensembl Gene ID"].isin(id_intersection)]

slope, intercept, r_value, p_value, std_err = stats.linregress(median_predicted_filtered.tolist(), median_truth_filtered.tolist())
print('Test R^2 = %.3f' % r_value ** 2)