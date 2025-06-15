import pandas as pd
from get_TSS import process_gtf

df_test_ids = pd.read_csv('../data/zebrafish_test_predictions_human_model.txt', sep='\t')["ID"]
test_ids = [gene_id.split("|")[0] for gene_id in df_test_ids]

df_all_values = pd.read_csv("../data/all_values.txt", sep=",")
df_all_values_minus_test = df_all_values[~df_all_values["Gene ID"].isin(test_ids)]

df_validation_ids = df_all_values_minus_test["Gene ID"][0:1500]
df_train_ids = df_all_values_minus_test["Gene ID"][1500:]

#process_gtf("../data/Danio_rerio.GRCz11.113.gtf", "data/zebrafish_validation_tss.csv", df_validation_ids)
process_gtf("../data/Danio_rerio.GRCz11.113.gtf", "data/zebrafish_train_tss.csv", df_train_ids)