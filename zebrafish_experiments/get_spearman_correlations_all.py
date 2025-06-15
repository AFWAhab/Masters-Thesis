import pandas as pd
from scipy.stats import spearmanr
import statistics


df = pd.read_csv("data/all_values.txt", sep=",")
expression_values = df.iloc[:, 2:len(df.columns) - 1]
correlations = spearmanr(expression_values)
statistic = correlations.statistic
pvalue = correlations.pvalue

statistic_list = []
pvalue_list = []
for i in range(0, statistic.shape[1]):
    for j in range(0, statistic.shape[1]):
        if i == j:
            continue

        statistic_list.append(statistic[i, j])
        pvalue_list.append(pvalue[i, j])

print(statistic_list)
print("mean of Spearman correlations: ", statistics.mean(statistic_list))