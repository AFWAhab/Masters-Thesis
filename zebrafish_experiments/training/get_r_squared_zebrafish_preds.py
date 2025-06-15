import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy import stats


#results = pd.read_csv("zebrafish_training/human_hyperparameters/predictions_7_iteration_binary_new.txt", sep="\t")
results = pd.read_csv("zebrafish_training_NEW/bestparams_7_new_median_adam_default.txt", sep="\t")
preds = [pred[2:-1] for pred in results["Pred"]] # weird formatting adds b' in front of the number and ' at the end
actual = [actual[2:-1] for actual in results["Actual"]]


r_squared = r2_score(actual, preds)
mse = mean_squared_error(actual, preds)
mean_actual_error = mean_absolute_error(actual, preds)
#acc = accuracy_score(actual, preds)

#print("r squared:", r_squared)
print("mse:", mse)
print("mean absolute error:", mean_actual_error)
#print("acc:", acc)
print("mean of test data:", np.mean([float(num) for num in actual]))
print("median of test data:", np.median([float(num) for num in actual]))

preds_int = [float(pred) for pred in preds]
actual_int = [float(actual) for actual in actual]

slope, intercept, r_value, p_value, std_err = stats.linregress(preds_int, actual_int)
print('Test R^2 = %.3f' % r_value ** 2)