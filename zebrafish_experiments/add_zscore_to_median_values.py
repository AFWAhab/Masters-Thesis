import pandas as pd
from scipy.stats import zscore

# Load the file into a DataFrame
df = pd.read_csv('data/all_values.txt')

# Compute z-score normalization
df['SRX661011_Z-score'] = zscore(df['SRX661011'])

# Save to a new file (or overwrite the original if you want)
df.to_csv('data/all_values.txt', index=False)