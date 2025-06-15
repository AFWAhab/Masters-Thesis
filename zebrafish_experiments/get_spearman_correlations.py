import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv("data/all_values.txt", sep=",")

print(len(df.columns))

heart_ERX009445 = df["ERX009445"]  # ERX009445,heart
brain_ERX009448 = df["ERX009448"]  # ERX009448,brain
eye_ERX4030546 = df["ERX4030546"]
digestive_tract_ERX4030547 = df["ERX4030547"]
mesonephros_SRX661008 = df["SRX661008"]
bone_element_SRX661009 = df["SRX661009"]

print(spearmanr(heart_ERX009445, brain_ERX009448))
print(spearmanr(heart_ERX009445, eye_ERX4030546))
print(spearmanr(heart_ERX009445, digestive_tract_ERX4030547))
print(spearmanr(heart_ERX009445, mesonephros_SRX661008))
print(spearmanr(heart_ERX009445, bone_element_SRX661009))

print(spearmanr(brain_ERX009448, eye_ERX4030546))
print(spearmanr(brain_ERX009448, digestive_tract_ERX4030547))
print(spearmanr(brain_ERX009448, mesonephros_SRX661008))
print(spearmanr(brain_ERX009448, bone_element_SRX661009))

print(spearmanr(eye_ERX4030546, digestive_tract_ERX4030547))
print(spearmanr(eye_ERX4030546, mesonephros_SRX661008))
print(spearmanr(eye_ERX4030546, bone_element_SRX661009))

print(spearmanr(digestive_tract_ERX4030547, mesonephros_SRX661008))
print(spearmanr(digestive_tract_ERX4030547, bone_element_SRX661009))

print(spearmanr(mesonephros_SRX661008, bone_element_SRX661009))