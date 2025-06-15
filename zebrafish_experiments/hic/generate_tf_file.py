import pandas as pd

TFLINK_FILE = "TFLink_Danio_rerio_interactions_All_simpleFormat_v1.0.tsv"
TF_MATRIX_FILE = "transcription_factor_zebrafish.csv"

df = pd.read_csv(TFLINK_FILE, sep='\t')

transcription_factors = df["Name.TF"].unique()
targets = df["Name.Target"].unique()
columns = transcription_factors.tolist()
columns.insert(0, "Gene")

print("Number of transcription factors:", len(transcription_factors))
print("Number of targets:", len(targets))

df_tf = pd.DataFrame(columns=columns)
df_tf["Gene"] = targets
df_tf.set_index("Gene", inplace=True)

print(df_tf.head())

for gene_name in targets:
    interacting_tfs = df[df["Name.Target"] == gene_name]["Name.TF"]
    for tf in transcription_factors:
        if tf in interacting_tfs.tolist():
            df_tf.at[gene_name, tf] = 1
        else:
            df_tf.at[gene_name, tf] = 0

df_tf.to_csv(TF_MATRIX_FILE, index=True)