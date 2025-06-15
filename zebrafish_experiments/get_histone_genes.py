import pandas as pd

# Load the TSV file
df = pd.read_csv("data/removedGenes/output_histone_search.csv", sep=",")

# Write to a new file
with open("data/removedGenes/bias_genes.txt", "w") as f:
    for idx, row in df.iterrows():
        ensembl_id = row["id_with_url"]
        gene_name = row["name"]
        description = row["description"]
        if "histone" in description:
            f.write(str(ensembl_id) + "," + str(gene_name) + "\n")

print("Done")