import pandas as pd

# Load the TSV file
df = pd.read_csv("data/removedGenes/amigo_results.txt", sep="\t")

# Extract unique Gene IDs
unique_gene_ids = df["Gene/product"].drop_duplicates()

print("Number of unique gene products", len(unique_gene_ids))

