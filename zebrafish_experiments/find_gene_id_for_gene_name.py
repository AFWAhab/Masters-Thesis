from util.gtf2df import gtf2df

wantedGenes = []
with open("data/removedGenes/manual_sex_gene_search.txt", mode='r') as f:
    wantedGenes = f.readlines()

wantedGenes = list(set(wantedGenes))
wantedGenes = [ x.rstrip() for x in wantedGenes]

df = gtf2df("data/Danio_rerio.GRCz11.113.gtf")

print(df.columns)

with open("data/removedGenes/bias_genes.txt", mode='a') as bias_genes:
    for gene in wantedGenes:
        bruh = df[(df['gene_name'] == gene) & (df['feature'] == "gene")]["gene_id"]
        if len(bruh) == 0:
            continue

        gene_id = bruh.iloc[0]
        # bias_genes.write(gene_id + "," + gene + "\n")
        print(gene_id + "," + gene + "\n")

print("Done")