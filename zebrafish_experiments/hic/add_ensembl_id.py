import pandas as pd

GTF_FILE = "../data/Danio_rerio.GRCz11.113.gtf"

# Read a GTF file with gene info
gtf = pd.read_csv(GTF_FILE, sep='\t', comment='#', header=None)
gtf.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

# Filter to only 'gene' features
genes = gtf[gtf['feature'] == 'gene']

# Parse attributes to extract gene_id and gene_name
genes['gene_id'] = genes['attribute'].str.extract('gene_id "([^"]+)"')
genes['gene_name'] = genes['attribute'].str.extract('gene_name "([^"]+)"')

# Now you can map:
gene_to_id = dict(zip(genes['gene_name'], genes['gene_id']))

TF_MATRIX_FILE = "transcription_factor_zebrafish.csv"

df = pd.read_csv(TF_MATRIX_FILE, sep=',')
gene_names = df["Gene"]
ensembl_ids = []
for gene_name in gene_names:
    if gene_name not in gene_to_id.keys():
        ensembl_ids.append("MISSING")
        print(gene_name, "is MISSING")
    else:
        ensembl_ids.append(gene_to_id[gene_name])

#ensembl_ids = [gene_to_id[gene_name] for gene_name in gene_names]
df = df.assign(ensembl_id=ensembl_ids)
df.to_csv("tf_zebrafish.csv", index=True)