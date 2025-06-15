import pandas as pd
from pybiomart import Server

MISSING_GENES = "missing genes"

missing_genes = pd.read_csv(MISSING_GENES, header=None)

# Connect to Ensembl BioMart
server = Server(host='http://www.ensembl.org')
dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['drerio_gene_ensembl']

f = open("found_genes.txt", "w")
for missing_gene in missing_genes:
    result = dataset.query(attributes=['external_gene_name', 'ensembl_gene_id'],
                           filters={'external_gene_name': [missing_gene]})