import re
import sys

# Run this
# python get_gene_lengths.py data/zebrafish_gene_name_ERP000447.txt data/Danio_rerio.GRCz11.113.gtf
# python get_gene_lengths.py data/zebrafish_gene_name_ERP121186.txt data/Danio_rerio.GRCz11.113.gtf
# python get_gene_lengths.py data/zebrafish_gene_name_SRP044781.txt data/Danio_rerio.GRCz11.113.gtf
# python get_gene_lengths.py data/pig_gene_name_SRP069860.txt data/Sus_scrofa.Sscrofa11.1.113.gtf
# python get_gene_lengths.py data/datasetsExtractedData/zebrafish_gene_name_estradiol.txt data/Danio_rerio.GRCz11.113.gtf

# source of Danio_rerio.GRCz11.113.gtf (summary of the gene annotation information)
# https://ftp.ensembl.org/pub/release-113/gtf/danio_rerio/

# source of Danio_rerio_RNA-Seq_read_counts_TPM_ERP000447.tsv (sequence reads)
# https://www.bgee.org/experiment/ERP000447

# soure of script
# https://bioinformatics.stackexchange.com/questions/4942/finding-gene-length-using-ensembl-id

wantedGenes = []
with open(sys.argv[1], mode='r') as f:
    wantedGenes = f.readlines()

wantedGenes = [ x.rstrip() for x in wantedGenes]

print("Starting to read lengths of genes")

with open(sys.argv[2], mode='r') as f:
    with open("data/datasetsExtractedData/zebrafish_gene_lengths_estradiol.txt", mode='w') as f_lengths:
        f_lengths.write('Gene\tstart\tend\tlength\n')
        counter = 0
        for line in f:
            if not line.startswith('#'):
                fields = line.split('\t')
                annotations = fields[8].split(' ')
                geneName = re.sub('[";]', '', annotations[1])
                if geneName in wantedGenes and fields[2] == 'gene':
                    f_lengths.write('%s\t%s\t%s\t%d\n' % (geneName, fields[3], fields[4], int(int(fields[4]) - int(fields[3]))))
                    if counter % 1000 == 0:
                        print("read length of", counter, "genes")
                    counter += 1

print("Done!")