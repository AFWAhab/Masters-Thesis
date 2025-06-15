import pandas as pd
from Bio import SeqIO
import gzip
import os


def get_promoter_seq(csv_file, genome_dir, output_fasta, kb_upstream, kb_downstream):
    # Read CSV
    df = pd.read_csv(csv_file, sep="\t")

    # Open output FASTA file for writing
    with open(output_fasta, "w") as fasta_out:
        counter = 0

        for _, row in df.iterrows():
            counter += 1
            if counter % 100 == 0:
                print(counter)

            gene_id = row.loc["Gene ID"]
            transcript_id = row.loc["Representative Transcript ID"]
            seq_id = str(row.loc["Chromosome"])  # Chromosome number
            tss = int(row.loc["TSS"])
            strand = row.loc["strand"]

            # Compute promoter region
            start = max(0, tss - kb_upstream)  # Ensure no negative start
            if start == 0:
                print("negative start for", gene_id)
                continue

            end = tss + kb_downstream

            # Locate the chromosome file
            chr_file = os.path.join(genome_dir, f"Danio_rerio.GRCz11.dna.chromosome.{seq_id}.fa.gz")

            if not os.path.exists(chr_file):
                print(f"Warning: File {chr_file} not found. Skipping {gene_id}.")
                continue

            # Extract sequence from the genome
            with gzip.open(chr_file, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    seq = record.seq[start:end]
                    if strand == "-":
                        seq = seq.reverse_complement()
                    break  # Only one sequence per file

            # Write to FASTA
            fasta_out.write(f">{gene_id}|{transcript_id}|{seq_id}:{start}-{end}\n{seq}\n")

    print(f"FASTA file saved: {output_fasta}")

# Paths
csv_file_tss = "data/pig_tss_CORRECTED.csv"
genome_dir = ""
output_fasta = ""

# promoter region 10kb upstream and downstream
get_promoter_seq(csv_file_tss, genome_dir, output_fasta, 10000, 10000)