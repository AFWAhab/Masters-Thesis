import csv
import gzip
import os

import Bio.SeqUtils
import pandas as pd
from Bio import SeqIO

from util.gtf2df import gtf2df

#genome_dir = "..\\data\\zebrafishGenome"
genome_dir = "" # PIG GENOME HERE

def get_orf_length(start_codon_row, stop_codon_row):
    start = start_codon_row.loc["start"]
    end = stop_codon_row.loc["start"]
    if start > end:
        return int(start_codon_row.loc["end"]) - int(stop_codon_row.loc["start"])
    else:
        return int(stop_codon_row.loc["end"]) - int(start_codon_row.loc["start"])


def get_intron_length(exon_rows):
    intron_length = 0
    if len(exon_rows) > 1:
        exon_regions = exon_rows.sort_values(by=['start'])
        i = 0
        for _, exon_region in exon_regions.iterrows():
            if i < len(exon_regions) - 1:
                intron_length += exon_regions.iloc[i + 1]["start"] - exon_regions.iloc[i]["end"] - 1
                i = i + 1

    return intron_length

def get_gc(rows, record):
    orf_sequences = ""
    for idx, cds_row in rows.iterrows():
        orf_sequences = orf_sequences + record.seq[int(cds_row.loc["start"]):int(cds_row.loc["end"]) + 1]

    if orf_sequences == "":
        return 0

    return Bio.SeqUtils.gc_fraction(orf_sequences)

def get_gc_orf(start_codon_row, stop_codon_row, record):
    start = start_codon_row.loc["start"]
    end = stop_codon_row.loc["start"]
    if start > end:
        orf_sequence = record.seq[stop_codon_row.loc["start"]:start_codon_row.loc["end"]]
    else:
        orf_sequence = record.seq[start_codon_row.loc["start"]:stop_codon_row.loc["end"]]

    return Bio.SeqUtils.gc_fraction(orf_sequence)


def get_orf_exon_junction_density(exons, len_orf):
    orf_exon_count = len(exons)
    orf_exon_junctions = orf_exon_count - 1 if orf_exon_count > 1 else 0
    orf_junction_density = (orf_exon_junctions / (len_orf / 1000)) if len_orf > 0 else 0
    return orf_junction_density

def get_utr_lengths(five_prime_utrs, three_prime_utrs):
    return (
        sum(int(utr.loc["end"]) - int(utr.loc["start"]) + 1 for idx, utr in five_prime_utrs.iterrows()),
        sum(int(utr.loc["end"]) - int(utr.loc["start"]) + 1 for idx, utr in three_prime_utrs.iterrows())
    )


def process_gtf(gtf_file, output_csv, ids):
    df = gtf2df(gtf_file)
    counter = 0

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Gene ID", "Representative Transcript ID", "5' UTR len", "3' UTR len", "ORF len", "5' UTR gc", "3' UTR gc", "ORF gc", "Intron len", "Exon junction density"])

        for idx, row in ids.iterrows():
            counter += 1
            if counter % 100 == 0:
                print("at counter", counter)

            transcript_id = row.loc["Representative Transcript ID"]

            #chr_file = os.path.join(genome_dir, f"Danio_rerio.GRCz11.dna.chromosome.{row.loc["Chromosome"]}.fa.gz")
            chr_file = os.path.join(genome_dir, f"Sus_scrofa.Sscrofa11.1.dna.primary_assembly.{row.loc["Chromosome"]}.fa.gz")
            if not os.path.exists(chr_file):
                print(f"Warning: File {chr_file} not found. Skipping {transcript_id}.")
                continue

            # Extract halflife data from the genome
            with gzip.open(chr_file, "rt") as handle:
                start_codon = df[(df["transcript_id"] == transcript_id) & (df["feature"] == "start_codon")]
                stop_codon = df[(df["transcript_id"] == transcript_id) & (df["feature"] == "stop_codon")]
                if start_codon.empty or stop_codon.empty:
                    #print("Transcript ", transcript_id, " has no start codon or stop codon.")
                    continue

                for record in SeqIO.parse(handle, "fasta"):
                    five_prime_utrs = df[(df["transcript_id"] == transcript_id) & (df["feature"] == "five_prime_utr")]
                    three_prime_utrs = df[(df["transcript_id"] == transcript_id) & (df["feature"] == "three_prime_utr")]
                    five_prime_utrs_len, three_prime_utrs_len = get_utr_lengths(five_prime_utrs, three_prime_utrs)
                    five_prime_utrs_gc = get_gc(five_prime_utrs, record)
                    three_prime_utrs_gc = get_gc(three_prime_utrs, record)

                    orf_len = get_orf_length(start_codon.iloc[0], stop_codon.iloc[0])
                    orf_gc = get_gc_orf(start_codon.iloc[0], stop_codon.iloc[0], record)
                    exons = df[(df["transcript_id"] == transcript_id) & (df["feature"] == "exon")]
                    intron_len = get_intron_length(exons)
                    orf_junction_density = get_orf_exon_junction_density(exons, orf_len)
                    writer.writerow([row.loc["Gene ID"], transcript_id, five_prime_utrs_len, three_prime_utrs_len, orf_len, five_prime_utrs_gc, three_prime_utrs_gc, orf_gc, intron_len, orf_junction_density])
                    #break

    print("done!")

#gene_ids_transcript_ids = pd.read_csv("..\\data\\zebrafish_tss_CORRECTED_w_strands.csv", sep="\t")
#process_gtf("..\\data\\Danio_rerio.GRCz11.113.gtf", "zebrafish_halflife_data.csv", gene_ids_transcript_ids)
gene_ids = pd.read_csv("..\\data\\pig_tss_CORRECTED.csv", sep=",")
process_gtf("..\\data\\Sus_scrofa.Sscrofa11.1.113.gtf", "pig_halflife_data.csv", gene_ids)