from operator import truediv
import time
import os

import pandas as pd
from sympy import false

UPSTREAM_BASES = 7000  # 7000 * 3
DOWNSTREAM_BASES = 3500  # 3500 * 3
CHROMOSOME_PATH = "../data/pig/raw_chromosomes/"
OUTPUT_PATH = "../data/pig/fasta_output/10k-corrected/N_values/"
LAST_GENE_ID = None
found_last_gene = False


def get_compliment(base):
    match base.upper():
        case 'A':
            return 'T'
        case 'T':
            return 'A'
        case 'C':
            return 'G'
        case 'G':
            return 'C'
    return base

def convert_bases_to_string(bases, reverse):
    result = ""
    if not reverse:
        for base in bases:
            result += base.upper()
        return result

    for base in bases:
        result += get_compliment(base).upper()
    reversed_result = result[::-1]

    return reversed_result

def save_fasta_file(gene_id, chromosome, tss, reverse):
    global found_last_gene
    if not found_last_gene:
        if LAST_GENE_ID is None or gene_id in LAST_GENE_ID:
            found_last_gene = True
        return
    if len(chromosome) > 2 or chromosome.upper() == "Y":
        print(f"{chromosome} not supported")
        return
    bases = []
    start = time.time()
    with open(CHROMOSOME_PATH + "Sus_scrofa.Sscrofa11.1.dna.primary_assembly." + chromosome + ".fa", 'r') as file:
        upstream = UPSTREAM_BASES
        downstream = DOWNSTREAM_BASES
        if reverse:
            upstream = DOWNSTREAM_BASES
            downstream = UPSTREAM_BASES
        first_base = tss - upstream
        bases_read = 0
        chars_per_line = 60
        lines_to_skip = first_base // 60
        lines_skipped = 0
        first_line = True
        N_found = 0
        for line in file:
            if first_line:
                first_line = False
                continue

            if lines_skipped < lines_to_skip:
                lines_skipped += 1
                bases_read += chars_per_line
                continue

            for base in line:
                if base.upper() not in "ACTG":
                    if base.upper() == "N" and bases_read < first_base:
                        bases_read += 1
                        continue
                    continue
                if bases_read > first_base and base.upper() == "N":
                    print("Found N!")
                    N_found += 1
                if len(bases) == upstream + downstream:
                    break
                if bases_read > first_base:
                    bases.append(base)
                bases_read += 1

    with open(OUTPUT_PATH + "N_data.csv", 'a') as file:
        title = "N_bases_skipped,total_bases,chromosome"
        # Add the csv title manually i cba
        #file.write(title + "\n")
        file.write(f"{N_found},{upstream + downstream},{chromosome}")
        file.write("\n")

    end = time.time()
    print("Created fasta file for gene: ", gene_id, " in time: ", end - start, "(s)")


def read_tss(tss_file):
    df = pd.read_csv(tss_file)
    for index, row in df.iterrows():
        gene_id = row["Gene ID"]
        chromosome = row["Chromosome"]
        tss = row["TSS"]
        reverse = False
        if "strand" in df.columns:
            if row["strand"] == '+':
                reverse = False
            elif row["strand"] == '-':
                reverse = True
            else:
                panic("No valid strand type found!")
        else:
            print("csv file does not support strand data!")
        save_fasta_file(gene_id, chromosome, tss, reverse)


if __name__ == '__main__':
    read_tss("../data/pig/pig_tss_CORRECTED.csv")
