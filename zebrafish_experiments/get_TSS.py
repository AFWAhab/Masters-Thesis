import csv
from collections import defaultdict
import pandas as pd
from util.gtf2df import gtf2df


def get_orf_length(cds_rows):
    return sum(int(cds_row["end"]) - int(cds_row["start"]) + 1 for idx, cds_row in cds_rows.iterrows())


def get_utr_lengths(five_prime_utrs, three_prime_utrs):
    return (
        sum(int(utr["end"]) - int(utr["start"]) + 1 for idx, utr in five_prime_utrs.iterrows()),
        sum(int(utr["end"]) - int(utr["start"]) + 1 for idx, utr in three_prime_utrs.iterrows())
    )


def get_tss(exon_rows, strand):
    if strand == "+":
        exons = sorted([row for idx, row in exon_rows.iterrows()],
                       key=lambda x: int(x["start"]))
        return int(exons[0]["start"])
    else:
        exons = sorted([row for idx, row in exon_rows.iterrows()],
                       key=lambda x: int(x["end"]), reverse=True)
        return int(exons[0]["end"])


def process_gtf(gtf_file, output_csv, ids):
    df = gtf2df(gtf_file)
    genes = defaultdict(list)

    # get biased genes
    #df_bias = pd.read_csv("data/removedGenes/bias_genes.txt", sep=",")
    #df_bias_ids = df_bias["gene_id"]

    counter = 0
    for id in ids:
        counter += 1
        if counter % 100 == 0:
            print("at counter", counter)

        #if id in df_bias_ids:
        #    continue

        gene_rows = df[df["gene_id"] == id]
        for idx, row in gene_rows.iterrows():
            if row["feature"] != "transcript":
                continue

            transcript_id = str(row["transcript_id"])

            start_codon = gene_rows[(gene_rows["transcript_id"] == transcript_id) & (gene_rows["feature"] == "start_codon")]
            stop_codon = gene_rows[(gene_rows["transcript_id"] == transcript_id) & (gene_rows["feature"] == "stop_codon")]
            if start_codon.empty or stop_codon.empty:
                continue

            exons = gene_rows[(gene_rows["transcript_id"] == transcript_id) & (gene_rows["feature"] == "exon")]
            transcript_cds = gene_rows[(gene_rows["transcript_id"] == transcript_id) & (gene_rows["feature"] == "cds")]
            five_prime_utrs = gene_rows[(gene_rows["transcript_id"] == transcript_id) & (gene_rows["feature"] == "five_prime_utr")]
            three_prime_utrs = gene_rows[(gene_rows["transcript_id"] == transcript_id) & (gene_rows["feature"] == "three_prime_utr")]
            if len(exons) > 0:
                genes[id].append((row, exons, transcript_cds, five_prime_utrs, three_prime_utrs))

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Gene ID", "Representative Transcript ID", "Chromosome", "TSS", "strand"])

        for gene_id, transcript_data in genes.items():
            best_transcript = None
            best_exons = None
            best_orf = 0
            best_five_utr = 0
            best_three_utr = 0

            for transcript, exons, cds, five_prime_utrs, three_prime_utrs in transcript_data:
                orf_length = get_orf_length(cds)
                five_utr, three_utr = get_utr_lengths(five_prime_utrs, three_prime_utrs)

                if (orf_length > best_orf or
                        (orf_length == best_orf and five_utr > best_five_utr) or
                        (orf_length == best_orf and five_utr == best_five_utr and three_utr > best_three_utr)):
                    best_transcript = transcript
                    best_exons = exons
                    best_orf = orf_length
                    best_five_utr = five_utr
                    best_three_utr = three_utr

            if best_transcript is not None:
                strand = best_exons.iloc[0]["strand"]
                tss = get_tss(best_exons, strand)
                if tss is not None:
                    writer.writerow([gene_id, best_transcript["transcript_id"], best_transcript["seqname"], tss, strand])

    print("done!")

# get list of orthologs
#df_orthologs = pd.read_csv("data/human_zebrafish_orthologs.csv", sep=",")
#df_ortholog_ids = df_orthologs["ensembl_id"]

#process_gtf("data/Danio_rerio.GRCz11.113.gtf", "data/zebrafish_tss.csv", df_ortholog_ids[0:1500])
#print("done!")

df_pig_gene_ids = pd.read_csv("data/all_values_pig.txt", sep=",")["Gene ID"]
process_gtf("data/Sus_scrofa.Sscrofa11.1.113.gtf", "data/pig_tss_CORRECTED.csv", df_pig_gene_ids[:])
#df_zebrafish_gene_ids = pd.read_csv("data/all_values.txt", sep=",")["Gene ID"]
#process_gtf("data/Danio_rerio.GRCz11.113.gtf", "data/zebrafish_tss_CORRECTED.csv", df_zebrafish_gene_ids[:])