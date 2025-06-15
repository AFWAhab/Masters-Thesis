import pandas as pd
from util.gtf2df import gtf2df

def process_gtf(gtf_file, csv_tss):
    df = gtf2df(gtf_file)
    df_tss = pd.read_csv(csv_tss, sep=",")
    df_transcript_ids = df_tss["Representative Transcript ID"]
    strands = []

    counter = 0
    for transcript_id in df_transcript_ids:
        counter += 1
        if counter % 100 == 0:
            print("at counter", counter)

        gene_rows = df[df["transcript_id"] == transcript_id]
        strand = gene_rows.iloc[0]["strand"]
        strands.append(strand)

    df_tss_w_transcript = df_tss.assign(strand=strands)
    df_tss_w_transcript.to_csv("data/zebrafish_tss_CORRECTED_w_strands.csv", sep="\t", index=False)
    print("done!")

process_gtf("data/Danio_rerio.GRCz11.113.gtf", "data/zebrafish_tss_CORRECTED.csv")