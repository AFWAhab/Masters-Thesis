# prep_data_chunked_debug.py -----------------------------------------------
"""
Same as prep_data_chunked.py, but with print()s to trace progress and catch
where anchors/edges might be disappearing, plus optimized sequence lookup
and progress reporting in step 6.
"""
import os
import sys
import re
import json
import time

import h5py
from collections import defaultdict

import numpy as np
import polars as pl
import cooler
import pyfastx
import subprocess

# --------------------- CONFIG ---------------------
#ROOT = r"C:\Users\gusta\PycharmProjects\ThesisPlayground\training\zebrafish_training"
#GTF_PATH = os.path.join(ROOT, "hi-c-setup", "Homo_sapiens.GRCh38.90.gtf")
#MCOOL_PATH = os.path.join(ROOT, "hi-c-setup", "4DNFITRVKRPA.mcool")
#PEAK_PATH = os.path.join(ROOT, "hi-c-setup", "ENCFF367KIF.bed")
#FASTA_PATH = os.path.join(ROOT, "extra_data", "hg38.fa")
#GTF_PATH = os.path.join("../data/gtfFiles/Danio_rerio.GRCz10.82.gtf")
GTF_PATH = os.path.join("../data/Danio_rerio.GRCz11.113.gtf")
#GTF_PATH = os.path.join("../data/Danio_rerio.GRCz11.113.gtf")
MCOOL_PATH = os.path.join("cairns/4DNFI8H9ECLI_processed.mcool")
PEAK_PATH = os.path.join("Bogdanovic/GSM803833_H3K27ac_48hpf_peaks.bed")
#FASTA_PATH = os.path.join("danRer10.fa")
FASTA_PATH = os.path.join("danRer11.fa.gz")
ZEBRAFISH_H5_PATH = os.path.join("../training/zebrafish.hdf5")

OUT_DIR = "gnn_setup"

for p in (GTF_PATH, MCOOL_PATH, PEAK_PATH, FASTA_PATH):
    if not os.path.exists(p):
        sys.exit(f"✗ File not found: {p}")

RES = 100_000
MAX_DIST = 1_000_000
OBS_EXP_TH = 2.0
PROM_WIN = 1_000

# ------------------- MAIN ----------------------------
def main():
    # ─── 1) Load & filter promoters ───────────────────────────────────────────
    rows = []
    for ln in open(GTF_PATH):
        if ln.startswith("#"):
            continue
        f = ln.rstrip().split("\t")
        if f[2] != "gene":
            continue
        chrom, start, end, strand = f[0], int(f[3]), int(f[4]), f[6]
        gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
        tss = start if strand == "+" else end
        rows.append((gid, chrom, tss, strand))
    gtf = pl.DataFrame(rows, schema=["gene_id", "chrom", "tss", "strand"], orient="row")
    gtf = gtf.with_columns(
        pl.when(pl.col("chrom").str.starts_with("chr"))
        .then(pl.col("chrom"))
        .otherwise(pl.lit("chr") + pl.col("chrom"))
        .alias("chrom")
    )

    xp_genes = []
    #for split in ("zebrafish_train", "zebrafish_val", "zebrafish_test"):
        #fn = os.path.join(ROOT, f"{split}.hdf5")
    with h5py.File(ZEBRAFISH_H5_PATH, "r") as h5:
        xp_genes += [g.decode() for g in h5["geneName"][:]]
    prom = gtf.join(pl.DataFrame(xp_genes, schema=["gene_id"]), on="gene_id")

    with open("genes_aligned.txt") as f:
        aligned_genes = set(line.strip() for line in f)
    prom = prom.filter(pl.col("gene_id").is_in(aligned_genes))
    print(f"[1] promoters: {len(prom)}")

    # ─── 2) Assign promoter bins + build bin→node map ─────────────────────────
    clr = cooler.Cooler(f"{MCOOL_PATH}::/resolutions/{RES}")
    bins = (
        pl.DataFrame(clr.bins()[:])
        .with_row_index("bin_id")
        .with_columns([
            pl.col("chrom").cast(pl.Utf8).alias("chrom"),
            (pl.col("start") // RES).alias("bin_idx")
        ])
        .select(["bin_id", "chrom", "start", "end", "bin_idx"])
    )
    prom = prom.with_columns((pl.col("tss") // RES).alias("bin_idx"))
    prom = prom.join(bins.select(["chrom", "bin_idx", "bin_id"]), on=["chrom", "bin_idx"])
    node_id_column = pl.Series("node_id", np.arange(len(prom)))
    prom = prom.with_columns(node_id_column)
    prom.write_parquet(os.path.join(OUT_DIR, "gnn_nodes_H3K27ac_100k.parquet"))
    print(f"[2] wrote {len(prom)} promoter nodes")

    # build a dict for fast lookup:
    bin_to_node = dict(zip(prom["bin_id"].to_list(),
                           prom["node_id"].to_list()))
    prom_bins = set(bin_to_node.keys())

    # ─── 3) Load anchors ───────────────────────────────────────────────────────
    peaks = pl.read_csv(
        PEAK_PATH,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name", "score", "strand", "signal", "p", "q", "summit"]
        [:len(open(PEAK_PATH).readline().split())]
    ).with_columns([
        pl.when(pl.col("chrom").str.starts_with("chr"))
        .then(pl.col("chrom"))
        .otherwise(pl.lit("chr") + pl.col("chrom"))
        .alias("chrom"),
        (pl.col("start") // RES).alias("bin_idx")
    ])
    anchor_tbl = (
        peaks.join(bins.select(["chrom", "bin_idx", "bin_id"]), on=["chrom", "bin_idx"])
        .select(["chrom", "bin_idx", "bin_id"])
        .unique()
    )
    print(f"[3] anchor bins total = {len(anchor_tbl)}")

    anchor_map = {gi: idx + len(prom)
                  for idx, gi in enumerate(anchor_tbl["bin_id"])}
    anchor_bins = set(anchor_map.keys())

    # ─── 4) Compute expected cis up to MAX_DIST ───────────────────────────────
    exp = {}
    maxd = MAX_DIST // RES
    sel = clr.matrix(balance=False, sparse=True)
    for chrom in clr.chromnames:
        S = sel.fetch(chrom).tocsr()
        rows, cols = S.nonzero()
        d = cols - rows
        mask = (d >= 0) & (d <= maxd)
        sums = np.bincount(d[mask], weights=S.data[mask], minlength=maxd + 1)
        counts = np.arange(S.shape[0], S.shape[0] - maxd - 1, -1, dtype=float)
        for i in range(maxd + 1):
            exp[(chrom, i * RES)] = sums[i] / counts[i]
        del S
    print(f"[4] computed expected for ±{maxd} bins")

    # ─── 5) Stream pixels & build edges ────────────────────────────────────────
    edges = {"src": [], "tgt": [], "w": []}
    chroms = set(anchor_tbl["chrom"].to_list())
    global2local = {b_id: b_idx for b_id, *_, b_idx in bins.iter_rows()}

    for chrom in sorted(chroms):
        pix = pl.DataFrame(
            clr.matrix(balance=False, as_pixels=True).fetch(chrom)
        ).select(["bin1_id", "bin2_id", "count"])
        pm = (pix
              .filter(pl.col("bin1_id").is_in(pl.Series(list(prom_bins))))
              .filter(pl.col("bin2_id").is_in(pl.Series(list(anchor_bins))))
              )
        print(f"[5:{chrom}] raw prom→anchor pixels = {len(pm)}")
        for b1, b2, cnt in pm.iter_rows():
            dist = abs(global2local[b2] - global2local[b1]) * RES
            e = exp.get((chrom, dist), np.nan)
            if np.isnan(e) or cnt / e <= OBS_EXP_TH:
                continue
            w = np.log2(cnt / e)
            p = bin_to_node[b1]
            a = anchor_map[b2]
            edges["src"] += [p, a]
            edges["tgt"] += [a, p]
            edges["w"] += [w, w]
        print(f"[5:{chrom}] kept {len(edges['src']) // 2} edges so far")

    pl.DataFrame(edges).write_parquet(os.path.join(OUT_DIR, "edges_H3K27ac_100k.parquet"))
    print(f"[5] wrote {len(edges['src']) // 2} edge pairs")

    # ─── 6) Extract anchor sequences (fast, with progress) ───────────────────
    fasta = pyfastx.Fasta(FASTA_PATH)
    # pre-build global→(chrom,start) lookup
    binid_to_coord = {
        row["bin_id"]: (row["chrom"], row["start"])
        for row in bins.iter_rows(named=True)
    }
    total_anchors = len(anchor_map)
    print(f"[6] starting sequence extraction for {total_anchors} anchors")

    # start timer
    t_start = time.time()
    first_k_reported = False

    seq_rows = []
    for i, (gi, node) in enumerate(anchor_map.items(), start=1):
        # print on the first iteration and every 1000
        if i == 1 or i % 1000 == 0:
            print(f"[6] processed {i}/{total_anchors} anchors", flush=True)

        # after exactly 1000 fetches, report elapsed time
        if not first_k_reported and i == 1000:
            dt = time.time() - t_start
            print(f"[6] 🕒 first 1000 fetches took {dt:.1f}s", flush=True)
            first_k_reported = True

        chrom, start = binid_to_coord[gi]
        mid = start + RES // 2
        s0, s1 = max(1, mid - PROM_WIN // 2), mid + PROM_WIN // 2
        seq = fasta.fetch(chrom, (s0, s1))
        seq_rows.append((node, chrom, start, seq))

    pl.DataFrame(seq_rows, schema=["node_id", "chrom", "start", "seq"]) \
        .write_parquet(os.path.join(OUT_DIR, "nodes_anchor_H3K27ac_100k.parquet"))
    print(f"[6] wrote {len(seq_rows)} anchor sequences")

    summary = {
        "promoter_nodes": len(prom),
        "anchor_nodes": len(anchor_map),
        "edges": len(edges["src"]) // 2,
        "mean_degree": round((len(edges["src"]) // 2) / len(prom), 2)
    }
    print("✅ Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
