#!/usr/bin/env python3
# cnn_expr_dualbert_v5_fixed.py  – dtype-safe + lower batch (32)
# ==============================================================
# Only the lines marked  ### NEW / CHANGED  differ from the previous script.

import io, os, ast, random, pickle, h5py, csv
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from biomart import BiomartServer

# ───────────── configuration ─────────────
DATA_ROOT = "D:/PythonProjects/Speciale"
EMB2_FILE = os.path.join(DATA_ROOT, "dnabert2_embeddings.h5")
EMB1_FILE = os.path.join(DATA_ROOT, "adam_experiments/dnabert1_embeddings_k6.h5")
XPRESSO_DIR = "xpresso/data/pM10Kb_1KTest"
TF_EXCEL  = "DeepLncLoc/varie/transcription_factor2.xlsx"
PROM_TSV  = "beating_seq2exp/data/processed/promoter_extra_feats.tsv"
CACHE_MAP = "gene_symbol_to_ensembl.pkl"

RUNS           = 10
MAX_EPOCHS     = 40
BATCH          = 32                      ### CHANGED (was 40)
EARLY_PATIENCE = 5
SWA_START      = 8
LR_MAIN, LR_PROJ = 8e-4, 1e-4
WD, DROPOUT    = 3e-5, 0.30
SEED           = 2025
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ───────────────────── Helper: gene-symbol ↔ ENSG ────────────────
def load_symbol2ensg():
    if os.path.exists(CACHE_MAP):
        return pd.read_pickle(CACHE_MAP)
    srv = BiomartServer("https://www.ensembl.org/biomart")
    ds  = srv.datasets["hsapiens_gene_ensembl"]
    txt = ds.search({"attributes": ["external_gene_name", "ensembl_gene_id"]}).text
    df  = pd.read_csv(io.StringIO(txt), sep="\t", names=["sym", "ensg"])
    mp  = dict(zip(df.sym, df.ensg)); pd.to_pickle(mp, CACHE_MAP); return mp

SYM2ENSG = load_symbol2ensg()
def as_ensg(x: str):
    x = str(x)
    return x.split(".")[0] if x.startswith("ENSG") else SYM2ENSG.get(x)

# ───────────────────── IO helpers ────────────────────────────────
def load_xpresso(path):
    with h5py.File(path) as f:
        y = f["label"][:].astype(np.float32)
        m = f["data"][:, [0, 1, 2, 3, 4, 5, 7]].astype(np.float32)  # 7 half-life feats
        g = [as_ensg(u.decode()) for u in f["geneName"][:]]
    return y, m, g

def load_tf_dict(xlsx):
    df = pd.read_excel(xlsx, header=None, names=["idx", "vec", "gene"])
    out = {}
    for g, v in zip(df.gene, df.vec):
        e = as_ensg(g)
        if e:
            try:
                out[e] = np.asarray(ast.literal_eval(v), dtype=np.float32)
            except Exception:
                pass
    return out

def load_cpg_dict(tsv):
    df = pd.read_csv(tsv, sep="\t", comment="#")
    df = df.rename(columns=str.lower).rename(
        columns={"genename": "gene",
                 "cpg_coverage": "cpg_frac",
                 "cpg_fraction": "cpg_frac",
                 "cpg": "cpg_frac"}
    )
    out = {}
    for _, r in df[["gene", "cpg_frac"]].iterrows():
        e = as_ensg(r.gene)
        if e: out[e] = np.array([r.cpg_frac], np.float32)
    return out

def merge_meta(base, genes, *dicts):
    mats = [base]
    for d in dicts:
        k = len(next(iter(d.values())))
        buf = np.zeros((len(genes), k), np.float32)
        for i, g in enumerate(genes):
            v = d.get(g)
            if v is not None: buf[i] = v
        mats.append(buf)
    return np.hstack(mats)

# ───────────────────── Dataset ───────────────────────────────────
class GeneDS(Dataset):
    def __init__(self, e, m, y):
        self.e = torch.tensor(e, dtype=torch.float32)  ### NEW
        self.m = torch.tensor(m, dtype=torch.float32)  ### NEW
        self.y = torch.tensor(y, dtype=torch.float32)[:, None]
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.e[i], self.m[i], self.y[i]

# ───────────────────── Model: larger MK-CNN ──────────────────────
class AttentionPool(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x):
        q = self.q.expand(x.size(0), -1, -1)
        return self.attn(q, x, x, need_weights=False)[0].squeeze(1)

class MKBlock(nn.Module):
    def __init__(self, in_c, mid, out, k, drop):
        super().__init__()
        pad = k // 2
        self.seq = nn.Sequential(
            nn.Conv1d(in_c, mid, k, padding=pad),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid, out, k, padding=pad*2, dilation=2),
            nn.BatchNorm1d(out), nn.GELU(),
            nn.Dropout(drop)
        )
    def forward(self, x): return self.seq(x)

class MKCNN(nn.Module):
    PROJ = 192; NF = 96; KS = (5, 9, 13)
    def __init__(self, drop):
        super().__init__()
        self.proj  = nn.Linear(768, self.PROJ)
        self.paths = nn.ModuleList([MKBlock(self.PROJ, self.NF, self.NF, k, drop) for k in self.KS])
        self.pool  = AttentionPool(self.NF * len(self.KS))
    def forward(self, x):               # x:(B,36,768)
        x = self.proj(x).transpose(1, 2)       # (B,192,36)
        x = torch.cat([p(x) for p in self.paths], 1).transpose(1, 2)  # (B,36,288)
        return self.pool(x)                    # (B,288)

class Predictor(nn.Module):
    def __init__(self, meta_dim, drop=0.30):
        super().__init__()
        self.emb  = MKCNN(drop)
        self.meta = nn.Sequential(nn.Linear(meta_dim, 64),
                                  nn.BatchNorm1d(64), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(288 + 64, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(drop),
            nn.Linear(128, 1)
        )
    def forward(self, e, m):
        return self.head(torch.cat([self.emb(e), self.meta(m)], 1))

# ───────────────────── Train / Eval helpers ──────────────────────
def train_epoch(model, dl, crit, opt):
    model.train(); tot = 0.
    for e, m, y in dl:
        e, m, y = [t.to(DEVICE) for t in (e, m, y)]
        opt.zero_grad(); loss = crit(model(e, m), y); loss.backward(); opt.step()
        tot += loss.item() * len(e)
    return tot / len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, crit):
    model.eval(); tot = 0.; P, T = [], []
    for e, m, y in dl:
        e, m, y = [t.to(DEVICE) for t in (e, m, y)]
        out = model(e, m); tot += crit(out, y).item() * len(e)
        P.append(out.cpu()); T.append(y.cpu())
    return tot / len(dl.dataset), torch.cat(P), torch.cat(T)

@torch.no_grad()
def update_bn(loader, model):
    model.train()
    for e, m, _ in loader:
        model(e.to(DEVICE), m.to(DEVICE))

if __name__ == "__main__":
    # ───────────────────── Load dual embeddings once ─────────────────
    with h5py.File(EMB2_FILE) as f2:
        Etr2 = f2["train_X_embeddings"][:].astype(np.float32)  ### NEW
        Eva2 = f2["valid_X_embeddings"][:].astype(np.float32)  ### NEW
        Ete2 = f2["test_X_embeddings"][:].astype(np.float32)  ### NEW

    if os.path.exists(EMB1_FILE):  # dual-BERT
        with h5py.File(EMB1_FILE) as f1:
            Etr = np.concatenate(
                [Etr2, f1["train_X_embeddings"][:].astype(np.float32)], 1)  ### NEW
            Eva = np.concatenate(
                [Eva2, f1["valid_X_embeddings"][:].astype(np.float32)], 1)  ### NEW
            Ete = np.concatenate(
                [Ete2, f1["test_X_embeddings"][:].astype(np.float32)], 1)  ### NEW
    else:
        Etr, Eva, Ete = Etr2, Eva2, Ete2

    # ───────────────────── Metadata tables ───────────────────────────
    xp = os.path.join(DATA_ROOT, XPRESSO_DIR)
    Ytr, Mtr, Gtr = load_xpresso(os.path.join(xp, "train.h5"))
    Yva, Mva, Gva = load_xpresso(os.path.join(xp, "valid.h5"))
    Yte, Mte, Gte = load_xpresso(os.path.join(xp, "test.h5"))

    TF = load_tf_dict(os.path.join(DATA_ROOT, TF_EXCEL))
    CP = load_cpg_dict(os.path.join(DATA_ROOT, PROM_TSV))
    Mtr = merge_meta(Mtr, Gtr, TF, CP)
    Mva = merge_meta(Mva, Gva, TF, CP)
    Mte = merge_meta(Mte, Gte, TF, CP)

    META_DIM = Mtr.shape[1]
    scaler   = StandardScaler().fit(np.vstack([Mtr, Mva]))
    Mtr, Mva, Mte = map(scaler.transform, (Mtr, Mva, Mte))

    tr_dl = DataLoader(GeneDS(Etr, Mtr, Ytr), BATCH, shuffle=True)
    va_dl = DataLoader(GeneDS(Eva, Mva, Yva), BATCH)
    te_dl = DataLoader(GeneDS(Ete, Mte, Yte), BATCH)

    # ───────────────────── Training loop ─────────────────────────────
    crit = nn.MSELoss(); all_scores = []
    for run in range(1, RUNS + 1):
        print(f"\n── Run {run}/{RUNS} ─────────────────────────")
        model = Predictor(META_DIM, DROPOUT).to(DEVICE)

        proj, rest = [], []
        for n, p in model.named_parameters():
            (proj if "emb.proj" in n else rest).append(p)
        opt = optim.Adam([{"params": proj, "lr": LR_PROJ},
                          {"params": rest, "lr": LR_MAIN}], weight_decay=WD)
        cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=4, T_mult=2)

        swa = AveragedModel(model); swa_on = False
        best, best_state, no_improve = float("inf"), None, 0

        for ep in range(1, MAX_EPOCHS + 1):
            tr = train_epoch(model, tr_dl, crit, opt)
            vl, Pv, Tv = evaluate(model, va_dl, crit)
            r2 = linregress(Pv.numpy().ravel(), Tv.numpy().ravel()).rvalue ** 2
            print(f"Ep{ep:02d}  tr {tr:.3f}  val {vl:.3f}  R² {r2:.3f}")
            cosine.step(ep)

            if vl < best:
                best, best_state, no_improve = vl, model.state_dict(), 0
            else:
                no_improve += 1

            if ep >= SWA_START:
                if not swa_on:
                    print(">>> SWA ON"); swa_on = True
                swa.update_parameters(model)

            if no_improve >= EARLY_PATIENCE:
                print("↯ early stop"); break

        final = swa if swa_on else model
        if swa_on:
            update_bn(tr_dl, swa)
        else:
            final.load_state_dict(best_state)

        _, Pt, Tt = evaluate(final, te_dl, crit)
        test_r2 = linregress(Pt.numpy().ravel(), Tt.numpy().ravel()).rvalue ** 2
        all_scores.append(test_r2)
        print(f"▶ Test R² {test_r2:.4f}")

    print("\n==== Summary ====")
    print("Runs:", [f"{x:.4f}" for x in all_scores])
    print(f"Best {max(all_scores):.4f}  Mean {np.mean(all_scores):.4f}")
