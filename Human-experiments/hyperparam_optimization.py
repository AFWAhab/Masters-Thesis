#!/usr/bin/env python3
# gene_expr_optuna_dualbert.py
# ================================================================
# Hyper-parameter optimisation for the dual-DNABERT MK-CNN predictor
# ---------------------------------------------------------------

import os, io, ast, random, datetime, time, h5py
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.swa_utils import AveragedModel          # ← only this
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import optuna
from biomart import BiomartServer

# ───────────── user knobs ────────────────────────────────────────
N_TRIALS        = 30
RUNS_PER_TRIAL  = 1
MAX_EPOCHS      = 40
EARLY_PATIENCE  = 5
SEED_BASE       = 2025

# ───────────── fixed paths ──────────────────────────────────────
DATA_ROOT   = "D:/PythonProjects/Speciale"
EMB2_FILE   = os.path.join(DATA_ROOT, "dnabert2_embeddings.h5")
EMB1_FILE   = os.path.join(DATA_ROOT, "adam_experiments/dnabert1_embeddings_k6.h5")
XPRESSO_DIR = "xpresso/data/pM10Kb_1KTest"
TF_EXCEL    = "DeepLncLoc/varie/transcription_factor2.xlsx"
PROM_TSV    = "beating_seq2exp/data/processed/promoter_extra_feats.tsv"
CACHE_MAP   = "gene_symbol_to_ensembl.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────── helper-functions ─────────────────────────────────
def load_symbol2ensg():
    if os.path.exists(CACHE_MAP):
        return pd.read_pickle(CACHE_MAP)
    srv = BiomartServer("https://www.ensembl.org/biomart")
    ds  = srv.datasets["hsapiens_gene_ensembl"]
    txt = ds.search({"attributes": ["external_gene_name","ensembl_gene_id"]}).text
    df  = pd.read_csv(io.StringIO(txt), sep="\t", names=["sym","ensg"])
    mp  = dict(zip(df.sym, df.ensg)); pd.to_pickle(mp, CACHE_MAP); return mp

SYM2ENSG = load_symbol2ensg()
def as_ensg(x: str):
    x = str(x)
    return x.split(".")[0] if x.startswith("ENSG") else SYM2ENSG.get(x)

def load_xpresso(path):
    with h5py.File(path) as f:
        y = f["label"][:].astype(np.float32)
        m = f["data"][:,[0,1,2,3,4,5,7]].astype(np.float32)
        g = [as_ensg(u.decode()) for u in f["geneName"][:]]
    return y, m, g

def load_tf_dict(xlsx):
    df = pd.read_excel(xlsx, header=None, names=["idx","vec","gene"])
    out = {}
    for g, v in zip(df.gene, df.vec):
        e = as_ensg(g)
        if e:
            try:
                out[e] = np.asarray(ast.literal_eval(v), np.float32)
            except Exception:
                pass
    return out

def load_cpg_dict(tsv):
    df = pd.read_csv(tsv, sep="\t", comment="#").rename(columns=str.lower)
    df = df.rename(columns={"genename":"gene",
                            "cpg_coverage":"cpg_frac",
                            "cpg_fraction":"cpg_frac",
                            "cpg":"cpg_frac"})
    out = {}
    for _, r in df[["gene","cpg_frac"]].iterrows():
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

# ───────────── Dataset & model classes ──────────────────────────
class GeneDS(Dataset):
    def __init__(self, e, m, y):
        self.e = torch.tensor(e, dtype=torch.float32)
        self.m = torch.tensor(m, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)[:,None]
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.e[i], self.m[i], self.y[i]

class AttentionPool(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__(); self.q = nn.Parameter(torch.randn(1,1,dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x):
        q = self.q.expand(x.size(0), -1, -1)
        return self.attn(q, x, x, need_weights=False)[0].squeeze(1)

class MKBlock(nn.Module):
    def __init__(self, in_c, mid, out, k, drop):
        super().__init__()
        pad = k//2
        self.seq = nn.Sequential(
            nn.Conv1d(in_c, mid, k, padding=pad),
            nn.BatchNorm1d(mid), nn.GELU(),
            nn.Conv1d(mid, out, k, padding=pad*2, dilation=2),
            nn.BatchNorm1d(out), nn.GELU(),
            nn.Dropout(drop))
    def forward(self, x): return self.seq(x)

class MKCNN(nn.Module):
    def __init__(self, proj_dim, nf, ks, drop):
        super().__init__()
        self.proj  = nn.Linear(768, proj_dim)
        self.paths = nn.ModuleList([MKBlock(proj_dim,nf,nf,k,drop) for k in ks])
        self.pool  = AttentionPool(nf*len(ks))
    def forward(self, x):
        x = self.proj(x).transpose(1,2)
        x = torch.cat([p(x) for p in self.paths], 1).transpose(1,2)
        return self.pool(x)

class Predictor(nn.Module):
    def __init__(self, meta_dim, proj_dim=192, nf=96, ks=(5,9,13), drop=0.30):
        super().__init__()
        self.emb  = MKCNN(proj_dim,nf,ks,drop)
        emb_out   = nf*len(ks)
        self.meta = nn.Sequential(nn.Linear(meta_dim,64),
                                  nn.BatchNorm1d(64), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(emb_out+64,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256,128),       nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(drop),
            nn.Linear(128,1))
    def forward(self, e, m):
        return self.head(torch.cat([self.emb(e), self.meta(m)], 1))

# ───────────── load data once ───────────────────────────────────
print("⌛ Loading embeddings …")
with h5py.File(EMB2_FILE) as f2:
    Etr2 = f2["train_X_embeddings"][:].astype(np.float32)
    Eva2 = f2["valid_X_embeddings"][:].astype(np.float32)
    Ete2 = f2["test_X_embeddings"][:].astype(np.float32)

if os.path.exists(EMB1_FILE):
    with h5py.File(EMB1_FILE) as f1:
        Etr = np.concatenate([Etr2, f1["train_X_embeddings"][:].astype(np.float32)],1)
        Eva = np.concatenate([Eva2, f1["valid_X_embeddings"][:].astype(np.float32)],1)
        Ete = np.concatenate([Ete2, f1["test_X_embeddings"][:].astype(np.float32)],1)
else:
    Etr, Eva, Ete = Etr2, Eva2, Ete2

print("⌛ Loading metadata …")
xp = os.path.join(DATA_ROOT, XPRESSO_DIR)
Ytr, Mtr, Gtr = load_xpresso(os.path.join(xp,"train.h5"))
Yva, Mva, Gva = load_xpresso(os.path.join(xp,"valid.h5"))
Yte, Mte, Gte = load_xpresso(os.path.join(xp,"test.h5"))

TF = load_tf_dict(os.path.join(DATA_ROOT, TF_EXCEL))
CP = load_cpg_dict(os.path.join(DATA_ROOT, PROM_TSV))
Mtr = merge_meta(Mtr,Gtr,TF,CP)
Mva = merge_meta(Mva,Gva,TF,CP)
Mte = merge_meta(Mte,Gte,TF,CP)

META_DIM = Mtr.shape[1]
scaler   = StandardScaler().fit(np.vstack([Mtr,Mva]))
Mtr, Mva, Mte = map(scaler.transform,(Mtr,Mva,Mte))

# ───────────── utils ────────────────────────────────────────────
def r2_np(pred, true):
    return linregress(pred.ravel(), true.ravel()).rvalue**2

@torch.no_grad()
def evaluate(model, dl, crit):
    model.eval(); tot=0.; P,T=[],[]
    for e,m,y in dl:
        e,m,y=[t.to(DEVICE) for t in (e,m,y)]
        out=model(e,m); tot+=crit(out,y).item()*len(e)
        P.append(out.cpu()); T.append(y.cpu())
    return tot/len(dl.dataset), torch.cat(P).numpy(), torch.cat(T).numpy()

@torch.no_grad()
def update_bn(loader, model):                 # ← custom BN refresh
    model.train()
    for e,m,_ in loader:
        model(e.to(DEVICE), m.to(DEVICE))

# ───────────── Optuna objective ─────────────────────────────────
def objective(trial: optuna.Trial)->float:
    start=time.perf_counter()

    batch_size = trial.suggest_categorical("batch",[16,32,48])
    dropout    = trial.suggest_float("dropout",0.15,0.35,step=0.05)
    lr_main    = trial.suggest_float("lr_main",3e-4,1e-3,log=True)
    lr_proj    = trial.suggest_float("lr_proj",3e-5,3e-4,log=True)
    swa_start  = trial.suggest_int("swa_start",6,12)
    proj_dim   = trial.suggest_categorical("proj_dim",[128,192,256])
    nf         = trial.suggest_categorical("nf",[64,96,128])
    ks         = trial.suggest_categorical("kernels",[(5,9,13),(3,7,11)])

    r2_scores=[]
    for run in range(RUNS_PER_TRIAL):
        seed = SEED_BASE + trial.number*10 + run
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        tr_dl = DataLoader(GeneDS(Etr,Mtr,Ytr),batch_size,shuffle=True)
        va_dl = DataLoader(GeneDS(Eva,Mva,Yva),batch_size)
        te_dl = DataLoader(GeneDS(Ete,Mte,Yte),batch_size)

        model=Predictor(META_DIM,proj_dim,nf,ks,dropout).to(DEVICE)

        proj,rest=[],[]
        for n,p in model.named_parameters():
            (proj if "emb.proj" in n else rest).append(p)

        opt = optim.Adam([{"params":proj,"lr":lr_proj},
                          {"params":rest,"lr":lr_main}], weight_decay=3e-5)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=4,T_mult=2)
        crit = nn.MSELoss()

        best_val, best_state, no_improve, swa_on = 1e9, None,0,False
        swa_model = AveragedModel(model)

        for ep in range(1,MAX_EPOCHS+1):
            model.train()
            for e,m,y in tr_dl:
                e,m,y=[t.to(DEVICE) for t in (e,m,y)]
                opt.zero_grad(); crit(model(e,m),y).backward(); opt.step()
            sched.step(ep)

            val_loss,Pv,Tv = evaluate(model,va_dl,crit)
            trial.report(r2_np(Pv,Tv), ep)
            if trial.should_prune(): raise optuna.TrialPruned()

            if val_loss < best_val:
                best_val, best_state, no_improve = val_loss, model.state_dict(),0
            else:
                no_improve += 1

            if ep >= swa_start:
                swa_on=True
                swa_model.update_parameters(model)

            if no_improve>=EARLY_PATIENCE: break

        final = swa_model if swa_on else model
        if swa_on:
            final.load_state_dict(swa_model.state_dict())
            update_bn(tr_dl, final)            # ← our function
        else:
            final.load_state_dict(best_state)

        _,Pt,Tt = evaluate(final,te_dl,crit)
        r2_scores.append(r2_np(Pt,Tt))

    score=float(np.mean(r2_scores))
    trial.set_user_attr("duration_sec", time.perf_counter()-start)
    print(f"Trial {trial.number+1:02d}/{N_TRIALS}  R²={score:.4f}")
    return score

# ───────────── run study ────────────────────────────────────────
def main():
    tag=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output",exist_ok=True)
    trials_csv=f"output/optuna_trials_{tag}.csv"
    best_csv  =f"output/optuna_best_{tag}.csv"

    study=optuna.create_study(direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED_BASE),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=8))

    print(f"\n=== Optimising {N_TRIALS} trials ===\n")
    study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=True)

    study.trials_dataframe(attrs=("number","value","params","user_attrs")
                           ).to_csv(trials_csv,index=False)
    best=study.best_trial
    pd.DataFrame([{**best.params,"test_r2":best.value,
                   "duration_sec":best.user_attrs["duration_sec"]}]
                 ).to_csv(best_csv,index=False)

    print("\nFinished. Best R² {:.4f}".format(best.value))

if __name__=="__main__":
    main()
