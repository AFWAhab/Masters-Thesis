#!/usr/bin/env python3
# cnn_expr_cpg_only_dual_nostop.py
# ───────────────────────────────────────────────────────────────────
# 10 runs on DNABERT-2 + 10 runs on DNABERT-1-k6, no early-stop.
# Logs:  output/cpg_only_dnabert2.csv , output/cpg_only_dnabert1.csv
# ───────────────────────────────────────────────────────────────────

import io, os, ast, random, csv, h5py, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from scipy.stats import linregress
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from biomart import BiomartServer

# ═══════════════════════ configuration ════════════════════════════
DATA_ROOT   = "D:/PythonProjects/Speciale"
EMB_FILES   = {
    "dnabert2" : "dnabert2_embeddings.h5",
}
XPRESSO_DIR = "xpresso/data/pM10Kb_1KTest"
TF_EXCEL    = "DeepLncLoc/varie/transcription_factor2.xlsx"
PROM_TSV    = "beating_seq2exp/data/processed/promoter_extra_feats.tsv"
CACHE_MAP   = "gene_symbol_to_ensembl.pkl"

RUNS_PER_MODEL, EPOCHS, BATCH = 10, 50, 64
LR, WD, DROPOUT = 1e-3, 0.0, 0.15
PATIENCE_LR = 5

SEED = 2025
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = os.path.join(DATA_ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ═════════════════════ helper functions (unchanged) ═══════════════
def load_symbol2ensg():
    if os.path.exists(CACHE_MAP):
        return pd.read_pickle(CACHE_MAP)
    srv = BiomartServer("https://www.ensembl.org/biomart")
    ds  = srv.datasets["hsapiens_gene_ensembl"]
    txt = ds.search({"attributes":["external_gene_name","ensembl_gene_id"]}).text
    df  = pd.read_csv(io.StringIO(txt), sep="\t", names=["sym","ensg"])
    mp  = dict(zip(df.sym, df.ensg)); pd.to_pickle(mp, CACHE_MAP); return mp

SYM2ENSG = load_symbol2ensg()
def as_ensg(x): x=str(x); return x.split(".")[0] if x.startswith("ENSG") else SYM2ENSG.get(x)

def load_xpresso(p):
    with h5py.File(p,"r") as f:
        y=f["label"][:].astype(np.float32)
        m=f["data"][:,[0,1,2,3,4,5,7]].astype(np.float32)
        g=[as_ensg(u.decode()) for u in f["geneName"][:]]
    return y,m,g

def load_tf_dict(xlsx):
    df=pd.read_excel(xlsx,header=None,names=["idx","vec","gene"])
    out={}
    for g,v in zip(df.gene,df.vec):
        e=as_ensg(g)
        if e:
            try: out[e]=np.asarray(ast.literal_eval(v),np.float32)
            except: pass
    return out

def load_cpg_dict(tsv):
    df=pd.read_csv(tsv,sep="\t",comment="#").rename(columns=str.lower)
    df=df.rename(columns={"genename":"gene","cpg_coverage":"cpg_frac",
                          "cpg_fraction":"cpg_frac","cpg":"cpg_frac"})
    if {"gene","cpg_frac"}-set(df.columns):
        raise ValueError(f"Missing columns in {tsv}: {df.columns}")
    out={};
    for _,r in df[["gene","cpg_frac"]].iterrows():
        e=as_ensg(r.gene);
        if e: out[e]=np.array([r.cpg_frac],np.float32)
    return out

def merge_meta(base,genes,*dicts):
    mats=[base]
    for d in dicts:
        k=len(next(iter(d.values())))
        buf=np.zeros((len(genes),k),np.float32)
        for i,g in enumerate(genes):
            v=d.get(g)
            if v is not None: buf[i]=v
        mats.append(buf)
    return np.hstack(mats)

# ═════════════════════ dataset & model ════════════════════════════
class GeneDS(Dataset):
    def __init__(self,e,m,y):
        self.e=torch.tensor(e,dtype=torch.float32)
        self.m=torch.tensor(m,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.e[i],self.m[i],self.y[i]

class Predictor(nn.Module):
    def __init__(self,emb_dim,meta_dim,hidden_emb=512,hidden_meta=32,h1=256,h2=128):
        super().__init__()
        self.emb=nn.Sequential(nn.Linear(emb_dim,hidden_emb),
                               nn.BatchNorm1d(hidden_emb),nn.ReLU(),
                               nn.Dropout(DROPOUT))
        self.meta=nn.Sequential(nn.Linear(meta_dim,hidden_meta),
                                nn.BatchNorm1d(hidden_meta),nn.ReLU())
        self.comb=nn.Sequential(
            nn.Linear(hidden_emb+hidden_meta,h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(h1,h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(h2,1))
    def forward(self,e,m): return self.comb(torch.cat((self.emb(e),self.meta(m)),1))

# ═════════════════════ train / eval helpers ═══════════════════════
def train_epoch(model,ldr,crit,opt):
    model.train(); tot=0.
    for e,m,y in ldr:
        e,m,y=e.to(DEVICE),m.to(DEVICE),y.to(DEVICE)
        opt.zero_grad(); loss=crit(model(e,m),y); loss.backward(); opt.step()
        tot+=loss.item()*len(e)
    return tot/len(ldr.dataset)

@torch.no_grad()
def evaluate(model,ldr,crit):
    model.eval(); tot=0.; P,T=[],[]
    for e,m,y in ldr:
        e,m,y=e.to(DEVICE),m.to(DEVICE),y.to(DEVICE)
        out=model(e,m); tot+=crit(out,y).item()*len(e)
        P.append(out.cpu()); T.append(y.cpu())
    return tot/len(ldr.dataset), torch.cat(P), torch.cat(T)

# ═════════════════════ loop over both embedding sets ══════════════
if __name__ == "__main__":
    for tag, emb_file in EMB_FILES.items():
        print(f"\n================  {tag.upper()}  =================")
        csv_path = os.path.join(OUT_DIR, f"cpg_only_{tag}.csv")
        with open(csv_path,"w",newline="") as f:
            csv.writer(f).writerow(["run","epoch","train_loss","val_loss","val_r2","test_r2"])

        # load embeddings ------------------------------------------------
        with h5py.File(os.path.join(DATA_ROOT, emb_file), "r") as f:
            Etr=f["train_X_embeddings"][:].reshape(-1,18*768)
            Eva=f["valid_X_embeddings"][:].reshape(-1,18*768)
            Ete=f["test_X_embeddings"] [:].reshape(-1,18*768)
        EMB_DIM=Etr.shape[1]

        # static data ----------------------------------------------------
        xp=os.path.join(DATA_ROOT,XPRESSO_DIR)
        Ytr,Mtr,Gtr=load_xpresso(os.path.join(xp,"train.h5"))
        Yva,Mva,Gva=load_xpresso(os.path.join(xp,"valid.h5"))
        Yte,Mte,Gte=load_xpresso(os.path.join(xp,"test.h5"))

        TF=load_tf_dict(os.path.join(DATA_ROOT,TF_EXCEL))
        CP=load_cpg_dict(os.path.join(DATA_ROOT,PROM_TSV))

        Mtr=merge_meta(Mtr,Gtr,TF,CP); Mva=merge_meta(Mva,Gva,TF,CP); Mte=merge_meta(Mte,Gte,TF,CP)
        META_DIM=Mtr.shape[1]

        scaler=StandardScaler().fit(np.vstack([np.hstack([Etr,Mtr]),np.hstack([Eva,Mva])]))
        Etr,Mtr=np.split(scaler.transform(np.hstack([Etr,Mtr])),[EMB_DIM],1)
        Eva,Mva=np.split(scaler.transform(np.hstack([Eva,Mva])),[EMB_DIM],1)
        Ete,Mte=np.split(scaler.transform(np.hstack([Ete,Mte])),[EMB_DIM],1)

        tr_dl=DataLoader(GeneDS(Etr,Mtr,Ytr),BATCH,shuffle=True)
        va_dl=DataLoader(GeneDS(Eva,Mva,Yva),BATCH)
        te_dl=DataLoader(GeneDS(Ete,Mte,Yte),BATCH)

        # training runs --------------------------------------------------
        crit=nn.MSELoss(); summary=[]
        for run in range(1, RUNS_PER_MODEL+1):
            print(f"\n-------- {tag}  |  RUN {run}/{RUNS_PER_MODEL} --------")
            model=Predictor(EMB_DIM,META_DIM).to(DEVICE)
            opt=optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
            sched=optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",patience=PATIENCE_LR,factor=0.5)

            best_val=float("inf"); best_state=None
            for ep in range(1,EPOCHS+1):                       # ← always full 50
                tr_loss=train_epoch(model,tr_dl,crit,opt)
                val_loss, Pval, Tval = evaluate(model, va_dl, crit)
                r_val = linregress(Pval.numpy().flatten(), Tval.numpy().flatten()).rvalue
                val_r2 = r_val ** 2
                sched.step(val_loss)

                print(f"  Ep{ep:02d} | Train {tr_loss:.4f} | Val {val_loss:.4f} | R² {val_r2:.4f}")
                with open(csv_path,"a",newline="") as f:
                    csv.writer(f).writerow([run,ep,tr_loss,val_loss,val_r2,None])

                if val_loss<best_val:
                    best_val,best_state=val_loss,model.state_dict()

            # test -------------------------------------------------------
            model.load_state_dict(best_state)
            _,Ptest,Ttest=evaluate(model,te_dl,crit)
            r = linregress(Ptest.numpy().flatten(), Ttest.numpy().flatten()).rvalue
            test_r2 = r ** 2
            summary.append(test_r2); print(f"▶️  RUN {run} Test R² = {test_r2:.4f}")

            with open(csv_path,"a",newline="") as f:
                csv.writer(f).writerow([run,"final",None,None,None,test_r2])

        # summary -------------------------------------------------------
        print(f"\n==== {tag.upper()} summary ====")
        print("R²:",[f"{x:.4f}" for x in summary])
        print(f"Best: {max(summary):.4f}  Mean: {np.mean(summary):.4f}  Worst: {min(summary):.4f}")
        print(f"📄 Log saved to: {csv_path}")