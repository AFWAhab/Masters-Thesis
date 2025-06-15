#!/usr/bin/env python3
# cnn_expr_cpg_mk4_swa.py  –  MK-CNN + attention pool + SWA + dilated kernels
# ==========================================================================

import io, os, ast, random, csv, h5py, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from scipy.stats import linregress
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from biomart import BiomartServer

# ───────────────────────── configuration ─────────────────────────
DATA_ROOT = "D:/PythonProjects/Speciale"
EMB_FILES = {
    "dnabert2": "dnabert2_embeddings.h5",
}

XPRESSO_DIR = "xpresso/data/pM10Kb_1KTest"
TF_EXCEL    = "DeepLncLoc/varie/transcription_factor2.xlsx"
PROM_TSV    = "beating_seq2exp/data/processed/promoter_extra_feats.tsv"
CACHE_MAP   = "gene_symbol_to_ensembl.pkl"

RUNS_PER_MODEL, EPOCHS, BATCH = 10, 30, 64          # 30 epochs with early SWA
LR_MAIN, LR_PROJ = 8e-4, 1e-4                       # two learning rates
WD, DROPOUT = 3e-5, 0.30
SWA_START_EPOCH = 8                                 # begin averaging here
EARLY_PATIENCE = 6                                  # early stop if no improvement
SEED = 2025
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = os.path.join(DATA_ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ─────────────────────── helper functions ────────────────────────
def load_symbol2ensg():
    if os.path.exists(CACHE_MAP):
        return pd.read_pickle(CACHE_MAP)
    srv = BiomartServer("https://www.ensembl.org/biomart")
    ds  = srv.datasets["hsapiens_gene_ensembl"]
    txt = ds.search({"attributes":["external_gene_name","ensembl_gene_id"]}).text
    df  = pd.read_csv(io.StringIO(txt), sep="\t", names=["sym","ensg"])
    mp  = dict(zip(df.sym, df.ensg)); pd.to_pickle(mp, CACHE_MAP); return mp

SYM2ENSG = load_symbol2ensg()
def as_ensg(x):
    x=str(x); return x.split(".")[0] if x.startswith("ENSG") else SYM2ENSG.get(x)

def load_xpresso(path):
    with h5py.File(path,"r") as f:
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
    out={}
    for _,r in df[["gene","cpg_frac"]].iterrows():
        e=as_ensg(r.gene)
        if e: out[e]=np.array([r.cpg_frac],np.float32)
    return out

def merge_meta(base,genes,*dicts):
    mats=[base]
    for d in dicts:
        k=len(next(iter(d.values())))
        buf=np.zeros((len(genes),k),np.float32)
        for i,g in enumerate(genes):
            v=d.get(g);  buf[i]=v if v is not None else 0.
        mats.append(buf)
    return np.hstack(mats)

# ───────────────────────── dataset class ─────────────────────────
class GeneDS(Dataset):
    def __init__(self,e,m,y):
        self.e=torch.tensor(e,dtype=torch.float32)  # (N,18,768)
        self.m=torch.tensor(m,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.e[i], self.m[i], self.y[i]

# ────────────────────────── model ────────────────────────────────
class AttentionPool(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1,1,dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x):
        q = self.q.expand(x.size(0), -1, -1)
        out,_ = self.attn(q, x, x, need_weights=False)
        return out.squeeze(1)

class MultiKernelBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c, k, dropout):
        super().__init__()
        pad=k//2
        self.seq = nn.Sequential(
            nn.Conv1d(in_c, mid_c, k, padding=pad),
            nn.BatchNorm1d(mid_c), nn.ReLU(),
            nn.Conv1d(mid_c, out_c, k, padding=pad*2, dilation=2),
            nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self,x): return self.seq(x)

class MKCNNBranch(nn.Module):
    PROJ_DIM  = 128
    N_FILTERS = 64
    KERNELS   = (3,7,11)
    def __init__(self, dropout):
        super().__init__()
        self.proj = nn.Linear(768, self.PROJ_DIM)
        self.paths = nn.ModuleList([
            MultiKernelBlock(self.PROJ_DIM, self.N_FILTERS,
                             self.N_FILTERS, k, dropout)
            for k in self.KERNELS
        ])
        self.pool = AttentionPool(self.N_FILTERS*len(self.KERNELS))
    def forward(self,x):
        x = self.proj(x)
        x = x.transpose(1,2)
        outs=[p(x) for p in self.paths]
        x = torch.cat(outs, dim=1)
        x = x.transpose(1,2)
        return self.pool(x)

class CNNPredictor(nn.Module):
    def __init__(self, meta_dim, dropout=0.3,
                 hidden_meta=64, h1=256, h2=128):
        super().__init__()
        self.emb  = MKCNNBranch(dropout)
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, hidden_meta),
            nn.BatchNorm1d(hidden_meta), nn.ReLU()
        )
        comb_in = 3*MKCNNBranch.N_FILTERS + hidden_meta
        self.comb = nn.Sequential(
            nn.Linear(comb_in, h1),
            nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2,1)
        )
    def forward(self,e,m):
        z1=self.emb(e); z2=self.meta(m)
        return self.comb(torch.cat([z1,z2], dim=1))

# ─────────────────────── train / eval helpers ────────────────────
def train_one_epoch(model,ldr,crit,opt):
    model.train(); tot=0.
    for e,m,y in ldr:
        e,m,y=[t.to(DEVICE) for t in (e,m,y)]
        opt.zero_grad()
        loss=crit(model(e,m),y); loss.backward(); opt.step()
        tot+=loss.item()*len(e)
    return tot/len(ldr.dataset)

@torch.no_grad()
def evaluate(model,ldr,crit):
    model.eval(); tot=0.; P,T=[],[]
    for e,m,y in ldr:
        e,m,y=[t.to(DEVICE) for t in (e,m,y)]
        out=model(e,m)
        tot+=crit(out,y).item()*len(e)
        P.append(out.cpu()); T.append(y.cpu())
    return tot/len(ldr.dataset), torch.cat(P), torch.cat(T)

@torch.no_grad()
def update_bn_manual(loader, model, device="cuda"):
    model.train()
    for e, m, _ in loader:
        e = e.to(device)
        m = m.to(device)
        model(e, m)

# ─────────────────────────── main loop ───────────────────────────
if __name__ == "__main__":
    for tag, emb_file in EMB_FILES.items():
        print(f"\n================  {tag.upper()}  =================")
        csv_path=os.path.join(OUT_DIR, f"cpg_mkcnn_swa_{tag}.csv")
        with open(csv_path,"w",newline="") as f:
            csv.writer(f).writerow(["run","epoch","train_loss",
                                    "val_loss","val_r2","test_r2"])

        with h5py.File(os.path.join(DATA_ROOT,emb_file),"r") as f:
            Etr=f["train_X_embeddings"][:]
            Eva=f["valid_X_embeddings"][:]
            Ete=f["test_X_embeddings"][:]

        xp=os.path.join(DATA_ROOT,XPRESSO_DIR)
        Ytr,Mtr,Gtr=load_xpresso(os.path.join(xp,"train.h5"))
        Yva,Mva,Gva=load_xpresso(os.path.join(xp,"valid.h5"))
        Yte,Mte,Gte=load_xpresso(os.path.join(xp,"test.h5"))

        TF=load_tf_dict(os.path.join(DATA_ROOT,TF_EXCEL))
        CP=load_cpg_dict(os.path.join(DATA_ROOT,PROM_TSV))
        Mtr=merge_meta(Mtr,Gtr,TF,CP); Mva=merge_meta(Mva,Gva,TF,CP); Mte=merge_meta(Mte,Gte,TF,CP)
        META_DIM=Mtr.shape[1]

        scaler=StandardScaler().fit(np.vstack([Mtr,Mva]))
        Mtr=scaler.transform(Mtr); Mva=scaler.transform(Mva); Mte=scaler.transform(Mte)

        tr_dl=DataLoader(GeneDS(Etr,Mtr,Ytr),BATCH,shuffle=True)
        va_dl=DataLoader(GeneDS(Eva,Mva,Yva),BATCH)
        te_dl=DataLoader(GeneDS(Ete,Mte,Yte),BATCH)

        crit=nn.MSELoss(); summary=[]
        for run in range(1, RUNS_PER_MODEL+1):
            print(f"\n-------- {tag} | RUN {run}/{RUNS_PER_MODEL} --------")

            model = CNNPredictor(META_DIM,DROPOUT).to(DEVICE)
            proj_params, other_params = [], []
            for name, param in model.named_parameters():
                if "emb.proj" in name:
                    proj_params.append(param)
                else:
                    other_params.append(param)
            opt=optim.Adam([
                {"params": proj_params, "lr": LR_PROJ},
                {"params": other_params, "lr": LR_MAIN}
            ], weight_decay=WD)

            sched = optim.lr_scheduler.ReduceLROnPlateau(opt,"min",patience=3,factor=0.5)
            swa_model = AveragedModel(model); swa_started=False
            swa_sched = SWALR(opt, swa_lr=LR_MAIN*0.6)

            best_val=float("inf"); best_state=None; patience_counter=0
            for ep in range(1,EPOCHS+1):
                tr_loss = train_one_epoch(model,tr_dl,crit,opt)
                val_loss,Pv,Tv = evaluate(model,va_dl,crit)
                r_val   = linregress(Pv.numpy().ravel(),Tv.numpy().ravel()).rvalue
                val_r2  = r_val**2

                if ep >= SWA_START_EPOCH:
                    if not swa_started:
                        print(">> SWA started")
                        swa_started=True
                    swa_model.update_parameters(model)
                    swa_sched.step()
                else:
                    sched.step(val_loss)

                print(f"  Ep{ep:02d} | Train {tr_loss:.4f} | Val {val_loss:.4f} | R² {val_r2:.4f}")
                with open(csv_path,"a",newline="") as f:
                    csv.writer(f).writerow([run,ep,tr_loss,val_loss,val_r2,None])

                if val_loss < best_val:
                    best_val,best_state = val_loss, model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= EARLY_PATIENCE:
                    print("↯ Early stopping triggered")
                    break

            if swa_started:
                update_bn_manual(tr_dl, swa_model, device=DEVICE)
                final = swa_model
            else:
                final = model
                final.load_state_dict(best_state)

            _,Pt,Tt=evaluate(final,te_dl,crit)
            test_r2 = linregress(Pt.numpy().ravel(),Tt.numpy().ravel()).rvalue**2
            summary.append(test_r2)
            print(f"▶️  RUN {run} Test R² = {test_r2:.4f}")

            with open(csv_path,"a",newline="") as f:
                csv.writer(f).writerow([run,"final",None,None,None,test_r2])

        print(f"\n==== {tag.upper()} summary ====")
        print("R²:",[f"{x:.4f}" for x in summary])
        print(f"Best {max(summary):.4f}  Mean {np.mean(summary):.4f}  Worst {min(summary):.4f}")
        print(f"Log saved to {csv_path}")
