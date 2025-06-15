# -------------------------------------------------
# brute_force_half_life_subsets.py
# -------------------------------------------------
import itertools, ast, h5py, time, torch
import numpy as np, pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ========== hyper‑params ==========
N_EPOCHS   = 50
BATCH_SIZE = 64
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 device:", DEVICE)

# ========== data helpers ==========
def load_labels_metadata_genename(h5_path):
    with h5py.File(h5_path, "r") as f:
        y     = f["label"][:]
        meta  = f["data"][:]             # keep ALL 8 half‑life cols
        genes = [g.decode() if isinstance(g, bytes) else g
                 for g in f["geneName"][:]]
    return y, meta, genes

def load_tf_data(xlsx_path):
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.rename(columns={df.columns[1]: "tf_vec",
                       df.columns[2]: "geneName"}, inplace=True)
    return {row["geneName"]:
            np.asarray(ast.literal_eval(row["tf_vec"]), dtype=np.float32)
            for _, row in df.iterrows()}

def add_tf(meta, genes, g2tf, fill=0.0):
    tf_len = len(next(iter(g2tf.values())))
    tf_mat = np.vstack([g2tf.get(g, np.full(tf_len, fill)) for g in genes])
    return np.hstack([meta, tf_mat])

# ========== torch dataset & model ==========
class GeneDS(Dataset):
    def __init__(self, e, m, y):
        self.e = torch.tensor(e, dtype=torch.float32)
        self.m = torch.tensor(m, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.e[i], self.m[i], self.y[i]

class Predictor(nn.Module):
    def __init__(self, emb_dim, meta_dim,
                 h_emb=512, h_meta=32,
                 h1=256, h2=128):
        super().__init__()
        self.eb = nn.Sequential(
            nn.Linear(emb_dim, h_emb),
            nn.BatchNorm1d(h_emb), nn.ReLU(), nn.Dropout(0.15))
        self.mb = nn.Sequential(
            nn.Linear(meta_dim, h_meta),
            nn.BatchNorm1d(h_meta), nn.ReLU())
        self.fuse = nn.Sequential(
            nn.Linear(h_emb+h_meta, h1),
            nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h2, 1))
    def forward(self, e, m):
        return self.fuse(torch.cat([self.eb(e), self.mb(m)], 1))

# ========== train / eval ==========
def train_once(tr_dl, va_dl, emb_dim, meta_dim):
    model = Predictor(emb_dim, meta_dim).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()
    best = {"r2": -1, "epoch": -1, "val_loss": np.inf}
    for epoch in range(1, N_EPOCHS+1):
        # --- train
        model.train()
        for e,m,y in tr_dl:
            e,m,y = e.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(e,m), y)
            loss.backward(); opt.step()
        # --- val
        model.eval(); preds=[]; targs=[]; vloss=0
        with torch.no_grad():
            for e,m,y in va_dl:
                e,m,y = e.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
                out   = model(e,m)
                vloss += crit(out,y).item()*y.size(0)
                preds.append(out.cpu()); targs.append(y.cpu())
        vloss /= len(va_dl.dataset)
        r2 = r2_score(torch.cat(targs), torch.cat(preds))
        if r2 > best["r2"]:
            best.update({"r2": r2, "epoch": epoch, "val_loss": vloss})
    return best

# ========== main sweep ==========
def main():
    emb_path = "D:/PythonProjects/Speciale/dnabert2_embeddings.h5"
    train_h5 = "D:/PythonProjects/Speciale/xpresso/data/pM10Kb_1KTest/train.h5"
    valid_h5 = "D:/PythonProjects/Speciale/xpresso/data/pM10Kb_1KTest/valid.h5"
    tf_xlsx  = "D:/PythonProjects/Speciale/DeepLncLoc/varie/transcription_factor2.xlsx"

    # embeddings
    with h5py.File(emb_path) as f:
        E_tr = f["train_X_embeddings"][:].reshape(-1, 18*768)
        E_va = f["valid_X_embeddings"][:].reshape(-1, 18*768)
    emb_dim = E_tr.shape[1]

    # metadata & TF
    y_tr, H_tr, g_tr = load_labels_metadata_genename(train_h5)
    y_va, H_va, g_va = load_labels_metadata_genename(valid_h5)
    tf_map = load_tf_data(tf_xlsx)
    H_tr   = add_tf(H_tr, g_tr, tf_map)
    H_va   = add_tf(H_va, g_va, tf_map)

    half_cols = list(range(8))             # 8 half-life features
    tf_dim    = H_tr.shape[1] - 8

    results=[]
    sweep_start = time.perf_counter()

    for k in range(1, 9):
        for subset in itertools.combinations(half_cols, k):
            t0 = time.perf_counter()
            mask = np.array([i in subset for i in half_cols] + [True]*tf_dim)
            Htr_sub, Hva_sub = H_tr[:,mask], H_va[:,mask]

            scaler = StandardScaler().fit(np.hstack([E_tr, Htr_sub]))
            tr_all = scaler.transform(np.hstack([E_tr, Htr_sub]))
            va_all = scaler.transform(np.hstack([E_va, Hva_sub]))

            Etr_s, Htr_s = tr_all[:,:emb_dim], tr_all[:,emb_dim:]
            Eva_s, Hva_s = va_all[:,:emb_dim], va_all[:,emb_dim:]

            tr_dl = DataLoader(GeneDS(Etr_s, Htr_s, y_tr),
                               BATCH_SIZE, shuffle=True)
            va_dl = DataLoader(GeneDS(Eva_s, Hva_s, y_va),
                               BATCH_SIZE)

            best = train_once(tr_dl, va_dl, emb_dim, Htr_s.shape[1])
            run_time = time.perf_counter() - t0

            mask_str = "".join("1" if i in subset else "0" for i in half_cols)
            results.append({
                "mask"      : mask_str,
                "size"      : k,
                "best_r2"   : best["r2"],
                "best_epoch": best["epoch"],
                "best_vloss": best["val_loss"],
                "seconds"   : run_time
            })
            print(f"subset {mask_str} (k={k}) → R²={best['r2']:.4f} "
                  f"| epoch {best['epoch']:02d} | {run_time:.1f}s")

    total_time = time.perf_counter() - sweep_start
    print(f"\n🕒 total sweep time: {total_time/60:.1f} minutes "
          f"for {len(results)} runs")

    df = pd.DataFrame(results).sort_values("best_r2", ascending=False)
    df.to_csv("half_life_subset_scan_50epochs.csv", index=False)
    print("✅ results saved to half_life_subset_scan.csv")
    print(df.head())

if __name__ == "__main__":
    main()
