import os
import ast
import h5py
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GraphNorm
from torch_geometric.utils import add_self_loops, degree
from scipy import stats

# === PATH CONFIG ===
NODES_PQ      = r"gnn_setup/gnn_nodes_H3K27ac_100k.parquet"
EDGES_PQ      = r"gnn_setup/edges_H3K27ac_100k.parquet"
NODES_ANCHOR  = r"gnn_setup/nodes_anchor_H3K27ac_100k.parquet"
EXP_PQ        = r"zebrafish_expression_median.parquet"
DNABERT_H5    = r"zebrafish_embeddings.h5"
TF_XLSX       = r"tf_excel.xlsx"
#TRAIN_XPRESSO = r"C:\Users\gusta\PycharmProjects\ThesisPlayground\training\zebrafish_training\zebrafish_train.hdf5"
#VALID_XPRESSO = r"C:\Users\gusta\PycharmProjects\ThesisPlayground\training\zebrafish_training\zebrafish_val.hdf5"
#TEST_XPRESSO  = r"C:\Users\gusta\PycharmProjects\ThesisPlayground\training\zebrafish_training\zebrafish_test.hdf5"
EMBEDDING_GENES_TRAIN = r"embedding_data/train-001.h5"
EMBEDDING_GENES_VALID = r"embedding_data/valid.h5"
EMBEDDING_GENES_TEST = r"embedding_data/test.h5"
XPRESSO       = r"C:\Users\gusta\PycharmProjects\ThesisPlayground\training\zebrafish.hdf5"

# --- Xpresso loader ---
def load_labels_metadata_genename(path):
    with h5py.File(path, 'r') as f:
        labels = f['label'][:]
        full_meta = f['halflifeData'][:]
        metadata = full_meta[:, [0,1,2,3,4,5,7]]
        genes = [g.decode() for g in f['geneName'][:]]
    return labels, metadata, genes

# --- TF loader ---
def load_tf_data(path):
    df = pd.read_excel(path)
    df.rename(columns={df.columns[1]: 'tf_value', df.columns[2]: 'gene_id'}, inplace=True)
    g2t = {}
    for _, row in df.iterrows():
        try:
            vec = ast.literal_eval(row['tf_value'])
            if isinstance(vec, list):
                g2t[row['gene_id']] = np.array(vec, dtype=np.float32)
        except:
            pass
    return g2t

# --- one-hot for anchors ---
def one_hot_encode_seq(seq, length=1000):
    mp = {'A':0,'C':1,'G':2,'T':3}
    arr = np.zeros((4, length), dtype=np.float32)
    for i, ch in enumerate(seq[:length]):
        if ch in mp:
            arr[mp[ch], i] = 1.0
    return arr.flatten()

# --- CNN projector ---
class ConvProjector(nn.Module):
    def __init__(self, in_c=768, out_c=256):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv(x)
        return self.pool(x).squeeze(-1)

# --- GAT regressor ---
class RegGNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.norm1 = GraphNorm(in_dim)
        self.gat1 = GATv2Conv(in_dim, 64, heads=2, edge_dim=1)
        self.gat2 = GATv2Conv(64 * 2, 64, heads=2, edge_dim=1)
        #self.norm2 = GraphNorm(256*4)
        self.norm2 = GraphNorm(64 * 2)
        self.fc = nn.Sequential(
            #nn.Linear(256*4, 128),
            nn.Linear(2 * 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, data):
        x, ei, eattr = data.x, data.edge_index, data.edge_weight
        x = self.norm1(x)
        x = F.relu(self.gat1(x, ei, edge_attr=eattr))
        x = F.relu(self.gat2(x, ei, edge_attr=eattr))
        x = self.norm2(x)
        return self.fc(x).squeeze()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # 1) promoters + labels
    nodes = pl.read_parquet(NODES_PQ)
    expr  = pl.read_parquet(EXP_PQ)
    expr = expr.with_columns([
        pl.col("gene_id").cast(pl.Utf8)
    ])
    prom  = nodes.join(expr, on='gene_id').sort('node_id')
    number_of_prom    = prom.shape[0]
    genes_prom = prom['gene_id'].to_list()
    gene_to_label = dict(zip(prom["gene_id"].to_list(), prom["label"].to_list()))

    # 2) xpresso splits
    y_tr, m_tr, g_tr = load_labels_metadata_genename(XPRESSO)
    #y_tr, m_tr, g_tr = load_labels_metadata_genename(TRAIN_XPRESSO)
    #y_va, m_va, g_va = load_labels_metadata_genename(VALID_XPRESSO)
    #y_te, m_te, g_te = load_labels_metadata_genename(TEST_XPRESSO)
    all_meta  = np.vstack([m_tr])
    gene_to_id_xpresso = {g:i for i,g in enumerate(g_tr)}

    # 3) DNABERT embeddings
    with h5py.File(DNABERT_H5,'r') as f:
        emb = np.vstack([f['train_X_embeddings'][:], f['valid_X_embeddings'][:], f['test_X_embeddings'][:]])
        #gene_to_embedding = {g:i for i,g in enumerate(np.concatenate([f['train_X_embeddings'][:], f['valid_X_embeddings'][:], f['test_X_embeddings'][:]])}
    with h5py.File(EMBEDDING_GENES_TRAIN,'r') as f:
        emb_genes_train = np.array(f["geneName"][:])
    with h5py.File(EMBEDDING_GENES_VALID,'r') as f:
        emb_genes_valid = np.array(f["geneName"][:])
    with h5py.File(EMBEDDING_GENES_TEST,'r') as f:
        emb_genes_test = np.array(f["geneName"][:])
    emb_genes = np.concatenate([emb_genes_train, emb_genes_valid, emb_genes_test]).tolist()
    emb_genes_str = [gene.decode("utf-8") for gene in emb_genes]
    proj = ConvProjector().to(device)
    with torch.no_grad():
        Xproj = proj(torch.tensor(emb, dtype=torch.float32, device=device)).cpu().numpy()

    # 5) TF
    g2t = load_tf_data(TF_XLSX)
    #tf_genes = g2t.keys()
    tf_dim = next(iter(g2t.values())).shape[0] if g2t else 0
    #tf_list = []
    #for i, g in enumerate(genes_prom):
    #    if g in g2t and g in gene_to_id_xpresso:
    #        tf_list.append(g2t[g])
    #X_tf = np.zeros((len(tf_list), td), dtype=np.float32)
    #for idx, tf in enumerate(tf_list):
    #    X_tf[idx] = tf

    # 4) align promoter embeddings & meta
    genes_align = [g for g in genes_prom if g in gene_to_id_xpresso and g in g2t and g in emb_genes_str]
    with open('genes_aligned.txt', 'w') as f:
        for line in genes_align:
            f.write("%s\n" % line)
    indexes_for_meta = []
    tfs = []
    prom_embeddings = []
    labels_aligned = []
    for gene in genes_align:
        # align TF, embedding and meta
        indexes_for_meta.append(gene_to_id_xpresso[gene])
        tfs.append(g2t[gene])
        emd_idx = emb_genes_str.index(gene)
        prom_embeddings.append(Xproj[emd_idx])
        labels_aligned.append(gene_to_label[gene])

    meta_prm = all_meta[indexes_for_meta]

    #emd_idx = [idx for idx, gene in enumerate(emb_genes) if gene.decode("utf-8") in genes_align]
    #X_prom_emb = Xproj[emd_idx]
    #meta_idx = [idx for (gene, idx) in gene_to_id_xpresso.items() if gene in genes_align]
    #meta_prom   = all_meta[meta_idx]

    # 6) combine prom features
    X_prom = np.hstack([prom_embeddings, tfs, meta_prm, np.ones((len(prom_embeddings),1),dtype=np.float32)])

    # 7) anchors
    anch = pl.read_parquet(NODES_ANCHOR)
    seqs = anch['seq'].to_list()
    X_seq = np.vstack([one_hot_encode_seq(s) for s in seqs])
    number_of_anch = X_seq.shape[0]
    proj_a = nn.Sequential(nn.Linear(X_seq.shape[1],256), nn.ReLU(), nn.Dropout(0.1)).to(device)
    with torch.no_grad():
        X_anc = proj_a(torch.tensor(X_seq, dtype=torch.float32, device=device)).cpu().numpy()
    X_anc = np.hstack([X_anc, np.zeros((number_of_anch, tf_dim)), np.zeros((number_of_anch, meta_prm.shape[1])), np.zeros((number_of_anch, 1))])

    # 8) full + scale
    X = np.vstack([X_prom, X_anc])
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    # 9) labels
    X_prom_len = len(X_prom)
    y = np.zeros(X_prom_len + number_of_anch, dtype=np.float32)
    #y[:number_of_prom] = prom['label'].to_numpy()
    y[:X_prom_len] = np.array(labels_aligned)

    # 10) graph w/ hic
    ed = pl.read_parquet(EDGES_PQ)
    ei = torch.tensor(ed[['src','tgt']].to_numpy().T, dtype=torch.long)
    ew = torch.tensor(ed['w'].to_numpy(), dtype=torch.float32).unsqueeze(1)
    ew = (ew - ew.mean()) / (ew.std() + 1e-6)
    ew = ew.clamp(-1.0, +1.0)  # focus on moderate contacts
    ei, ew = add_self_loops(ei, edge_attr=ew, fill_value=1.0, num_nodes=X_prom_len + number_of_anch)

    data = Data(x=torch.tensor(X, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.float32),
                edge_index=ei,
                edge_weight=ew)

    # 11) masks
    perm = torch.randperm(X_prom_len)
    tcut, vcut = int(X_prom_len * 0.8), int(X_prom_len * 0.9)
    tm = torch.zeros(X_prom_len + number_of_anch, dtype=torch.bool);
    vm = tm.clone()
    sm = tm.clone()
    tm[perm[:tcut]] = True; vm[perm[tcut:vcut]] = True; sm[perm[vcut:]] = True
    pmask = torch.zeros_like(tm); pmask[:X_prom_len] = True
    data.train_mask = tm & pmask
    data.valid_mask = vm & pmask
    data.test_mask  = sm & pmask

    print(f"Mean promoter degree: {degree(ei[0], num_nodes=X.shape[0])[:number_of_prom].mean():.2f}")
    #uhh = ei[0]
    #degree_of_prom = degree(uhh, num_nodes=X_prom_len)
    #print(f"Mean promoter degree: {degree_of_prom[:X_prom_len].mean():.2f}")


    # 12) train
    model = RegGNN(X.shape[1]).to(device)
    data = data.to(device)
    opt = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=10,factor=0.5)
    loss_fn = nn.MSELoss()
    best_r2 = -np.inf
    for ep in range(1,201):
        model.train()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(data)[data.valid_mask].cpu().numpy()
            vt = data.y[data.valid_mask].cpu().numpy()
            vr2 = r2_score(vt, vp)
        sched.step(loss)
        print(f"Epoch {ep:03d} | Train Loss {loss:.4f} | Val R² {vr2:.4f}")
        if vr2>best_r2:
            best_r2=vr2; torch.save(model.state_dict(),'best_model.pt')
    # 13) test
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    with torch.no_grad():
        tp = model(data)[data.test_mask].cpu().numpy()
        tt = data.y[data.test_mask].cpu().numpy()
    print(f"Test R²: {r2_score(tt,tp):.4f}")

    slope, intercept, r_value, p_value, std_err = stats.linregress(tp, tt)
    print('Test R^2 = %.3f' % r_value ** 2)