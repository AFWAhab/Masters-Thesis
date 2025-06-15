import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import ast
import os

def load_labels_metadata_genename(h5_path):
    with h5py.File(h5_path, "r") as f:
        labels = f["label"][:]
        full_meta = f["data"][:]
        metadata = full_meta[:, [0,1,2,3,4,5,7]]
        gene_names = [name.decode('utf-8') for name in f["geneName"][:]]
    return labels, metadata, gene_names

def load_tf_data(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df.rename(columns={df.columns[1]: 'tf_value', df.columns[2]: 'geneName'}, inplace=True)
    gene_to_tf = {}
    for idx, row in df.iterrows():
        gene_id, tf_str = row['geneName'], row['tf_value']
        try:
            tf_vector = ast.literal_eval(tf_str)
            if isinstance(tf_vector, list):
                gene_to_tf[gene_id] = np.array(tf_vector, dtype=np.float32)
        except:
            continue
    return gene_to_tf

def add_tf_to_metadata(metadata, gene_names, gene_to_tf, fill_missing=0.0):
    tf_list = []
    expected_len = None
    for g in gene_names:
        if g in gene_to_tf:
            tf_vec = gene_to_tf[g]
            if expected_len is None:
                expected_len = len(tf_vec)
            elif len(tf_vec) != expected_len:
                tf_vec = np.resize(tf_vec, expected_len)
            tf_list.append(tf_vec)
        else:
            if expected_len is None:
                tf_list.append(np.array([], dtype=np.float32))
            else:
                tf_list.append(np.full((expected_len,), fill_missing, dtype=np.float32))
    tf_array = np.array(tf_list, dtype=np.float32)
    return np.concatenate([metadata, tf_array], axis=1)

class GeneExpressionDataset(Dataset):
    def __init__(self, embeddings, metadata, targets):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.metadata = torch.tensor(metadata, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.metadata[idx], self.targets[idx]

class ExpressionPredictorEnhanced(nn.Module):
    def __init__(self, embedding_dim, metadata_dim, hidden_emb=512, hidden_meta=32, combined_hidden1=256, combined_hidden2=128):
        super().__init__()
        self.emb_branch = nn.Sequential(
            nn.Linear(embedding_dim, hidden_emb),
            nn.BatchNorm1d(hidden_emb),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        self.meta_branch = nn.Sequential(
            nn.Linear(metadata_dim, hidden_meta),
            nn.BatchNorm1d(hidden_meta),
            nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(hidden_emb + hidden_meta, combined_hidden1),
            nn.BatchNorm1d(combined_hidden1),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(combined_hidden1, combined_hidden2),
            nn.BatchNorm1d(combined_hidden2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(combined_hidden2, 1)
        )
    def forward(self, x_emb, x_meta):
        emb_out = self.emb_branch(x_emb)
        meta_out = self.meta_branch(x_meta)
        return self.combined(torch.cat((emb_out, meta_out), dim=1))

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_emb, x_meta, y in dataloader:
        x_emb, x_meta, y = x_emb.to(device), x_meta.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x_emb, x_meta)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_emb.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for x_emb, x_meta, y in dataloader:
            x_emb, x_meta, y = x_emb.to(device), x_meta.to(device), y.to(device)
            output = model(x_emb, x_meta)
            loss = criterion(output, y)
            total_loss += loss.item() * x_emb.size(0)
            preds.append(output.cpu().numpy())
            targets.append(y.cpu().numpy())
    return total_loss / len(dataloader.dataset), np.concatenate(preds), np.concatenate(targets)

if __name__ == "__main__":
    main_path = "D:\\PythonProjects\\Speciale\\"
    emb_path = main_path + "adam_experiments/dnabert1_embeddings_k6.h5"
    train_xpresso_path = main_path + "/xpresso/data/pM10Kb_1KTest/train.h5"
    valid_xpresso_path = main_path + "/xpresso/data/pM10Kb_1KTest/valid.h5"
    test_xpresso_path = main_path + "/xpresso/data/pM10Kb_1KTest/test.h5"
    tf_xlsx_path = main_path + "/DeepLncLoc/varie/transcription_factor2.xlsx"

    os.makedirs(main_path + "output", exist_ok=True)

    with h5py.File(emb_path, "r") as f:
        X_train_emb = f["train_X_embeddings"][:].reshape(-1, 18 * 768)
        X_valid_emb = f["valid_X_embeddings"][:].reshape(-1, 18 * 768)
        X_test_emb = f["test_X_embeddings"][:].reshape(-1, 18 * 768)

    y_train, meta_train, gene_train = load_labels_metadata_genename(train_xpresso_path)
    y_valid, meta_valid, gene_valid = load_labels_metadata_genename(valid_xpresso_path)
    y_test, meta_test, gene_test = load_labels_metadata_genename(test_xpresso_path)

    gene_to_tf = load_tf_data(tf_xlsx_path)
    meta_train = add_tf_to_metadata(meta_train, gene_train, gene_to_tf)
    meta_valid = add_tf_to_metadata(meta_valid, gene_valid, gene_to_tf)
    meta_test = add_tf_to_metadata(meta_test, gene_test, gene_to_tf)

    emb_dim = X_train_emb.shape[1]
    train_all = np.concatenate([X_train_emb, meta_train], axis=1)
    valid_all = np.concatenate([X_valid_emb, meta_valid], axis=1)
    test_all = np.concatenate([X_test_emb, meta_test], axis=1)

    scaler = StandardScaler()
    train_all_scaled = scaler.fit_transform(train_all)
    valid_all_scaled = scaler.transform(valid_all)
    test_all_scaled = scaler.transform(test_all)

    new_meta_dim = meta_train.shape[1]
    X_train_emb_scaled, X_train_meta_scaled = train_all_scaled[:, :emb_dim], train_all_scaled[:, emb_dim:]
    X_valid_emb_scaled, X_valid_meta_scaled = valid_all_scaled[:, :emb_dim], valid_all_scaled[:, emb_dim:]
    X_test_emb_scaled, X_test_meta_scaled = test_all_scaled[:, :emb_dim], test_all_scaled[:, emb_dim:]

    train_loader = DataLoader(GeneExpressionDataset(X_train_emb_scaled, X_train_meta_scaled, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(GeneExpressionDataset(X_valid_emb_scaled, X_valid_meta_scaled, y_valid), batch_size=64, shuffle=False)
    test_loader = DataLoader(GeneExpressionDataset(X_test_emb_scaled, X_test_meta_scaled, y_test), batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    logs = []
    for run_ix in range(1, 11):
        print(f"======= RUN {run_ix}/10 =======")
        model = ExpressionPredictorEnhanced(embedding_dim=emb_dim, metadata_dim=new_meta_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(1, 51):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_targets = evaluate_model(model, val_loader, criterion, device)
            r_val = linregress(val_preds.flatten(), val_targets.flatten()).rvalue
            val_r2 = r_val ** 2
            scheduler.step(val_loss)

            logs.append({
                "run": run_ix,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_r2": val_r2,
                "test_r2": None
            })

            print(f"Epoch {epoch}/50 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

        model.load_state_dict(best_model_state)
        _, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)
        r_test = linregress(test_preds.flatten(), test_targets.flatten()).rvalue
        r2_test = r_test ** 2
        print(f"✅ Run {run_ix} Test R²: {r2_test:.4f}")

        logs.append({
            "run": run_ix,
            "epoch": "final",
            "train_loss": None,
            "val_loss": None,
            "val_r2": None,
            "test_r2": r2_test
        })

    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(main_path + "output/cnn_attempt_3_dnabert1.csv", index=False,
                   columns=["run", "epoch", "train_loss", "val_loss", "val_r2", "test_r2"])
    print("📁 Saved logs to output/cnn_attempt_3.csv")