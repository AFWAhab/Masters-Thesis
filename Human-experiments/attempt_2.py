import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import os

# ------------------ Data Loading ------------------

def load_embeddings(emb_path):
    with h5py.File(emb_path, "r") as f:
        X_train = f["train_X_embeddings"][:].reshape(-1, 18 * 768)
        X_valid = f["valid_X_embeddings"][:].reshape(-1, 18 * 768)
        X_test = f["test_X_embeddings"][:].reshape(-1, 18 * 768)
    return X_train, X_valid, X_test

def load_labels_and_metadata(h5_path):
    with h5py.File(h5_path, "r") as f:
        labels = f["label"][:]
        metadata = f["data"][:]
    return labels, metadata

# ------------------ Dataset ------------------

class GeneExpressionDataset(Dataset):
    def __init__(self, embeddings, metadata, targets):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.metadata = torch.tensor(metadata, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.metadata[idx], self.targets[idx]

# ------------------ Enhanced Model ------------------

class ExpressionPredictorEnhanced(nn.Module):
    def __init__(self, embedding_dim, metadata_dim, hidden_emb=512, hidden_meta=32, combined_hidden1=256, combined_hidden2=128):
        super(ExpressionPredictorEnhanced, self).__init__()
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
        combined_input = torch.cat((emb_out, meta_out), dim=1)
        output = self.combined(combined_input)
        return output

# ------------------ Training ------------------

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x_emb, x_meta, y in dataloader:
        x_emb, x_meta, y = x_emb.to(device), x_meta.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x_emb, x_meta)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_emb.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_emb, x_meta, y in dataloader:
            x_emb, x_meta, y = x_emb.to(device), x_meta.to(device), y.to(device)
            outputs = model(x_emb, x_meta)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x_emb.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return running_loss / len(dataloader.dataset), all_preds, all_targets

# ------------------ Main ------------------

if __name__ == "__main__":
    main_path = "D:\\PythonProjects\\Speciale\\"
    emb_path = main_path + "adam_experiments/dnabert1_embeddings_k6.h5"
    train_xpresso_path = main_path + "/xpresso/data/pM10Kb_1KTest/train.h5"
    valid_xpresso_path = main_path + "/xpresso/data/pM10Kb_1KTest/valid.h5"
    test_xpresso_path = main_path + "/xpresso/data/pM10Kb_1KTest/test.h5"

    os.makedirs("output", exist_ok=True)

    print("🚀 Loading data...")
    X_train_emb, X_valid_emb, X_test_emb = load_embeddings(emb_path)
    y_train, meta_train = load_labels_and_metadata(train_xpresso_path)
    y_valid, meta_valid = load_labels_and_metadata(valid_xpresso_path)
    y_test, meta_test = load_labels_and_metadata(test_xpresso_path)

    emb_dim = X_train_emb.shape[1]
    scaler = StandardScaler()
    train_all = np.concatenate([X_train_emb, meta_train], axis=1)
    valid_all = np.concatenate([X_valid_emb, meta_valid], axis=1)
    test_all = np.concatenate([X_test_emb, meta_test], axis=1)

    train_all_scaled = scaler.fit_transform(train_all)
    valid_all_scaled = scaler.transform(valid_all)
    test_all_scaled = scaler.transform(test_all)

    X_train_emb_scaled = train_all_scaled[:, :emb_dim]
    X_train_meta_scaled = train_all_scaled[:, emb_dim:]
    X_valid_emb_scaled = valid_all_scaled[:, :emb_dim]
    X_valid_meta_scaled = valid_all_scaled[:, emb_dim:]
    X_test_emb_scaled = test_all_scaled[:, :emb_dim]
    X_test_meta_scaled = test_all_scaled[:, emb_dim:]

    train_dataset = GeneExpressionDataset(X_train_emb_scaled, X_train_meta_scaled, y_train)
    val_dataset = GeneExpressionDataset(X_valid_emb_scaled, X_valid_meta_scaled, y_valid)
    test_dataset = GeneExpressionDataset(X_test_emb_scaled, X_test_meta_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_runs = 10
    num_epochs = 50
    logs = []

    for run in range(1, num_runs + 1):
        print(f"\n🔥 Run {run}/{num_runs}")
        model = ExpressionPredictorEnhanced(embedding_dim=emb_dim, metadata_dim=meta_train.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(1, num_epochs + 1):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_targets = evaluate_model(model, val_loader, criterion, device)
            r_val = linregress(val_preds.flatten(), val_targets.flatten()).rvalue
            val_r2 = r_val ** 2
            scheduler.step(val_loss)

            logs.append({
                "run": run,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_r2": val_r2,
                "test_r2": None
            })

            print(f"Run {run} | Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R²: {val_r2:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

        model.load_state_dict(best_model_state)
        _, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)
        r_test = linregress(test_preds.flatten(), test_targets.flatten()).rvalue
        test_r2 = r_test ** 2
        print(f"✅ Run {run} Test R²: {test_r2:.4f}")

        logs.append({
            "run": run,
            "epoch": "final",
            "train_loss": None,
            "val_loss": None,
            "val_r2": None,
            "test_r2": test_r2
        })

    df_logs = pd.DataFrame(logs)
    df_logs.to_csv("output/cnn_attempt_2_dnabert1.csv", index=False,
                   columns=["run", "epoch", "train_loss", "val_loss", "val_r2", "test_r2"])
    print("\n📁 Metrics saved to output/cnn_attempt_2.csv")
