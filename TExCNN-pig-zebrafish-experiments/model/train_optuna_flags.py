import h5py
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score
import time
from optuna_dashboard import run_server
from imblearn.over_sampling import SMOTE

# STUDY_NAME = "10k-corrected-100-early-stop-patience-10-flags"
STUDY_NAME = "FLAG_TEST-32-positive-negative-balanced-threshold-tuning-mean-all"
WINDOWS = 18  # base=1i8


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x shape: (batch, channels, seq_len)
        """
        skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + skip  # residual add
        x = self.act(x)
        return x


class DNABERTResNet(nn.Module):
    """
    A deeper net:
      - entry_conv: from input=768 channels to conv_channels
      - num_blocks of residual blocks
      - global pool
      - optional tabular features
      - final MLP
    """

    def __init__(self, conv_channels=128, kernel_size=3, fc_units=128, tabular_dim=0, num_blocks=2):
        super().__init__()
        self.entry_conv = nn.Conv1d(768, conv_channels, kernel_size=kernel_size,
                                    padding=kernel_size // 2)
        self.entry_bn = nn.BatchNorm1d(conv_channels)

        # stack multiple ResBlocks
        self.blocks = nn.ModuleList([
            ResBlock(conv_channels, kernel_size=kernel_size)
            for _ in range(num_blocks)
        ])

        self.pool = nn.AdaptiveMaxPool1d(1)

        # after pooling, we have conv_channels dims
        self.tabular_dim = tabular_dim
        self.total_input_dim = conv_channels + tabular_dim

        self.fc1 = nn.Linear(self.total_input_dim, fc_units)
        self.fc2 = nn.Linear(fc_units, 1)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, tabular_features=None):
        """
        x shape: (batch, 18, 768) => permute => (batch, 768, 18)
        ...
        after pool => (batch, conv_channels)
        if tabular_features is not None, cat => shape (batch, conv_channels + tabular_dim)
        final => (batch, 1)
        """
        # permute so channels are second dimension
        x = x.permute(0, 2, 1)  # (batch, 768, 18)

        # entry conv + BN
        x = self.entry_conv(x)  # (batch, conv_channels, 18)
        x = self.entry_bn(x)
        x = self.act(x)

        # pass through each residual block
        for block in self.blocks:
            x = block(x)  # shape remains (batch, conv_channels, 18)

        # global max pool or avg pool
        x = self.pool(x).squeeze(-1)  # (batch, conv_channels)

        if tabular_features is not None:
            x = torch.cat([x, tabular_features], dim=1)  # (batch, conv_channels + tabular_dim)

        x = self.dropout(self.act(self.fc1(x)))
        out = self.fc2(x)
        return out


def objective_early_stopping(trial):
    import copy

    # === Hyperparameter Suggestions ===
    epochs = trial.suggest_int('epochs', 100, 100, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 128, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    conv_channels = trial.suggest_int('conv_channels', 64, 256, log=True)
    kernel_size = 5
    fc_units = trial.suggest_int('fc_units', 64, 256, log=True)
    num_blocks = trial.suggest_int('num_blocks', 2, 4, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Data Splitting ===
    X_train_all = np.concatenate([X_train, X_valid], axis=0)
    y_train_all = np.concatenate([y_train, y_valid], axis=0)

    X_train_val, X_eval, y_train_val, y_eval = train_test_split(X_train_all, y_train_all, test_size=0.2,
                                                                random_state=42)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_val, y_train_val, test_size=0.2,
                                                                              random_state=42)

    if tabular_train is not None:
        tabular_all = np.concatenate([tabular_train, tabular_valid], axis=0)
        tabular_all_tensor = torch.from_numpy(tabular_all).float()
        tabular_test_tensor = torch.from_numpy(tabular_test).float()
    else:
        tabular_all_tensor = None
        tabular_test_tensor = None
        print("No tabular data D:")

    # === Dataset & Dataloader ===
    train_dataset = TensorDataset(torch.from_numpy(X_train_split).float(), torch.from_numpy(y_train_split).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val_split).float(), torch.from_numpy(y_val_split).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tabular_dim = tabular_train.shape[1] if tabular_train is not None else 0
    model = DNABERTResNet(conv_channels=conv_channels,
                          kernel_size=kernel_size,
                          fc_units=fc_units,
                          tabular_dim=tabular_dim,
                          num_blocks=num_blocks).to(device)

    absent = 3878
    present = 12636
    pos_weight = torch.tensor([absent / present], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # === Early Stopping Setup ===
    patience = 5
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    # === Training Loop ===
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if tabular_all_tensor is not None:
                batch_tab = tabular_all_tensor[i * batch_size: (i + 1) * batch_size].to(device)
            else:
                batch_tab = None

            optimizer.zero_grad()
            preds = model(batch_x, batch_tab).squeeze(-1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # === Validation Step ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                if tabular_all_tensor is not None:
                    batch_tab = tabular_all_tensor[i * batch_size: (i + 1) * batch_size].to(device)
                else:
                    batch_tab = None
                preds = model(batch_x, batch_tab).squeeze(-1)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # === Check for improvement ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    train_time = (time.time() - t0) / 60
    print(f"Training finished in {train_time:.2f} min.")

    # === Load Best Model ===
    model.load_state_dict(best_model)

    # === Final Evaluation on Test Set ===
    model.eval()
    preds_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if tabular_test_tensor is not None:
                batch_tab = tabular_test_tensor[i * batch_size: (i + 1) * batch_size].to(device)
            else:
                batch_tab = None
            preds = model(batch_x, batch_tab).squeeze(-1)
            preds_list.append(preds.cpu().numpy())
            y_list.append(batch_y.cpu().numpy())

    preds_all = np.concatenate(preds_list)
    y_true_all = np.concatenate(y_list)

    # === Threshold Tuning ===
    best_threshold = 0.5
    PPV_factor = 0
    TPR_factor = 0.33  # Recall
    TNR_factor = 0.33
    NPV_factor = 0.33
    best_score = -np.inf  # Score to maximize, could be custom (e.g., recall + npv)

    def balanced_score(recall, tnr, npv, precision):
        metrics = [recall, tnr, npv, precision]
        mean_val = np.mean(metrics)
        std_val = np.std(metrics)
        return 1 - mean_val  # Smaller std ⇒ more balanced ⇒ higher score

    for thresh in np.arange(0.1, 0.91, 0.05):
        preds_binary = (preds_all > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_all, preds_binary).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # score = recall * TPR_factor
        # score += npv * NPV_factor
        # score += tnr * TNR_factor

        score = balanced_score(recall, tnr, npv, precision)

        if score > best_score:
            best_score = score
            best_threshold = thresh
            print(f"Found better threshold: {thresh} ({score})")
        else:
            print(f"Threshold: {thresh} is not better ({score} < {best_score})")

    # === Final Metrics at Best Threshold ===
    preds_binary = (preds_all > best_threshold).astype(int)
    acc = accuracy_score(y_true_all, preds_binary)
    tn, fp, fn, tp = confusion_matrix(y_true_all, preds_binary).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    print(f"\n=== Test Metrics @ Threshold {best_threshold:.2f} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (PPV): {precision:.4f}")
    print(f"Recall (TPR): {recall:.4f}")
    print(f"True Negative Rate (TNR): {tnr:.4f}")
    print(f"Negative Predictive Value (NPV): {npv:.4f}")

    # test_mse = mean_squared_error(y_true_all, preds_all)
    # test_r2 = r2_score(y_true_all, preds_all)
    # print(f"\n=== Test Results ===\nMSE: {test_mse:.4f}\nR²: {test_r2:.4f}")

    global trial_no
    with open("analysis/run_data/" + STUDY_NAME + ".csv", 'a') as file:
        file.write(
            f"{trial_no},{acc:.4f},{precision:.4f},{recall:.4f},{tnr:.4f},{npv:.4f},{epoch + 1},{batch_size},{lr},{weight_decay},{conv_channels},{kernel_size},{fc_units},{num_blocks},{PPV_factor},{TPR_factor},{TNR_factor},{NPV_factor},{best_threshold:.2f}\n")
    trial_no += 1
    print(f"Trial number: {trial_no}")

    return 1 - acc


def eval_final(trial):
    import copy

    # === Hyperparameter Suggestions ===
    epochs = trial.suggest_int('epochs', 100, 100, log=True)
    batch_size = trial.suggest_int('batch_size', 95, 95, log=True)
    lr = trial.suggest_float('lr', 0.0026281237492164987, 0.0026281237492164987, log=True)
    weight_decay = trial.suggest_float('weight_decay', 2.166626878749766e-07, 2.166626878749766e-07, log=True)
    conv_channels = trial.suggest_int('conv_channels', 68, 68, log=True)
    kernel_size = 5
    fc_units = trial.suggest_int('fc_units', 204, 204, log=True)
    num_blocks = trial.suggest_int('num_blocks', 4, 4, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Data Splitting ===
    X_train_all = np.concatenate([X_train, X_valid], axis=0)
    y_train_all = np.concatenate([y_train, y_valid], axis=0)

    X_train_val, X_eval, y_train_val, y_eval = train_test_split(X_train_all, y_train_all, test_size=0.2,
                                                                random_state=42)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_val, y_train_val, test_size=0.2,
                                                                              random_state=42)

    if tabular_train is not None:
        tabular_all = np.concatenate([tabular_train, tabular_valid], axis=0)
        tabular_all_tensor = torch.from_numpy(tabular_all).float()
        tabular_test_tensor = torch.from_numpy(tabular_test).float()
    else:
        tabular_all_tensor = None
        tabular_test_tensor = None

    # === Dataset & Dataloader ===
    train_dataset = TensorDataset(torch.from_numpy(X_train_split).float(), torch.from_numpy(y_train_split).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val_split).float(), torch.from_numpy(y_val_split).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tabular_dim = tabular_train.shape[1] if tabular_train is not None else 0
    model = DNABERTResNet(conv_channels=conv_channels,
                          kernel_size=kernel_size,
                          fc_units=fc_units,
                          tabular_dim=tabular_dim,
                          num_blocks=num_blocks).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # === Early Stopping Setup ===
    patience = 100
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    # === Training Loop ===
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if tabular_all_tensor is not None:
                batch_tab = tabular_all_tensor[i * batch_size: (i + 1) * batch_size].to(device)
            else:
                batch_tab = None

            optimizer.zero_grad()
            preds = model(batch_x, batch_tab).squeeze(-1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # === Validation Step ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                if tabular_all_tensor is not None:
                    batch_tab = tabular_all_tensor[i * batch_size: (i + 1) * batch_size].to(device)
                else:
                    batch_tab = None
                preds = model(batch_x, batch_tab).squeeze(-1)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # === Check for improvement ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    train_time = (time.time() - t0) / 60
    print(f"Training finished in {train_time:.2f} min.")

    # === Load Best Model ===
    model.load_state_dict(best_model)

    # === Final Evaluation on Test Set ===
    model.eval()
    preds_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if tabular_test_tensor is not None:
                batch_tab = tabular_test_tensor[i * batch_size: (i + 1) * batch_size].to(device)
            else:
                batch_tab = None
            preds = model(batch_x, batch_tab).squeeze(-1)
            preds_list.append(preds.cpu().numpy())
            y_list.append(batch_y.cpu().numpy())

    preds_all = np.concatenate(preds_list)
    y_true_all = np.concatenate(y_list)

    preds_binary = (preds_all > 0.5).astype(int)
    acc = accuracy_score(y_true_all, preds_binary)
    print(f"\n=== Test Accuracy: {acc:.4f} ===")

    # test_mse = mean_squared_error(y_true_all, preds_all)
    # test_r2 = r2_score(y_true_all, preds_all)
    # print(f"\n=== Test Results ===\nMSE: {test_mse:.4f}\nR²: {test_r2:.4f}")

    global trial_no
    with open("analysis/run_data/" + STUDY_NAME + ".csv", 'a') as file:
        file.write(
            f"{trial_no},{acc:.4f},{precision:.4f},{recall:.4f},{tnr:.4f},{npv:.4f},{epoch + 1},{batch_size},{lr},{weight_decay},{conv_channels},{kernel_size},{fc_units},{num_blocks}\n")
    trial_no += 1
    print(f"Trial number: {trial_no}")

    return 1 - acc


def objective(trial):
    epochs = trial.suggest_int('epochs', 30, 100, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 128, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    conv_channels = trial.suggest_int('conv_channels', 64, 256, log=True)
    kernel_size = 5  # seems hard coded by embedding form
    fc_units = trial.suggest_int('fc_units', 64, 256, log=True)
    num_blocks = trial.suggest_int('num_blocks', 2, 4, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # combine train + valid
    X_train_all = np.concatenate([X_train, X_valid], axis=0)
    y_train_all = np.concatenate([y_train, y_valid], axis=0)

    if tabular_train is not None:
        tabular_all = np.concatenate([tabular_train, tabular_valid], axis=0)
        tabular_all_tensor = torch.from_numpy(tabular_all).float()
        tabular_test_tensor = torch.from_numpy(tabular_test).float()
    else:
        tabular_all_tensor = None
        tabular_test_tensor = None

    # create datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train_all).float(),
                                  torch.from_numpy(y_train_all).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(),
                                 torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tabular_dim = tabular_train.shape[1] if tabular_train is not None else 0
    model = DNABERTResNet(conv_channels=conv_channels,
                          kernel_size=kernel_size,
                          fc_units=fc_units,
                          tabular_dim=tabular_dim,
                          num_blocks=num_blocks).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # reduce LR when train loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=2,
                                                     verbose=True)

    # training loop
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if tabular_train is not None:
                batch_tab = tabular_all_tensor[i * batch_size: (i + 1) * batch_size].to(device)
            else:
                batch_tab = None

            optimizer.zero_grad()
            preds = model(batch_x, batch_tab).squeeze(-1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)  # updates LR if needed

    train_time = (time.time() - t0) / 60
    print(f"Training finished in {train_time:.2f} min.")

    # evaluation on test
    model.eval()
    preds_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if tabular_test is not None:
                batch_tab = tabular_test_tensor[i * batch_size: (i + 1) * batch_size].to(device)
            else:
                batch_tab = None

            preds = model(batch_x, batch_tab).squeeze(-1)
            preds_list.append(preds.cpu().numpy())
            y_list.append(batch_y.cpu().numpy())

    preds_all = np.concatenate(preds_list)
    y_true_all = np.concatenate(y_list)

    test_mse = mean_squared_error(y_true_all, preds_all)
    test_r2 = r2_score(y_true_all, preds_all)
    print(f"\n=== Test Results ===\nMSE: {test_mse:.4f}\nR²: {test_r2:.4f}")
    global trial_no
    with open("analysis/run_data/" + STUDY_NAME + ".csv", 'a') as file:
        file.write(
            f"{trial_no},{test_mse},{test_r2},{epochs},{batch_size},{lr},{weight_decay},{conv_channels},{kernel_size},{fc_units},{num_blocks}\n")
    trial_no += 1
    print(f"Trial number: {trial_no}")
    # return model, test_mse, test_r2
    return test_mse


def train_and_eval_cnn(
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test,
        hparams,
        tabular_train=None,
        tabular_valid=None,
        tabular_test=None
):
    """
    - Creates a DNABERTResNet model with residual blocks
    - Uses AdamW + ReduceLROnPlateau
    - MSE loss, prints R² on test set
    """
    epochs = hparams.get('epochs', 20)
    batch_size = hparams.get('batch_size', 64)
    lr = hparams.get('lr', 1e-3)
    weight_decay = hparams.get('weight_decay', 1e-5)
    conv_channels = hparams.get('conv_channels', 128)
    kernel_size = hparams.get('kernel_size', 3)
    fc_units = hparams.get('fc_units', 128)
    num_blocks = hparams.get('num_blocks', 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # combine train + valid
    X_train_all = np.concatenate([X_train, X_valid], axis=0)
    y_train_all = np.concatenate([y_train, y_valid], axis=0)

    if tabular_train is not None:
        tabular_all = np.concatenate([tabular_train, tabular_valid], axis=0)
        tabular_all_tensor = torch.from_numpy(tabular_all).float()
        tabular_test_tensor = torch.from_numpy(tabular_test).float()
    else:
        tabular_all_tensor = None
        tabular_test_tensor = None

    # create datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train_all).float(),
                                  torch.from_numpy(y_train_all).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(),
                                 torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tabular_dim = tabular_train.shape[1] if tabular_train is not None else 0
    model = DNABERTResNet(conv_channels=conv_channels,
                          kernel_size=kernel_size,
                          fc_units=fc_units,
                          tabular_dim=tabular_dim,
                          num_blocks=num_blocks).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # reduce LR when train loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=2,
                                                     verbose=True)

    # training loop
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if tabular_train is not None:
                batch_tab = tabular_all_tensor[i * batch_size: (i + 1) * batch_size].to(device)
            else:
                batch_tab = None

            optimizer.zero_grad()
            preds = model(batch_x, batch_tab).squeeze(-1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)  # updates LR if needed

    train_time = (time.time() - t0) / 60
    print(f"Training finished in {train_time:.2f} min.")

    # evaluation on test
    model.eval()
    preds_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if tabular_test is not None:
                batch_tab = tabular_test_tensor[i * batch_size: (i + 1) * batch_size].to(device)
            else:
                batch_tab = None

            preds = model(batch_x, batch_tab).squeeze(-1)
            preds_list.append(preds.cpu().numpy())
            y_list.append(batch_y.cpu().numpy())

    preds_all = np.concatenate(preds_list)
    y_true_all = np.concatenate(y_list)

    test_mse = mean_squared_error(y_true_all, preds_all)
    test_r2 = r2_score(y_true_all, preds_all)
    print(f"\n=== Test Results ===\nMSE: {test_mse:.4f}\nR²: {test_r2:.4f}")
    return model, test_mse, test_r2


###################################################
# 4) Putting it all together in main
###################################################
if __name__ == "__main__":

    global X_train, y_train, X_valid, y_valid, X_test, y_test
    global tabular_train, tabular_valid, tabular_test
    tabular_train = None
    tabular_valid = None
    tabular_test = None
    # load your data as before
    print("🚀 Loading data from .h5 files...")
    path = "dataScripts/"
    with h5py.File(path + "embeddings/10k-corrected/pig_embeddings_18_500_0_corrected.h5", "r") as f:
        X_train = f["train_X_embeddings"][:]
        X_valid = f["valid_X_embeddings"][:]
        X_test = f["test_X_embeddings"][:]

    y_path = "data/pig/out/10k-corrected/"
    with h5py.File(y_path + "train.h5", "r") as f:
        y_train = f["flags"][:]
    with h5py.File(y_path + "valid.h5", "r") as f:
        y_valid = f["flags"][:1000]
    with h5py.File(y_path + "test.h5", "r") as f:
        y_test = f["flags"][:1000]

    # Reshape if needed
    if X_train.ndim == 2 and X_train.shape[1] == WINDOWS * 768:
        X_train = X_train.reshape(-1, WINDOWS, 768)
        X_valid = X_valid.reshape(-1, WINDOWS, 768)
        X_test = X_test.reshape(-1, WINDOWS, 768)

    print("After potential reshape:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # Flatten each sample => scale => reshape back
    n_tr, w_tr, f_tr = X_train.shape
    X_train_2d = X_train.reshape(n_tr, w_tr * f_tr)
    n_va, w_va, f_va = X_valid.shape
    X_valid_2d = X_valid.reshape(n_va, w_va * f_va)
    n_te, w_te, f_te = X_test.shape
    X_test_2d = X_test.reshape(n_te, w_te * f_te)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_valid_2d = scaler.transform(X_valid_2d)
    X_test_2d = scaler.transform(X_test_2d)

    X_train = X_train_2d.reshape(n_tr, w_tr, f_tr)
    X_valid = X_valid_2d.reshape(n_va, w_va, f_va)
    X_test = X_test_2d.reshape(n_te, w_te, f_te)

    global trial_no
    trial_no = 0
    with open("analysis/run_data/" + STUDY_NAME + ".csv", 'w') as f:
        f.write(
            "trial_no,ACC,PREC,REC,TNR,NPV,epochs,batch_size,lr,weight_decay,conv_channels,kernel_size,fc_units,num_blocks,PPV_factor,TPR_factor,TNR_factor,NPV_factor,threshold\n")
    storage = "sqlite:///{}.db".format("Parameters")
    study = optuna.create_study(storage=storage, study_name=STUDY_NAME)
    # study.optimize(objective, n_trials=10)
    study.optimize(objective_early_stopping, n_trials=100)
    # study.optimize(eval_final, n_trials=1)
    run_server(storage, host="localhost", port=8080)

    # print(f"\nFinal Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
