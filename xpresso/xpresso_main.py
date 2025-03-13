import os, sys, gzip, h5py
import numpy as np
import pandas as pd
from scipy import stats
from functools import partial
from mimetypes import guess_type
from Bio import SeqIO

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

##############################################################################
# One-hot encoding function
##############################################################################
def one_hot(seq_list):
    """
    Converts a list of DNA sequences into a 3D NumPy array with shape:
      (num_seqs, seq_len, 4)
    """
    num_seqs = len(seq_list)
    seq_len = len(seq_list[0])
    seqindex = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
                'a': 0, 'c': 1, 'g': 2, 't': 3}
    seq_vec = np.zeros((num_seqs, seq_len, 4), dtype=np.float32)
    for i, seq in enumerate(seq_list):
        for j, base in enumerate(seq):
            if base in seqindex:
                seq_vec[i, j, seqindex[base]] = 1.0
    return seq_vec

##############################################################################
# PyTorch model replicating the “best identified” TF architecture:
#   Conv1d(128, kernel=6, dilation=1) -> MaxPool1d(30)
#   Conv1d(32, kernel=9, dilation=1) -> MaxPool1d(10)
#   Flatten; then concatenate the halflife features;
#   Dense(64) -> dropout -> ReLU -> Dense(2) -> dropout -> ReLU -> Dense(1)
#
# Note: The second dense layer is small (size 2) to keep parity.
##############################################################################
class BestConvModel(nn.Module):
    def __init__(self, dropout1=0.00099, dropout2=0.01546):
        super(BestConvModel, self).__init__()
        # Convolution blocks
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=6,
                               dilation=1, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=30)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=9,
                               dilation=1, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=10)
        # Dense layers – fc1’s in_features will be set dynamically.
        self.fc1 = nn.Linear(in_features=0, out_features=64)  # placeholder
        self.dropout1 = nn.Dropout(p=dropout1)
        self.fc2 = nn.Linear(64, 2)
        self.dropout2 = nn.Dropout(p=dropout2)
        self.out = nn.Linear(2, 1)
        self.relu = nn.ReLU()

    def set_fc1_infeatures(self, in_size):
        """Reinitialize fc1 with the proper input dimension."""
        out_features = self.fc1.out_features
        self.fc1 = nn.Linear(in_size, out_features)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0.)

    def forward(self, promoter_x, halflife_x):
        # promoter_x shape: (batch, length, 4); we need (batch, 4, length)
        promoter_x = promoter_x.permute(0, 2, 1)
        # Block 1
        x = self.conv1(promoter_x)
        x = self.relu(x)
        x = self.pool1(x)
        # Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        # Flatten and then concatenate the halflife features
        x = x.view(x.size(0), -1)
        x = torch.cat((x, halflife_x), dim=1)
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.out(x)
        return x

##############################################################################
# Training function with early stopping.
##############################################################################
def train_model(model, train_loader, valid_loader, device, lr=0.0005, momentum=0.9,
                n_epochs=100, patience=7):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state_dict = None
    train_history = []
    valid_history = []
    print("Model is on device:", next(model.parameters()).device)
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_epoch = 0.0
        for (p_batch, h_batch, y_batch) in train_loader:
            p_batch, h_batch, y_batch = p_batch.to(device), h_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(p_batch, h_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * p_batch.size(0)
        train_loss_epoch /= len(train_loader.dataset)
        model.eval()
        valid_loss_epoch = 0.0
        with torch.no_grad():
            for (p_batch, h_batch, y_batch) in valid_loader:
                p_batch, h_batch, y_batch = p_batch.to(device), h_batch.to(device), y_batch.to(device)
                outputs = model(p_batch, h_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                valid_loss_epoch += loss.item() * p_batch.size(0)
        valid_loss_epoch /= len(valid_loader.dataset)
        train_history.append(train_loss_epoch)
        valid_history.append(valid_loss_epoch)
        print(f"Epoch [{epoch}/{n_epochs}] Train MSE: {train_loss_epoch:.4f} | Val MSE: {valid_loss_epoch:.4f}")
        if valid_loss_epoch < best_val_loss:
            best_val_loss = valid_loss_epoch
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model, train_history, valid_history

##############################################################################
# Main training function.
#
# For training, the promoter sequences (from the HDF5 file) are 20,000 nt long.
# We slice them using leftpos=3000 and rightpos=13500 to get a 10,500-nt input.
##############################################################################
def main(datadir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    params = {
        'batchsize': 128,
        'leftpos': 3000,
        'rightpos': 13500,
        'dropout1': 0.00099,
        'dropout2': 0.01546,
    }
    # Load training data
    with h5py.File(os.path.join(datadir, 'train.h5'), 'r') as f:
        X_trainhalflife = np.array(f['data'])
        X_trainpromoter = np.array(f['promoter'])  # shape (N, 20000, 4)
        y_train = np.array(f['label']).astype(np.float32)
    with h5py.File(os.path.join(datadir, 'valid.h5'), 'r') as f:
        X_validhalflife = np.array(f['data'])
        X_validpromoter = np.array(f['promoter'])
        y_valid = np.array(f['label']).astype(np.float32)
    with h5py.File(os.path.join(datadir, 'test.h5'), 'r') as f:
        X_testhalflife = np.array(f['data'])
        X_testpromoter = np.array(f['promoter'])
        y_test = np.array(f['label']).astype(np.float32)
        geneName_test = np.array(f['geneName'])
    # Slice promoter sequences to 10,500 nt for training/validation/test
    leftpos = params['leftpos']
    rightpos = params['rightpos']
    X_trainpromoterSubseq = X_trainpromoter[:, leftpos:rightpos, :]
    X_validpromoterSubseq = X_validpromoter[:, leftpos:rightpos, :]
    X_testpromoterSubseq = X_testpromoter[:, leftpos:rightpos, :]
    # Build and initialize model
    model = BestConvModel(dropout1=params['dropout1'], dropout2=params['dropout2'])
    halflife_dim = X_trainhalflife.shape[1]
    model = model.to(device)
    with torch.no_grad():
        dummy_promoter = torch.zeros((1, rightpos - leftpos, 4), dtype=torch.float32).to(device)
        dummy_halflife = torch.zeros((1, halflife_dim), dtype=torch.float32).to(device)
        x = dummy_promoter.permute(0, 2, 1)
        x = model.conv1(x)
        x = model.relu(x)
        x = model.pool1(x)
        x = model.conv2(x)
        x = model.relu(x)
        x = model.pool2(x)
        x = x.view(x.size(0), -1)
        flatten_dim = x.size(1) + halflife_dim
    model.set_fc1_infeatures(flatten_dim)
    model = model.to(device)
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_trainpromoterSubseq, dtype=torch.float32),
        torch.tensor(X_trainhalflife, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    valid_dataset = TensorDataset(
        torch.tensor(X_validpromoterSubseq, dtype=torch.float32),
        torch.tensor(X_validhalflife, dtype=torch.float32),
        torch.tensor(y_valid, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_testpromoterSubseq, dtype=torch.float32),
        torch.tensor(X_testhalflife, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=params['batchsize'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batchsize'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Train with early stopping
    model, train_history, valid_history = train_model(model, train_loader, valid_loader, device,
                                                      lr=0.0005, momentum=0.9, n_epochs=100, patience=7)
    # Evaluate on test set
    model.eval()
    preds_test = []
    y_test_vals = []
    with torch.no_grad():
        for (p_batch, h_batch, y_batch) in test_loader:
            p_batch, h_batch = p_batch.to(device), h_batch.to(device)
            outputs = model(p_batch, h_batch)
            preds_test.extend(outputs.cpu().numpy().flatten())
            y_test_vals.extend(y_batch.numpy().flatten())
    preds_test = np.array(preds_test)
    y_test_vals = np.array(y_test_vals)
    slope, intercept, r_value, p_value, std_err = stats.linregress(preds_test, y_test_vals)
    print(f"Test R^2 = {r_value ** 2:.3f}")
    # Save predictions and model state
    df_out = pd.DataFrame({
        'Gene': geneName_test,
        'Pred': preds_test,
        'Actual': y_test_vals
    })
    df_out.to_csv(os.path.join(datadir, 'predictions_pytorch.txt'), sep='\t', index=False)
    torch.save(model.state_dict(), os.path.join(datadir, 'bestparams_pytorch.pt'))
    print("Saved PyTorch model to", os.path.join(datadir, 'bestparams_pytorch.pt'))

##############################################################################
# Function to generate predictions from a saved model and a FASTA file.
#
# In the TF setup the FASTA file already has 10,500-nt sequences.
# Therefore, we do NOT slice the input sequences here.
##############################################################################
def generate_predictions(model_file, input_file, output_file, halflife_dim=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    # Build the same model architecture
    model = BestConvModel()
    # Create a dummy pass – here FASTA sequences are assumed to be 10,500 nt long.
    promoter_length = 10500
    model = model.to(device)
    with torch.no_grad():
        dummy_promoter = torch.zeros((1, promoter_length, 4), dtype=torch.float32).to(device)
        dummy_halflife = torch.zeros((1, halflife_dim), dtype=torch.float32).to(device)
        x = dummy_promoter.permute(0, 2, 1)
        x = model.conv1(x)
        x = model.relu(x)
        x = model.pool1(x)
        x = model.conv2(x)
        x = model.relu(x)
        x = model.pool2(x)
        x = x.view(x.size(0), -1)
        flatten_dim = x.size(1) + halflife_dim
    model.set_fc1_infeatures(flatten_dim)
    model = model.to(device)
    # Load the trained weights
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    # Determine file encoding for FASTA input (gzip or plain text)
    encoding = guess_type(input_file)[1]
    _open = open if (encoding is None) else partial(gzip.open, mode='rt')
    names = []
    sequences = []
    predictions = []
    batch_size = 32
    with _open(input_file) as f:
        i = 0
        for fasta in SeqIO.parse(f, 'fasta'):
            name = fasta.id
            seq_str = str(fasta.seq)
            # Expect each sequence to be exactly 10500 nt long.
            if len(seq_str) != promoter_length:
                sys.exit(f"Error in sequence {name}, length = {len(seq_str)}; must be 10500.")
            sequences.append(seq_str)
            names.append(name)
            i += 1
            if i % batch_size == 0:
                promoter_np = one_hot(sequences)  # shape: (bs, 10500, 4)
                halflife_batch = np.zeros((len(sequences), halflife_dim), dtype=np.float32)
                with torch.no_grad():
                    p_torch = torch.tensor(promoter_np, dtype=torch.float32).to(device)
                    h_torch = torch.tensor(halflife_batch, dtype=torch.float32).to(device)
                    out = model(p_torch, h_torch).cpu().numpy().flatten()
                    predictions.extend(out.tolist())
                sequences = []
        if len(sequences) > 0:
            promoter_np = one_hot(sequences)
            halflife_batch = np.zeros((len(sequences), halflife_dim), dtype=np.float32)
            with torch.no_grad():
                p_torch = torch.tensor(promoter_np, dtype=torch.float32).to(device)
                h_torch = torch.tensor(halflife_batch, dtype=torch.float32).to(device)
                out = model(p_torch, h_torch).cpu().numpy().flatten()
                predictions.extend(out.tolist())
    df = pd.DataFrame({'ID': names, 'SCORE': predictions})
    print(df.head(10))
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Wrote predictions to {output_file}")

##############################################################################
# Entry point.
#
# For training:
#   The directory structure should be:
#     datadir/
#       train.h5, valid.h5, test.h5
#
# For prediction:
#   The input FASTA file should have sequences of length 10500.
##############################################################################
if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 2:
        print("Usage: python myfile.py [human|mouse]")
        sys.exit(1)
    dataset_type = sys.argv[1].lower()
    if dataset_type == 'human':
        datadir = os.path.join('data', 'human-dataset')
        input_fasta = os.path.join('data', 'xpresso-predict', 'input_fasta', 'human_promoters.fa')
        output_predictions = os.path.join('data', 'xpresso-predict', 'human_promoter_predictions.txt')
    elif dataset_type == 'mouse':
        datadir = os.path.join('data', 'mouse-dataset')
        input_fasta = os.path.join('data', 'xpresso-predict', 'input_fasta', 'mouse_promoters.fa')
        output_predictions = os.path.join('data', 'xpresso-predict', 'mouse_promoter_predictions.txt')
    else:
        print("Invalid dataset type. Please choose 'human' or 'mouse'.")
        sys.exit(1)
    # Train the model using the data in the selected folder.
    main(datadir)
    # Use the saved model to generate predictions on the FASTA file.
    model_file = os.path.join(datadir, 'bestparams_pytorch.pt')
    generate_predictions(model_file=model_file,
                         input_file=input_fasta,
                         output_file=output_predictions,
                         halflife_dim=8)
