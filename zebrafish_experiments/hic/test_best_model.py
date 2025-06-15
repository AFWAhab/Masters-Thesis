from hic.adams_gnn_train import RegGNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RegGNN(281).to(device)

model.load_state_dict(torch.load('best_model.pt'))
model.eval()
with torch.no_grad():
    tp = model(data)[data.test_mask].cpu().numpy()
    tt = data.y[data.test_mask].cpu().numpy()
print(f"Test R²: {r2_score(tt, tp):.4f}")