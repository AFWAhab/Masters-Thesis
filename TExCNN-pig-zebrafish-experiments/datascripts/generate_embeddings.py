import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import time

import os
os.environ["HOME"] = os.path.expanduser("~")

if __name__ == "__main__":
    print("Loading model")
    model_name = "zhihan1996/DNABERT-2-117M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Load config and explicitly disable FlashAttention
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_fp8 = False
    config.use_flash_attn = False

    # Now load the model with the updated config
    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    model = model.to(device)
    print(f"Model loaded, using device: {device}")

    data_path = "../data/pig/out/10k-corrected-half-life/"
    #data_path = "../data/zebrafish/out/"

    ### === TRAIN ===
    with h5py.File(data_path + "train.h5", 'r') as f:
        train_X_promoter = f['promoter'][:]
    train_size = train_X_promoter.shape[0]  # === MOD ===
    print(f"Train samples: {train_size}")  # === MOD ===

    train_data = np.zeros((train_size, 18, 768))
    print("Entering loop for train")
    for i in range(train_size):
        if i < 10:
            start_time = time.time()  # === MOD ===

        seq = ''
        for j in train_X_promoter[i]:
            if j[0]: seq += 'A'
            elif j[1]: seq += 'C'
            elif j[2]: seq += 'G'
            elif j[3]: seq += 'T'
            else: seq += 'N'
        for j in range(18):
            window = seq[0 + j * 500: 0 + j * 500 + 2000]
            inputs = tokenizer(window, return_tensors='pt', truncation=True)["input_ids"].to(device)
            with torch.no_grad():
                hidden_states = model(inputs)[0]
                embedding_mean = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
            train_data[i, j] = embedding_mean

        if i < 10:
            duration = time.time() - start_time  # === MOD ===
            print(f"Time for train sample {i}: {duration:.2f} seconds")  # === MOD ===
        if i % 1000 == 0:
            print("Train:", i)

    ### === VALID ===
    valid_size = 1000
    valid_data = np.zeros((valid_size, 18, 768))

    with h5py.File(data_path + "valid.h5", 'r') as f:
        valid_X_promoter = f['promoter'][:]

    print("Entering loop for valid")
    for i in range(valid_size):
        seq = ''
        for j in valid_X_promoter[i]:
            if j[0]: seq += 'A'
            elif j[1]: seq += 'C'
            elif j[2]: seq += 'G'
            elif j[3]: seq += 'T'
            else: seq += 'N'
        for j in range(18):
            window = seq[0 + j * 500: 0 + j * 500 + 2000]
            inputs = tokenizer(window, return_tensors='pt', truncation=True)["input_ids"].to(device)
            with torch.no_grad():
                hidden_states = model(inputs)[0]
                embedding_mean = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
            valid_data[i, j] = embedding_mean
        if i % 1000 == 0:
            print("Train:", i)

    ### === TEST ===
    test_size = 1000
    test_data = np.zeros((test_size, 18, 768))

    with h5py.File(data_path + "test.h5", 'r') as f:
        test_X_promoter = f['promoter'][:]

    print("Entering loop for test")
    for i in range(test_size):
        seq = ''
        for j in test_X_promoter[i]:
            if j[0]: seq += 'A'
            elif j[1]: seq += 'C'
            elif j[2]: seq += 'G'
            elif j[3]: seq += 'T'
            else: seq += 'N'
        for j in range(18):
            window = seq[0 + j * 500: 0 + j * 500 + 2000]
            inputs = tokenizer(window, return_tensors='pt', truncation=True)["input_ids"].to(device)
            with torch.no_grad():
                hidden_states = model(inputs)[0]
                embedding_mean = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
            test_data[i, j] = embedding_mean
        if i % 1000 == 0:
            print("Train:", i)

    ### === SAVE TO .h5 ===
    with h5py.File("embeddings/10k-corrected-half-life/pig_embeddings_18_500_0_corrected.h5", "w") as f:
        f.create_dataset("train_X_embeddings", data=train_data, compression="gzip")
        f.create_dataset("valid_X_embeddings", data=valid_data, compression="gzip")
        f.create_dataset("test_X_embeddings", data=test_data, compression="gzip")