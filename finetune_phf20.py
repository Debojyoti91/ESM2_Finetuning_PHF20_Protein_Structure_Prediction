import os
import torch
import esm
import pickle
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from utils import kabsch_align, save_predicted_pdb

class EmbeddingRegressionDataset(Dataset):
    """
    Dataset to map ESM-2 embeddings to 3D Cα coordinates.
    """
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), \
               torch.tensor(self.targets[idx], dtype=torch.float32)

class RegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def extract_esm2_embeddings(sequence, model, alphabet, device):
    """
    Runs the ESM model to extract residue-wise embeddings.
    """
    batch_converter = alphabet.get_batch_converter()
    data = [("PHF20", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
    # Residue embeddings, excluding [CLS] and [EOS]
    token_representations = results["representations"][33][0, 1:-1]
    return token_representations.cpu().numpy()

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    mse = mean_squared_error(targets, preds)
    return mse, preds, targets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ESM2 model
    print("Loading ESM-2 model...")
    model_name = "esm2_t33_650M_UR50D"
    model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
    model.eval().to(device)

    # Load preprocessed sequence and coordinates
    with open("data/processed_embeddings.pkl", "rb") as f:
        df = pickle.load(f)

    sequence = df["sequence"].iloc[0]
    ca_coords = df["coordinates"].iloc[0]
    n_residues = ca_coords.shape[0]

    # Extract residue-wise embeddings using ESM2
    print("Extracting ESM-2 embeddings...")
    embeddings = extract_esm2_embeddings(sequence, model, alphabet, device)

    if embeddings.shape[0] != n_residues:
        raise ValueError("Mismatch in number of residues between sequence and coordinates.")

    # Flatten 3D coordinates for regression target
    X = embeddings
    y = ca_coords.reshape((n_residues, 3))

    dataset = EmbeddingRegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    reg_head = RegressionHead(input_dim=X.shape[1], output_dim=3).to(device)
    optimizer = torch.optim.Adam(reg_head.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    print("Training regression head...")
    for epoch in range(100):
        loss = train(reg_head, dataloader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

    # Final prediction
    reg_head.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred = reg_head(X_tensor).cpu().numpy()

    # Align prediction to reference
    aligned_pred = kabsch_align(y_pred, y)

    # Save predicted structure as PDB
    save_predicted_pdb(aligned_pred, sequence, output_path="outputs/predicted_structure.pdb")
    print("Predicted structure saved to outputs/predicted_structure.pdb")

    # Optionally save metrics
    np.save("outputs/raw_prediction.npy", y_pred)
    np.save("outputs/aligned_prediction.npy", aligned_pred)

if __name__ == "__main__":
    main()

