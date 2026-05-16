import torch
import pickle

from torch.utils.data import TensorDataset, DataLoader
from model import LTSM
from data_process import Standardizer

checkpoint = "ep_16_st_52768.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.load("data/processed/x_train.pt")
X_val = torch.load("data/processed/x_val.pt")
y_val = torch.load("data/processed/y_val.pt")

print("Loaded dataset")

dataset = TensorDataset(X_val, y_val)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LTSM(input_size=21, hidden_dim=512, output_size=14, num_layers=8, dropout=0.1)
model.load_state_dict(torch.load(f"models/{checkpoint}"))
model = model.to(device)
print("Loaded model")

with open("data/processed/standardizer.pkl", "rb") as f:
    scaler = pickle.load(f)
print("Loaded scaler")

steps = int(input("Steps: "))
last_tensor = X_train[-1][-1]

print(last_tensor)