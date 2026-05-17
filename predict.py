import torch
import pickle
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from model import LTSM
from data_process import Standardizer

checkpoint = "ep_16_st_13160.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.load("data/processed/x_train.pt").to(device)
y_train = torch.load("data/processed/y_train.pt").to(device)

X_val = torch.load("data/processed/x_val.pt").to(device)
y_val = torch.load("data/processed/y_val.pt").to(device)

print("Loaded dataset")

dataset = TensorDataset(X_val, y_val)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LTSM(input_size=21, hidden_dim=512, output_size=14, num_layers=8, dropout=0.1)
model.load_state_dict(torch.load(f"models/{checkpoint}"))
model = model.to(device)
print("Loaded model")

with open("data/processed/standardizer.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("data/processed/target_standardizer.pkl", "rb") as f:
    scaler_features = pickle.load(f)
print("Loaded scalers")

steps = 16#int(input("Steps: "))
last_tensor_x = X_train[-1][-1].cpu().numpy()
last_tensor_y = y_train[-1].cpu().numpy()

kaunas = last_tensor_x[14]
klaipeda = last_tensor_x[15]
vilnius = last_tensor_x[16]

cities = torch.tensor([kaunas, klaipeda, vilnius]).to(device)
window = X_train[-1].clone().to(device)

print("---")
with torch.no_grad():
    for step in range(steps):
        pred = model(window)
        last_row = window[-1].clone()

        hour = 24 * (np.atan2(last_row[10].item(), last_row[11].item()) % (2*np.pi)) / (2*np.pi)
        doy  = 365 * (np.atan2(last_row[12].item(), last_row[13].item()) % (2*np.pi)) / (2*np.pi)

        next_hour = (hour + 1) % 24
        next_doy = (doy + (1 if next_hour < hour else 0)) % 365

        next_h_sin = np.sin(2 * np.pi * next_hour / 24)
        next_h_cos = np.cos(2 * np.pi * next_hour / 24)
        next_d_sin = np.sin(2 * np.pi * next_doy / 365)
        next_d_cos = np.cos(2 * np.pi * next_doy / 365)

        next_row = last_row.clone()
        next_row[:10] = pred[:10]
        next_row[10] = next_h_sin
        next_row[11] = next_h_cos
        next_row[12] = next_d_sin
        next_row[13] = next_d_cos

        window = torch.cat([window[1:], next_row.unsqueeze(0)], dim=0)
        unscaled_last_row = scaler.inverse_transform(last_row[:10].cpu())
        unscaled_pred = scaler.inverse_transform(pred[:10].cpu())

        print(f"hour: {hour}")
        print(f"temp. pred.: {unscaled_pred[0].item():.2f} °C")
        # print(f"actual temp.: {unscaled_last_row[0].item():.2f} °C")
        # print(f"diff.: {np.abs(unscaled_pred[0].item() - unscaled_last_row[0].item()):.2f} °C")
        print("---")

        # h_sin = last_tensor_x[10]
        # h_cos = last_tensor_x[11]
        # d_sin = last_tensor_x[12]
        # d_cos = last_tensor_x[13]

        # y = model(X_train[-1])

        # listed = list(y.cpu().numpy())
        # listed.insert(10, h_sin)
        # listed.insert(11, h_cos)
        # listed.insert(12, d_sin)
        # listed.insert(13, d_cos)

        # reshaped = torch.tensor(listed).to(device)
        # processed = torch.cat((reshaped, cities))
        # X_train[-1].add(processed)

        # npc = processed.cpu().numpy()
        # features = scaler_features.inverse_transform(y.cpu().numpy())

        # raw_features = list(last_tensor_x)
        # inputs = scaler.inverse_transform(torch.tensor(raw_features[:10]))

        # hour = 24 * (np.atan2(h_sin, h_cos) % (2 * np.pi)) / (2 * np.pi) + 1

        # print("hour: " + str(hour))
        # print("prediction: " + str(y.cpu().numpy()[0].item()))
        # print("temp.: " + str(features[0].item()))
        # print("correct: " + str(last_tensor_y[0].item()))
        # print("correct temp. " + str(scaler_features.inverse_transform(last_tensor_y)[0].item()))
        # print("___")

        # last_tensor_x = processed.cpu().numpy()