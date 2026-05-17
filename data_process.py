import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

SEQ_LEN = 256

class Standardizer:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X):
        return X * self.std + self.mean

def add_features(df):
    dt = pd.to_datetime(df["time"])
    hour = dt.dt.hour
    day_of_year = dt.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    df["wind_dir10_sin"] = np.sin(2 * np.pi * df["wind_direction_10m"] / 360)
    df["wind_dir10_cos"] = np.cos(2 * np.pi * df["wind_direction_10m"] / 360)

    df["wind_dir100_sin"] = np.sin(2 * np.pi * df["wind_direction_100m"] / 360)
    df["wind_dir100_cos"] = np.cos(2 * np.pi * df["wind_direction_100m"] / 360)

    return df

def make_sequences(df, feature_cols, target_cols, seq_len):
    X, y = [], []

    for _, city_df in df.groupby("location"):
        city_df = city_df.sort_values("time").reset_index(drop=True)

        features = city_df[feature_cols].to_numpy(dtype=np.float32)
        target = city_df[target_cols].to_numpy(dtype=np.float32)

        for i in range(len(city_df) - seq_len):
            X.append(features[i:i + seq_len])
            y.append(target[i + seq_len - 1])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y

if __name__ == "__main__":
    folder = Path("data/raw")
    frames = []

    for file in folder.glob("*.csv"):
        if file.stem == "merged":
            continue
        tmp = pd.read_csv(file)
        tmp["location"] = file.stem
        frames.append(tmp)

    if not frames:
        raise FileNotFoundError("No CSV files found in data/raw")

    df = pd.concat(frames, ignore_index=True)
    df = add_features(df)

    cities = {
        city: i
        for i, city in enumerate(pd.unique(df["location"]))
    }

    location_onehot = pd.get_dummies(df["location"], dtype=np.float32)
    location_onehot = location_onehot.reindex(columns=cities.keys(), fill_value=0.0)
    location_onehot.columns = [f"city_{name}" for name in location_onehot.columns]

    df = pd.concat([df, location_onehot], axis=1)
    print("Tensorized cities")

    numeric_cols = [
        "temperature",
        "humidity",
        "wind_speed_10m",
        "surface_pressure",
        "wind_speed_100m",
        "precipitation",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high"
    ]

    feature_cols = [
        "temperature", "humidity", "wind_speed_10m", "surface_pressure",
        "wind_speed_100m", "precipitation", "cloud_cover", "cloud_cover_low",
        "cloud_cover_mid", "cloud_cover_high",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "wind_dir10_sin", "wind_dir10_cos",
        "wind_dir100_sin", "wind_dir100_cos",
        "city_klaipeda", "city_vilnius", "city_kaunas"
    ]

    target_base_cols = [
        "temperature",
        "humidity",
        "wind_speed_10m",
        "surface_pressure",
        "wind_speed_100m",
        "precipitation",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "wind_dir10_sin", "wind_dir10_cos",
        "wind_dir100_sin", "wind_dir100_cos",
    ]

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["location", "time"]).reset_index(drop=True)

    for col in target_base_cols:
        df[col + "_next"] = df.groupby("location")[col].shift(-1)

    target_cols = [c + "_next" for c in target_base_cols]
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    scaler = Standardizer().fit(train_df[numeric_cols].to_numpy())
    scaler_features = Standardizer().fit(train_df[target_cols].to_numpy())

    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols].to_numpy())
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols].to_numpy())

    train_df[target_cols] = scaler_features.transform(train_df[target_cols].to_numpy())
    val_df[target_cols] = scaler_features.transform(val_df[target_cols].to_numpy())

    train_df.to_csv("_train.csv")
    print("Standardized values and saved csv")

    with open("data/processed/standardizer.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("data/processed/target_standardizer.pkl", "wb") as f:
        pickle.dump(scaler_features, f)

    X_train, y_train = make_sequences(train_df, feature_cols, target_cols=target_cols, seq_len=SEQ_LEN)
    X_val, y_val = make_sequences(val_df, feature_cols, target_cols=target_cols, seq_len=SEQ_LEN)

    torch.save(X_train, "data/processed/x_train.pt")
    torch.save(X_val, "data/processed/x_val.pt")

    torch.save(y_train, "data/processed/y_train.pt")
    torch.save(y_val, "data/processed/y_val.pt")

    print("Saved tensor values")