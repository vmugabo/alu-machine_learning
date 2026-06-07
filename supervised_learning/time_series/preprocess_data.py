import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load datasets
coinbase = pd.read_csv("coinbaseUSD_1-min_data.csv")
bitstamp = pd.read_csv("bitstampUSD_1-min_data.csv")

# Combine
df = pd.concat([coinbase, bitstamp], ignore_index=True)

# Clean
df = df.dropna()

# Select features
features = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume_(BTC)",
        "Weighted_Price"
    ]
]

# Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

# Save scaler
joblib.dump(scaler, "btc_scaler.pkl")

# Sequence length: 24 hours (1-min intervals)
WINDOW = 1440

X, y = [], []

for i in range(len(scaled) - WINDOW):
    X.append(scaled[i:i + WINDOW])
    y.append(scaled[i + WINDOW][3])  # Close price

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Save arrays
np.save("X.npy", X)
np.save("y.npy", y)

print("X shape:", X.shape)
print("y shape:", y.shape)