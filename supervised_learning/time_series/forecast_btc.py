import numpy as np
import tensorflow as tf

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Train/validation split (time-based)
split = int(len(X) * 0.8)

X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# tf.data pipelines
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(64).prefetch(tf.data.AUTOTUNE)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(1440, 6)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Save model
model.save("btc_forecaster.keras")