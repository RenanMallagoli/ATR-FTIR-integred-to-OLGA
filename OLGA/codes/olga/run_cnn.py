import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import data_loader as dl

np.random.seed(42)
tf.random.set_seed(42)

train_path = "dataset/train/train.csv"
test_path = "dataset/test/test.csv"

print("[INFO] Carregando e limpando dados para a CNN 1D...")
X_total, y_total, wavenumbers = dl.load_and_clean_real_data(train_path, test_path)

# Separação clássica de 80% treino e 20% validação
X_train, X_val, y_train, y_val = train_test_split(
    X_total, y_total, test_size=0.20, random_state=42, stratify=y_total
)

# A CNN 1D exige uma terceira dimensão nos dados (batch, steps, channels)
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_cnn = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

def build_cnn1d(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(filters=32, kernel_size=7, strides=2, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        
        layers.Conv1D(filters=64, kernel_size=5, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.BatchNormalization(),
        
        layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ], name="CNN_1D_FTIR")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

print("\n[INFO] Treinando a CNN 1D com os dados limpos...")
cnn_model = build_cnn1d(X_train.shape[1])
history_cnn = cnn_model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

y_pred_prob = cnn_model.predict(X_val_cnn)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, zero_division=0)
rec = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)

print(f"\n--- Métricas da CNN 1D ---")
print(f"Acurácia:  {acc:.4f}")
print(f"Precisão:  {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")