import os
import numpy as np
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import joblib # type: ignore

# Folder data
DATA_DIR = "data"
MODEL_PATH = "models/gesture_model.pkl"

def load_data():
    X = []
    y = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            label = file.replace(".csv", "")
            data = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
            X.extend(data.values)
            y.extend([label] * len(data))

    return np.array(X), np.array(y)

def train_model():
    print(" Memuat data gesture...")
    X, y = load_data()
    print(f" Total data: {len(X)} sampel dari {len(set(y))} gesture.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🤖 Melatih smodel (Random Forest)...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f" Akurasi model: {acc*100:.2f}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f" Model disimpan ke: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
