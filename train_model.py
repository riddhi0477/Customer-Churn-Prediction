import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dropna(inplace=True)

# Drop ID
df.drop("customerID", axis=1, inplace=True)

# Encode categorical
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Features
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# =====================
# 🔥 ML MODELS
# =====================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "NaiveBayes": GaussianNB()
}

best_acc = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    joblib.dump(model, f"models/{name}.pkl")

    print(f"{name}: {acc}")

    if acc > best_acc:
        best_acc = acc
        best_model = model

joblib.dump(best_model, "models/model.pkl")

# =====================
# 🔥 DL MODELS
# =====================

# Dense NN
dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=10)
dl_model.save("models/dl_model.h5")

# RNN + LSTM input
X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

# RNN
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(1, X_train.shape[1])),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer='adam', loss='binary_crossentropy')
rnn_model.fit(X_train_rnn, y_train, epochs=10)
rnn_model.save("models/rnn_model.h5")

# LSTM
lstm_model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1])),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
lstm_model.fit(X_train_rnn, y_train, epochs=10)
lstm_model.save("models/lstm_model.h5")

print("✅ Training complete!")