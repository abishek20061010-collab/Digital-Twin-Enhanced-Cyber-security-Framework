import pandas as pd
import numpy as np
import time
import os
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical

def build_lstm(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, activation='tanh'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    directory = r'E:\Digital Twin\preprocessed_data'
    train_path = os.path.join(directory, 'train.csv')
    
    print("Loading 10,000 rows from train.csv for timing test...")
    df = pd.read_csv(train_path)
    
    from sklearn.model_selection import train_test_split
    # Stratified sample of 10,000 rows
    _, df_sample = train_test_split(df, test_size=10000, random_state=42, stratify=df['category'])
    
    print(f"Actual sample size: {len(df_sample)}")
    
    # Features and labels
    exclude = ['category', 'attack_type']
    features = [c for c in df_sample.columns if c not in exclude]
    
    X = df_sample[features].values.astype(np.float32)
    y_str = df_sample['category'].values
    
    le = LabelEncoder()
    y_int = le.fit_transform(y_str)
    
    print("\n--- Model 1: Linear SVM (SGDClassifier) ---")
    svm = SGDClassifier(loss='hinge', random_state=42, n_jobs=-1)
    start_time = time.time()
    svm.fit(X, y_int)
    svm_time = time.time() - start_time
    print(f"Training time (10k rows): {svm_time:.4f} seconds")
    
    print("\n--- Model 2: Random Forest (n=100) ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    start_time = time.time()
    rf.fit(X, y_int)
    rf_time = time.time() - start_time
    print(f"Training time (10k rows): {rf_time:.4f} seconds")
    
    print("\n--- Model 3: LSTM/Dense Network ---")
    # Reshape X for LSTM (samples, timesteps, features) -> timesteps=1
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
    y_cat = to_categorical(y_int)
    
    lstm_model = build_lstm((1, X.shape[1]), len(le.classes_))
    start_time = time.time()
    # 5 epochs for the timing test to get a per-epoch estimate
    lstm_model.fit(X_lstm, y_cat, epochs=5, batch_size=256, verbose=1)
    lstm_time = time.time() - start_time
    print(f"Training time (10k rows, 5 epochs): {lstm_time:.4f} seconds")
    
    print("\n=== Extrapolations for 1.2M rows ===")
    multiplier = len(df) / len(df_sample)
    print(f"Data scale multiplier: {multiplier:.2f}x")
    
    # Simple linear extrapolation (often underestimates tree building but gives a ballpark)
    print(f"Linear SVM est. full train: {svm_time * multiplier / 60:.2f} minutes")
    print(f"Random Forest est. full train: {rf_time * multiplier / 60:.2f} minutes")
    print(f"LSTM (5 epochs) est. full train: {lstm_time * multiplier / 60:.2f} minutes")

if __name__ == "__main__":
    main()
