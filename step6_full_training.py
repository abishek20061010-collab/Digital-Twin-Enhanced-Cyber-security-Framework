import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping

# 1. Reproducibility
SEED = 42
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
class F1LogCallback(Callback):
    def __init__(self, validation_data, label_encoder):
        super().__init__()
        self.val_data = validation_data
        self.le = label_encoder
        
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val_cat = self.val_data
        y_val_true = np.argmax(y_val_cat, axis=1)
        y_val_pred_probs = self.model.predict(X_val, verbose=0)
        y_val_pred = np.argmax(y_val_pred_probs, axis=1)
        
        f1_scores = f1_score(y_val_true, y_val_pred, average=None)
        
        print(f"\n--- Epoch {epoch+1} Val F1 Scores ---")
        for idx, f1 in enumerate(f1_scores):
            print(f"  {self.le.classes_[idx]}: {f1:.4f}")
        print("-------------------------------")

def build_lstm(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, activation='tanh'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model_name, y_true, y_pred, train_time, infer_time, classes):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    metrics = {
        'Model': model_name,
        'Train Time (s)': train_time,
        'Inference Time (s)': infer_time,
        'Macro F1': report['macro avg']['f1-score'],
        'Weighted F1': report['weighted avg']['f1-score'],
    }
    for c in classes:
        metrics[f'{c} Precision'] = report[c]['precision']
        metrics[f'{c} Recall'] = report[c]['recall']
        metrics[f'{c} F1'] = report[c]['f1-score']
        
    return metrics

def run_evaluation():
    set_seeds()
    print(f"Random seed set to {SEED} for reproducibility across os, random, np, tf.")
    
    directory = r'E:\Digital Twin\preprocessed_data'
    train_path = os.path.join(directory, 'train.csv')
    val_path = os.path.join(directory, 'val.csv')
    
    print("Loading datasets (test.csv is explicitly ignored)...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    features = [c for c in df_train.columns if c not in ['category', 'attack_type']]
    
    X_train = df_train[features].values.astype(np.float32)
    y_train_str = df_train['category'].values
    
    X_val = df_val[features].values.astype(np.float32)
    y_val_str = df_val['category'].values
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_val = le.transform(y_val_str)
    classes = le.classes_
    
    print("Computing class weights ('balanced')...")
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights_arr))
    
    all_results = []
    
    # ---------------------------------------------------------
    # PART 1: Single Train/Val Split Evaluation
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("PART 1: SINGLE TRAIN/VAL EVALUATION")
    print("="*50)
    
    # SGD
    print("\nTraining SGDClassifier...")
    sgd = SGDClassifier(loss='hinge', class_weight='balanced', random_state=SEED, n_jobs=-1)
    t0 = time.time()
    sgd.fit(X_train, y_train)
    t_train = time.time() - t0
    t0 = time.time()
    preds = sgd.predict(X_val)
    t_infer = time.time() - t0
    res = evaluate_model("SGD (Single)", y_val, preds, t_train, t_infer, classes)
    all_results.append(res)
    
    # Random Forest
    print("\nTraining RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    t0 = time.time()
    preds = rf.predict(X_val)
    t_infer = time.time() - t0
    res = evaluate_model("RF (Single)", y_val, preds, t_train, t_infer, classes)
    all_results.append(res)
    
    # LSTM
    print("\nTraining LSTM...")
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    
    set_seeds()
    lstm = build_lstm((1, X_train.shape[1]), len(classes))
    f1_callback = F1LogCallback((X_val_lstm, y_val_cat), le)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    t0 = time.time()
    lstm.fit(X_train_lstm, y_train_cat, epochs=15, batch_size=256, 
             validation_data=(X_val_lstm, y_val_cat),
             class_weight=class_weights_dict, 
             callbacks=[f1_callback, early_stop], verbose=1)
    t_train = time.time() - t0
    t0 = time.time()
    preds_prob = lstm.predict(X_val_lstm, batch_size=256)
    preds = np.argmax(preds_prob, axis=1)
    t_infer = time.time() - t0
    res = evaluate_model("LSTM (Single)", y_val, preds, t_train, t_infer, classes)
    all_results.append(res)
    
    # ---------------------------------------------------------
    # PART 2: 5-Fold Cross Validation on train.csv
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("PART 2: 5-FOLD CROSS VALIDATION ON TRAIN.CSV")
    print("="*50)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold = 1
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        print(f"\n--- FOLD {fold} ---")
        X_kf_train, X_kf_val = X_train[train_idx], X_train[val_idx]
        y_kf_train, y_kf_val = y_train[train_idx], y_train[val_idx]
        
        # SGD
        print(f"Training SGDClassifier (Fold {fold})...")
        sgd = SGDClassifier(loss='hinge', class_weight='balanced', random_state=SEED, n_jobs=-1)
        t0 = time.time()
        sgd.fit(X_kf_train, y_kf_train)
        t_train = time.time() - t0
        t0 = time.time()
        preds = sgd.predict(X_kf_val)
        t_infer = time.time() - t0
        res = evaluate_model(f"SGD (CV Fold {fold})", y_kf_val, preds, t_train, t_infer, classes)
        all_results.append(res)
        
        # Random Forest
        print(f"Training RandomForest (Fold {fold})...")
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1)
        t0 = time.time()
        rf.fit(X_kf_train, y_kf_train)
        t_train = time.time() - t0
        t0 = time.time()
        preds = rf.predict(X_kf_val)
        t_infer = time.time() - t0
        res = evaluate_model(f"RF (CV Fold {fold})", y_kf_val, preds, t_train, t_infer, classes)
        all_results.append(res)
        
        # LSTM
        print(f"Training LSTM (Fold {fold})...")
        set_seeds() # Fresh model init
        lstm = build_lstm((1, X_train.shape[1]), len(classes))
        
        X_kf_train_lstm = X_kf_train.reshape((X_kf_train.shape[0], 1, X_kf_train.shape[1]))
        X_kf_val_lstm = X_kf_val.reshape((X_kf_val.shape[0], 1, X_kf_val.shape[1]))
        y_kf_train_cat = to_categorical(y_kf_train)
        y_kf_val_cat = to_categorical(y_kf_val)
        
        # Sanity check: Initial loss before training
        if fold <= 2:
            initial_loss, _ = lstm.evaluate(X_kf_val_lstm, y_kf_val_cat, verbose=0, batch_size=256)
            print(f"Sanity Check (Fold {fold}): Initial LSTM Validation Loss (Untrained) = {initial_loss:.4f}")
            
        f1_callback = F1LogCallback((X_kf_val_lstm, y_kf_val_cat), le)
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        t0 = time.time()
        lstm.fit(X_kf_train_lstm, y_kf_train_cat, epochs=15, batch_size=256, 
                 validation_data=(X_kf_val_lstm, y_kf_val_cat),
                 class_weight=class_weights_dict, 
                 callbacks=[f1_callback, early_stop], verbose=1)
        t_train = time.time() - t0
        t0 = time.time()
        preds_prob = lstm.predict(X_kf_val_lstm, batch_size=256)
        preds = np.argmax(preds_prob, axis=1)
        t_infer = time.time() - t0
        res = evaluate_model(f"LSTM (CV Fold {fold})", y_kf_val, preds, t_train, t_infer, classes)
        all_results.append(res)
        
        fold += 1
        
    # Aggregate and save
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(directory, 'baseline_model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "="*50)
    print("FINAL SUMMARY (All Runs)")
    print("="*50)
    print(results_df[['Model', 'Train Time (s)', 'Inference Time (s)', 'Macro F1', 'Weighted F1']].to_string())

if __name__ == "__main__":
    run_evaluation()
