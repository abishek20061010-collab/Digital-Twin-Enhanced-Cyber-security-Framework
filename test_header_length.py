import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def get_labels_from_filename(filename):
    basename = os.path.basename(filename)
    if "Benign" in basename:
        return "Benign"
    elif "Spoofing" in basename:
        return "Spoofing"
    elif "Recon" in basename:
        return "Recon"
    elif "MQTT" in basename:
        return "MQTT"
    elif "DDoS" in basename:
        return "DDoS"
    elif "DoS" in basename:
        return "DoS"
    else:
        return "Unknown"

def main():
    directory = r'E:\Digital Twin'
    file_pattern = os.path.join(directory, '*_train.pcap.csv')
    csv_files = glob.glob(file_pattern)
    
    print(f"Loading {len(csv_files)} files...")
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        category = get_labels_from_filename(file)
        df['category'] = category
        df_list.append(df)
        
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows before sampling: {len(df)}")
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)
    print(f"Total rows after sampling: {len(df)}")
    
    # 1. Clip negative IAT values to 0.0
    print("Clipping negative IAT values...")
    if 'IAT' in df.columns:
        df['IAT'] = df['IAT'].clip(lower=0.0)
    
    # Get numeric features (exclude 'category')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Header_Length' not in numeric_cols:
        print("Header_Length not found!")
        return

    # 2. Create two versions:
    # Version (a): With Header_Length (capped 95th percentile, log1p)
    # Version (b): Without Header_Length
    
    df_a = df.copy()
    
    print("Preparing version (a) - With Header_Length (capped & log1p)...")
    header_95 = df_a['Header_Length'].quantile(0.95)
    df_a['Header_Length'] = df_a['Header_Length'].clip(upper=header_95)
    df_a['Header_Length'] = np.log1p(df_a['Header_Length'])
    
    features_a = numeric_cols
    features_b = [c for c in numeric_cols if c != 'Header_Length']
    
    X_a = df_a[features_a].copy()
    X_b = df_a[features_b].copy()
    y = df_a['category']
    
    # 3. Switch to RobustScaler
    print("Scaling with RobustScaler...")
    scaler_a = RobustScaler()
    X_a_scaled = scaler_a.fit_transform(X_a)
    
    scaler_b = RobustScaler()
    X_b_scaled = scaler_b.fit_transform(X_b)
    
    print("Splitting data...")
    X_train_a, X_test_a, y_train, y_test = train_test_split(X_a_scaled, y, test_size=0.2, random_state=42, stratify=y)
    X_train_b, X_test_b, _, _ = train_test_split(X_b_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train Random Forest
    print("Training Random Forest on Version (a)...")
    rf_a = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_a.fit(X_train_a, y_train)
    preds_a = rf_a.predict(X_test_a)
    
    print("Training Random Forest on Version (b)...")
    rf_b = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_b.fit(X_train_b, y_train)
    preds_b = rf_b.predict(X_test_b)
    
    # 5. Compare per-class F1 scores
    print("\n" + "="*50)
    print("PER-CLASS F1 SCORES: Version (a) [WITH Header_Length]")
    print("="*50)
    print(classification_report(y_test, preds_a, digits=4))
    
    print("\n" + "="*50)
    print("PER-CLASS F1 SCORES: Version (b) [WITHOUT Header_Length]")
    print("="*50)
    print(classification_report(y_test, preds_b, digits=4))
    
    # Print Feature Importances for Version (a)
    print("\n" + "="*50)
    print("FEATURE IMPORTANCES: Version (a)")
    print("="*50)
    importances = rf_a.feature_importances_
    imp_df = pd.DataFrame({'Feature': features_a, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    imp_df.index = np.arange(1, len(imp_df) + 1)
    print(imp_df.to_string())
    
    hl_row = imp_df[imp_df['Feature'] == 'Header_Length']
    if not hl_row.empty:
        rank = hl_row.index[0]
        imp = hl_row['Importance'].values[0]
        print(f"\n--> Header_Length is ranked {rank} out of {len(features_a)} features (Importance: {imp:.4f})")

if __name__ == "__main__":
    main()
