import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def get_labels_from_filename(filename):
    basename = os.path.basename(filename)
    if "Benign" in basename: return "Benign"
    elif "Spoofing" in basename: return "Spoofing"
    elif "Recon" in basename: return "Recon"
    elif "MQTT" in basename: return "MQTT"
    elif "DDoS" in basename: return "DDoS"
    elif "DoS" in basename: return "DoS"
    else: return "Unknown"

def main():
    directory = r'E:\Digital Twin'
    file_pattern = os.path.join(directory, '*_train.pcap.csv')
    csv_files = glob.glob(file_pattern)
    
    print(f"Loading {len(csv_files)} files...")
    df_list = []
    for file in csv_files:
        print(f"Reading {os.path.basename(file)}...")
        df = pd.read_csv(file)
        df['category'] = get_labels_from_filename(file)
        df['attack_type'] = os.path.basename(file).replace('_train.pcap.csv', '')
        df_list.append(df)
        
    print("Concatenating...")
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    
    # 1. Drop Header_Length
    if 'Header_Length' in df.columns:
        df.drop(columns=['Header_Length'], inplace=True)
        print("1. Dropped Header_Length.")
        
    # 2. Keep 6 extra columns
    print("2. Kept extra columns: Number, Magnitue, Radius, Covariance, Variance, Weight.")
    
    # 3. Clip negative IAT values to 0.0
    if 'IAT' in df.columns:
        df['IAT'] = df['IAT'].clip(lower=0.0)
        print("3. Clipped negative IAT values to 0.0.")
        
    # 5. Encode Protocol Type
    if 'Protocol Type' in df.columns:
        print("5. Bucketing and One-hot encoding Protocol Type...")
        df['Protocol Type'] = df['Protocol Type'].round().astype(int)
        protocol_dummies = pd.get_dummies(df['Protocol Type'], prefix='Proto')
        protocol_dummies = protocol_dummies.astype('int8')
        df = pd.concat([df.drop(columns=['Protocol Type']), protocol_dummies], axis=1)
        
    # 4. Apply RobustScaler
    print("4. Applying RobustScaler to numeric features...")
    exclude_cols = ['category', 'attack_type']
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not str(c).startswith('Proto_') and c not in exclude_cols]
    
    for col in numeric_cols:
        df[col] = df[col].astype(np.float32)
        
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"   Scaled {len(numeric_cols)} numeric columns.")
    
    # 6. Stratified split (70/15/15)
    print("6. Splitting dataset (70% train, 15% val, 15% test)...")
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['category'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['category'])
    
    print(f"   Train size: {len(train_df)}")
    print(f"   Val size: {len(val_df)}")
    print(f"   Test size: {len(test_df)}")
    
    # 7. Save to disk
    output_dir = os.path.join(directory, 'preprocessed_data')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"7. Saving datasets to {output_dir} (CSV format)...")
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print("="*50)
    print("STEP 5 PREPROCESSING COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
