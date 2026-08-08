import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import mutual_info_score
from collections import defaultdict

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
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)
        df['category'] = get_labels_from_filename(file)
        df_list.append(df)
        
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows: {len(df)}\n")
    
    extra_cols = ['Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']
    
    print("="*60)
    print("1. OVERALL UNIQUE VALUE COUNTS")
    print("="*60)
    for col in extra_cols:
        if col in df.columns:
            n_unique = df[col].nunique()
            print(f"{col:15s} : {n_unique} unique values")
        else:
            print(f"{col} NOT FOUND")
            
    print("\n" + "="*60)
    print("2. PER-FILE MIN, MAX, UNIQUE VALUE COUNTS")
    print("="*60)
    
    for col in extra_cols:
        if col not in df.columns: continue
        print(f"\n--- Column: {col} ---")
        
        # Group by source_file
        grouped = df.groupby('source_file')[col]
        agg_df = grouped.agg(
            min_val='min',
            max_val='max',
            unique_count='nunique'
        )
        print(agg_df.to_string())
        
    print("\n" + "="*60)
    print("3. RELATIONSHIP TO SOURCE_FILE AND CATEGORY")
    print("="*60)
    
    # We will check how perfectly a column predicts source_file or category
    for col in extra_cols:
        if col not in df.columns: continue
        
        # Check if the column is constant per source file
        # If unique_count == 1 for all files, it's a constant per file
        grouped_sf = df.groupby('source_file')[col].nunique()
        constant_files = (grouped_sf == 1).sum()
        total_files = len(grouped_sf)
        print(f"{col:15s} : Constant in {constant_files} out of {total_files} files.")
        
        # Cross tabulation check for category
        grouped_cat = df.groupby('category')[col].nunique()
        print(f"  Unique values per category for {col}:")
        for cat, count in grouped_cat.items():
            print(f"    {cat:10s} : {count}")
            
        # Optional: Normalized Mutual Information
        # Using string representation to handle continuous values as distinct categories if there are very few
        n_unique = df[col].nunique()
        if n_unique < 100:
            mi_sf = mutual_info_score(df['source_file'], df[col].astype(str))
            mi_cat = mutual_info_score(df['category'], df[col].astype(str))
            # Normalize by target entropy
            sf_entropy = mutual_info_score(df['source_file'], df['source_file'])
            cat_entropy = mutual_info_score(df['category'], df['category'])
            
            sf_nmi = mi_sf / sf_entropy if sf_entropy > 0 else 0
            cat_nmi = mi_cat / cat_entropy if cat_entropy > 0 else 0
            
            print(f"  Predictive Power (NMI) -> source_file: {sf_nmi:.2f}, category: {cat_nmi:.2f}")

if __name__ == "__main__":
    main()
