import pandas as pd
import os

def check_split(filename):
    print(f"\n{'='*50}")
    print(f"Checking {os.path.basename(filename)}")
    print(f"{'='*50}")
    
    df = pd.read_csv(filename)
    
    # 1. Row and Column Count
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Class Distribution (Percentages)
    print("\n-- Category Distribution (%) --")
    print((df['category'].value_counts(normalize=True) * 100).round(2).to_string())
    
    print("\n-- Attack Type Distribution (%) --")
    print((df['attack_type'].value_counts(normalize=True) * 100).round(2).to_string())
    
    # 3. NaN Check
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    print(f"\n-- NaN Check --")
    if total_nans == 0:
        print("Total NaNs: 0 (Passed)")
    else:
        print(f"WARNING: Found NaNs in the following columns:")
        print(nan_counts[nan_counts > 0].to_string())
        
    return df

def main():
    dir_path = r'E:\Digital Twin\preprocessed_data'
    splits = ['train.csv', 'val.csv', 'test.csv']
    
    df_train = None
    for split in splits:
        file_path = os.path.join(dir_path, split)
        df = check_split(file_path)
        if split == 'train.csv':
            df_train = df
            
    # 4. List of all column names and verify Proto_ scaling
    print("\n" + "="*50)
    print("COLUMN VERIFICATION")
    print("="*50)
    columns = list(df_train.columns)
    print(f"Total columns: {len(columns)}")
    print(", ".join(columns))
    
    print("\n-- Checking for Header_Length --")
    if 'Header_Length' in columns:
        print("FAIL: Header_Length is still present.")
    else:
        print("PASS: Header_Length is gone.")
        
    print("\n-- Checking Proto_ columns (One-Hot Encoded) --")
    proto_cols = [c for c in columns if c.startswith('Proto_')]
    print(f"Found {len(proto_cols)} Proto_ columns.")
    
    # Print unique values for a few Proto_ columns to ensure they are 0/1 and not scaled
    if proto_cols:
        sample_protos = proto_cols[:5]
        for c in sample_protos:
            unique_vals = sorted(df_train[c].unique())
            print(f"{c} unique values: {unique_vals}")
            
if __name__ == "__main__":
    main()
