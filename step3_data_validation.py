import pandas as pd
import os
import glob

def main():
    directory = r'E:\Digital Twin'
    file_pattern = os.path.join(directory, '*_train.pcap.csv')
    csv_files = glob.glob(file_pattern)
    
    print("Loading files and checking column consistency...")
    
    column_sets = {}
    df_list = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        column_sets[os.path.basename(file)] = list(df.columns)
        df_list.append(df)
        
    # Check consistency
    reference_columns = column_sets[os.path.basename(csv_files[0])]
    inconsistencies = False
    
    print("\n" + "="*60)
    print("COLUMN CONSISTENCY")
    print("="*60)
    for file, cols in column_sets.items():
        if set(cols) != set(reference_columns):
            inconsistencies = True
            missing = set(reference_columns) - set(cols)
            extra = set(cols) - set(reference_columns)
            print(f"File {file} has different columns!")
            if missing: print(f"  Missing: {missing}")
            if extra: print(f"  Extra: {extra}")
            
    if not inconsistencies:
        print("All files have exactly the same columns.")
        
    print("\nConcatenating dataframes...")
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print("\n" + "="*60)
    print("MISSING VALUES & DUPLICATES")
    print("="*60)
    missing_counts = combined_df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Total missing values across all cells: {total_missing:,}")
    if total_missing > 0:
        print("Columns with missing values:")
        print(missing_counts[missing_counts > 0].to_string())
        
    num_duplicates = combined_df.duplicated().sum()
    print(f"Total duplicate rows: {num_duplicates:,} ({(num_duplicates/len(combined_df))*100:.2f}%)")
    
    print("\n" + "="*60)
    print("DATA TYPES SUMMARY")
    print("="*60)
    
    dtype_summary = []
    for col in combined_df.columns:
        col_dtype = str(combined_df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(combined_df[col])
        flag = "FLAG: Might need numeric conversion" if not is_numeric else "OK"
        if col.lower() in ["protocol type", "protocol"]: 
            flag = "Expected Categorical"
            
        dtype_summary.append({
            'Column Name': col,
            'Data Type': col_dtype,
            'Is Numeric': is_numeric,
            'Status': flag
        })
        
    summary_df = pd.DataFrame(dtype_summary)
    pd.set_option('display.max_rows', None)
    print(summary_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
