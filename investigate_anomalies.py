import pandas as pd
import os
import glob
import numpy as np

def main():
    directory = r'E:\Digital Twin'
    file_pattern = os.path.join(directory, '*_train.pcap.csv')
    csv_files = glob.glob(file_pattern)
    
    print("="*80)
    print("2. COLUMN COUNT AND ORDER CHECK ACROSS FILES")
    print("="*80)
    
    first_file = csv_files[0]
    first_df = pd.read_csv(first_file)
    reference_columns = list(first_df.columns)
    
    mismatch_found = False
    for file in csv_files:
        df = pd.read_csv(file)
        current_columns = list(df.columns)
        if current_columns != reference_columns:
            mismatch_found = True
            print(f"\nFLAG: Column mismatch in {os.path.basename(file)}")
            print(f"  Expected {len(reference_columns)} cols. Got {len(current_columns)} cols.")
            if len(current_columns) == len(reference_columns):
                # Check for order mismatches
                for i, (ref, cur) in enumerate(zip(reference_columns, current_columns)):
                    if ref != cur:
                        print(f"  Mismatch at index {i}: expected '{ref}', got '{cur}'")
                        break
            else:
                missing = set(reference_columns) - set(current_columns)
                extra = set(current_columns) - set(reference_columns)
                if missing: print(f"  Missing: {missing}")
                if extra: print(f"  Extra: {extra}")
    
    if not mismatch_found:
        print("All 19 files have identical column counts AND identical column order.")
    
    print("\nLoading data to check Header_Length and IAT...")
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)
        attack_type = os.path.basename(file).replace('_train.pcap.csv', '')
        df['attack_type'] = attack_type
        df_list.append(df)
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print("\n" + "="*80)
    print("1. TOP 20 ROWS WITH HIGHEST Header_Length")
    print("="*80)
    top_20 = combined_df.nlargest(20, 'Header_Length')
    # Save full details to CSV
    top_20.to_csv(os.path.join(directory, 'top_20_header_length.csv'), index=False)
    # Print select columns to console
    cols_to_show = ['source_file', 'attack_type', 'Header_Length', 'Duration', 'Rate', 'IAT', 'Protocol Type']
    print(top_20[cols_to_show].to_string(index=False))
    print(f"\n(Note: Full top 20 rows with all {len(combined_df.columns)} columns saved to top_20_header_length.csv)")
    
    print("\n" + "="*80)
    print("3. Header_Length PERCENTILES PER SOURCE FILE")
    print("="*80)
    percentiles = [.01, .25, .50, .75, .95, .99]
    header_pct = combined_df.groupby('source_file')['Header_Length'].describe(percentiles=percentiles)
    pct_cols = ['1%', '25%', '50%', '75%', '95%', '99%', 'max']
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(header_pct[pct_cols].to_string())
    
    print("\n" + "="*80)
    print("4. IAT PERCENTILES PER SOURCE FILE")
    print("="*80)
    iat_pct = combined_df.groupby('source_file')['IAT'].describe(percentiles=percentiles)
    pct_cols_iat = ['min', '1%', '25%', '50%', '75%', '95%', '99%', 'max']
    print(iat_pct[pct_cols_iat].to_string())

if __name__ == "__main__":
    main()
