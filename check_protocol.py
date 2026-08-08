import pandas as pd
import glob
import os

def main():
    directory = r'E:\Digital Twin'
    file_pattern = os.path.join(directory, '*_train.pcap.csv')
    csv_files = glob.glob(file_pattern)
    
    # Only load 'Protocol Type' to save memory and time
    print("Loading Protocol Type from all files...")
    df_list = []
    for f in csv_files:
        df = pd.read_csv(f, usecols=['Protocol Type'])
        df_list.append(df)
        
    df = pd.concat(df_list, ignore_index=True)
    
    val_counts = df['Protocol Type'].value_counts(dropna=False)
    total_rows = len(df)
    
    print("\n--- Top 20 Protocol Type values ---")
    print(val_counts.head(20).to_string())
    
    print(f"\nTotal unique values: {len(val_counts)}")
    
    # Check whole numbers vs fractional
    is_whole = (df['Protocol Type'] % 1 == 0)
    whole_count = is_whole.sum()
    frac_count = total_rows - whole_count
    print(f"\nWhole numbers: {whole_count} ({whole_count/total_rows*100:.2f}%)")
    print(f"Fractional numbers: {frac_count} ({frac_count/total_rows*100:.2f}%)")
    
    # Check where fractional numbers cluster
    if frac_count > 0:
        frac_df = df[~is_whole].copy()
        frac_df['nearest_int'] = frac_df['Protocol Type'].round()
        cluster_counts = frac_df['nearest_int'].value_counts(normalize=True) * 100
        print("\n--- Fractional values cluster around these integers (%) ---")
        print(cluster_counts.head(10).to_string())
        
        # Check standard deviation of fractional values from their nearest int
        frac_df['dist'] = (frac_df['Protocol Type'] - frac_df['nearest_int']).abs()
        print(f"\nAverage distance from nearest integer: {frac_df['dist'].mean():.4f}")
    
if __name__ == "__main__":
    main()
