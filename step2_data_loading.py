import pandas as pd
import os
import glob

def get_labels_from_filename(filename):
    basename = os.path.basename(filename)
    
    # Derive attack type by removing suffix
    attack_type = basename.replace('_train.pcap.csv', '')
    
    # Derive category
    if "Benign" in basename:
        category = "Benign"
    elif "Spoofing" in basename:
        category = "Spoofing"
    elif "Recon" in basename:
        category = "Recon"
    elif "MQTT" in basename:
        category = "MQTT"
    elif "DDoS" in basename:
        category = "DDoS"
    elif "DoS" in basename:
        category = "DoS"
    else:
        category = "Unknown"
        
    return category, attack_type

def main():
    directory = r'E:\Digital Twin'
    file_pattern = os.path.join(directory, '*_train.pcap.csv')
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        print("No CSV files found matching the pattern.")
        return
        
    print(f"Found {len(csv_files)} files. Loading...")
    
    df_list = []
    
    for file in csv_files:
        print(f"Loading {os.path.basename(file)}...")
        df = pd.read_csv(file)
        
        category, attack_type = get_labels_from_filename(file)
        df['category'] = category
        df['attack_type'] = attack_type
        
        df_list.append(df)
        
    print("\nConcatenating dataframes...")
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print("=" * 50)
    print("DATA LOADING COMPLETE")
    print("=" * 50)
    print(f"Total Row Count: {len(combined_df):,}")
    print("\n" + "=" * 50)
    print("Rows per Category:")
    print(combined_df['category'].value_counts().to_string())
    print("\n" + "=" * 50)
    print("Rows per Attack Type:")
    print(combined_df['attack_type'].value_counts().to_string())
    print("=" * 50)
    
    # Save a sample just in case we want to use it later, but not required right now.
    
if __name__ == "__main__":
    main()
