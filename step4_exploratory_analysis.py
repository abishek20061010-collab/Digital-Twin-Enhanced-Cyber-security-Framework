import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def get_labels_from_filename(filename):
    basename = os.path.basename(filename)
    attack_type = basename.replace('_train.pcap.csv', '')
    if "Benign" in basename: category = "Benign"
    elif "Spoofing" in basename: category = "Spoofing"
    elif "Recon" in basename: category = "Recon"
    elif "MQTT" in basename: category = "MQTT"
    elif "DDoS" in basename: category = "DDoS"
    elif "DoS" in basename: category = "DoS"
    else: category = "Unknown"
    return category, attack_type

def main():
    directory = r'E:\Digital Twin'
    file_pattern = os.path.join(directory, '*_train.pcap.csv')
    csv_files = glob.glob(file_pattern)
    
    print("Loading data for analysis...")
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        category, attack_type = get_labels_from_filename(file)
        df['category'] = category
        df['attack_type'] = attack_type
        df_list.append(df)
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print("Generating class distribution plots...")
    sns.set_theme(style="whitegrid")
    
    # 1. Category Distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=combined_df, y='category', order=combined_df['category'].value_counts().index, hue='category', palette='viridis', legend=False)
    plt.title('Distribution of Category Labels')
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.tight_layout()
    cat_plot_path = os.path.join(directory, 'category_distribution.png')
    plt.savefig(cat_plot_path)
    plt.close()
    
    # 2. Attack Type Distribution
    plt.figure(figsize=(12, 10))
    ax = sns.countplot(data=combined_df, y='attack_type', order=combined_df['attack_type'].value_counts().index, hue='attack_type', palette='magma', legend=False)
    plt.title('Distribution of Attack Type Labels')
    plt.xlabel('Count')
    plt.ylabel('Attack Type')
    plt.tight_layout()
    attack_plot_path = os.path.join(directory, 'attack_type_distribution.png')
    plt.savefig(attack_plot_path)
    plt.close()
    
    print(f"Plots successfully saved!")
    
    print("\nCalculating descriptive statistics (mean/std/min/max) grouped by category...")
    numeric_cols = combined_df.select_dtypes(include='number').columns
    
    # Group by category and compute stats
    grouped_stats = combined_df.groupby('category')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
    
    # Saving full stats to CSV
    stats_path = os.path.join(directory, 'descriptive_statistics.csv')
    grouped_stats.to_csv(stats_path)
    print(f"Full statistics (45 columns) saved to: {stats_path}")
    
    # Print a small representative sample to the console
    features_to_show = ['Header_Length', 'Duration', 'Rate', 'Tot size', 'IAT']
    
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS SUMMARY (Sample Features)")
    print("="*60)
    for cat in combined_df['category'].unique():
        print(f"\n--- Category: {cat} ---")
        cat_stats = grouped_stats.loc[cat, features_to_show]
        # Restructure for pretty printing
        df_print = pd.DataFrame()
        for feat in features_to_show:
            df_print[feat] = cat_stats[feat]
        print(df_print.T.to_string())

if __name__ == "__main__":
    main()
