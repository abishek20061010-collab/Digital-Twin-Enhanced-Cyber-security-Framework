import pandas as pd
import numpy as np

df = pd.read_csv('preprocessed_data/baseline_model_comparison.csv')
cv_df = df[df['Model'].str.contains('CV')].copy()

# Extract base model name
cv_df['BaseModel'] = cv_df['Model'].apply(lambda x: x.split(' (')[0])

metrics = ['Precision', 'Recall', 'F1']
classes = ['Benign', 'Spoofing', 'Recon', 'MQTT', 'DoS', 'DDoS']

results = []
for model in ['SGD', 'RF', 'LSTM']:
    model_df = cv_df[cv_df['BaseModel'] == model]
    for c in classes:
        for m in metrics:
            col = f'{c} {m}'
            mean_val = model_df[col].mean()
            std_val = model_df[col].std()
            results.append({'Model': model, 'Class': c, 'Metric': m, 'Mean': mean_val, 'Std': std_val})

res_df = pd.DataFrame(results)

print("Markdown Table:")
print("| Model | Class | Precision (Mean ± Std) | Recall (Mean ± Std) | F1 (Mean ± Std) |")
print("| :--- | :--- | :--- | :--- | :--- |")

for model in ['SGD', 'RF', 'LSTM']:
    for c in classes:
        p_row = res_df[(res_df['Model']==model) & (res_df['Class']==c) & (res_df['Metric']=='Precision')].iloc[0]
        r_row = res_df[(res_df['Model']==model) & (res_df['Class']==c) & (res_df['Metric']=='Recall')].iloc[0]
        f_row = res_df[(res_df['Model']==model) & (res_df['Class']==c) & (res_df['Metric']=='F1')].iloc[0]
        
        c_str = c
        if c in ['Spoofing', 'Recon']:
            c_str = f"**{c}**"
            
        print(f"| {model} | {c_str} | {p_row['Mean']:.4f} ± {p_row['Std']:.4f} | {r_row['Mean']:.4f} ± {r_row['Std']:.4f} | {f_row['Mean']:.4f} ± {f_row['Std']:.4f} |")
