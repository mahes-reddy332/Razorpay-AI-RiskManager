import sys
import os
import pandas as pd
import networkx as nx

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model_v2 import compute_inflow_diversity, compute_counterparty_repeat_rate, apply_frozen_config, FROZEN_CONFIG, extract_features

def smb_regression_check():
    print("Running SMB Regression Check...")
    
    # Run full feature extraction to get the dataset
    df = extract_features()
    
    # We need to find an SMB account. In our generator, they are usually labeled with high diversity or specific MCCs.
    # Let's look for accounts with "behavior_type" == "smb" in accounts.csv or find an account with MCC = RETAIL/WHOLESALE and high diversity.
    acc_df = pd.read_csv('data/accounts.csv').set_index('account_id')
    merged = df.join(acc_df, rsuffix='_acc')
    
    # Find a legit SMB: Not a mule, MCC not risky, high in_degree
    legit_smbs = merged[(merged['is_mule'] == False) & (merged['inflow_diversity'] > 15)]
    
    if len(legit_smbs) == 0:
        print("No high-diversity legitimate SMBs found in this dataset.")
        return
        
    sample_smb_id = legit_smbs.index[0]
    sample = legit_smbs.loc[sample_smb_id]
    
    print(f"\n--- SMB REGRESSION CHECK: {sample_smb_id} ---")
    print(f"MCC Code: {sample['mcc_code']}")
    print(f"Inflow Diversity: {sample['inflow_diversity']} (Raw count of unique senders)")
    print(f"Repeat Rate: {sample['counterparty_repeat_rate']}")
    print(f"Velocity Ratio: {sample['max_velocity_ratio']}")
    
    # Compute its final score using the current FROZEN_CONFIG
    # Create a single-row dataframe for apply_frozen_config
    single_df = merged.loc[[sample_smb_id]].copy()
    
    # Re-apply scoring logic
    score = apply_frozen_config(single_df, FROZEN_CONFIG)[0]
    
    print(f"Calculated Score: {score}")
    print(f"FROZEN_CONFIG diversity_weight: {FROZEN_CONFIG['diversity_weight']}")
    
    if score >= FROZEN_CONFIG['dec_thresh']:
        print(">>> WARNING: SMB is incorrectly flagged as MULE (Auto-freeze)! Feature weights need constraint.")
    elif score >= FROZEN_CONFIG['manual_thresh']:
        print(">>> CAUTION: SMB is routed to MANUAL REVIEW.")
    else:
        print(">>> PASS: SMB is correctly marked SAFE.")

if __name__ == "__main__":
    smb_regression_check()
