import pandas as pd
import numpy as np
from scipy import stats
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Complete TRIBE v2 Analysis: Engagement + Marketing Pillars")
    parser.add_argument("file_path", help="Path to the input CSV file")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # --- 1. DATA CLEANING ---
    cols_to_fix = [
        'plat_view_count', 'plat_like_count', 'plat_comment_count', 
        'plat_save_count', 'derived_age_days', 'video_duration_seconds'
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    brain_cols = [
        "region_visual_cortex", "region_fusiform_face", "region_amygdala",
        "region_prefrontal", "region_temporal_language", "region_hippocampus",
        "region_motor", "region_reward", "overall_score"
    ]
    for col in brain_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 2. METRIC CALCULATION (OLD + NEW) ---
    # OLD: Popularity & Velocity
    df['log_likes'] = np.log10(df['plat_like_count'].clip(lower=1))
    df['vpd'] = df['plat_view_count'] / df['derived_age_days'].clip(lower=1)
    df['log_vpd'] = np.log10(df['vpd'].clip(lower=1))
    
    # NEW: Recall (Comments)
    df['log_comments'] = np.log10(df['plat_comment_count'].clip(lower=1))
    
    # NEW: Intent/Efficiency (Likes per View)
    df['er_percent'] = (df['plat_like_count'] / df['plat_view_count'].replace(0, np.nan)) * 100
    
    # NEW: Retention (View/Duration Ratio)
    if 'video_duration_seconds' in df.columns:
        df['retention_proxy'] = df['plat_view_count'] / df['video_duration_seconds'].replace(0, np.nan)
        df['log_retention'] = np.log10(df['retention_proxy'].clip(lower=1))

    # --- 3. HEALTH CHECK ---
    print(f"--- Comprehensive Data Health Check ---")
    print(f"Total rows:      {len(df)}")
    print(f"Valid Views:     {df['plat_view_count'].notna().sum()}")
    print(f"Valid Likes:     {df['plat_like_count'].notna().sum()}")
    print(f"Valid Comments:  {df['plat_comment_count'].notna().sum()}")
    if 'video_duration_seconds' in df.columns:
        print(f"Valid Duration:  {df['video_duration_seconds'].notna().sum()}")
    print("-" * 45)

    # --- 4. THE COMPLETE PILLAR ANALYSIS ---
    targets = {
        "log_likes":     "POPULARITY (Total Likes)",
        "log_vpd":       "AWARENESS (Views/Day)",
        "log_comments":  "RECALL (Comment Volume)",
        "er_percent":    "INTENT (Like/View Ratio)",
    }
    
    # Add retention if we have duration data
    if 'log_retention' in df.columns and df['log_retention'].notna().any():
        targets["log_retention"] = "RETENTION (View/Duration)"

    for target_key, target_name in targets.items():
        print(f"\n=== Analysis: {target_name} ===")
        print(f"{'Brain Region':<25} | {'r-value':<8} | {'p-value':<8} | {'n':<5}")
        print("-" * 55)
        
        for col in brain_cols:
            if col not in df.columns: continue
            
            # Filter for valid pairs
            sub = df[[col, target_key]].dropna()
            sub = sub[np.isfinite(sub[target_key])]
            
            if len(sub) < 5: continue

            r, p = stats.pearsonr(sub[col], sub[target_key])
            sig_star = "*" if p < 0.05 else ""
            print(f"  {col:<23} | {r:>+7.3f} | {p:>7.3f}{sig_star} | {len(sub):<5}")

    # --- 5. BRAND SPECIFIC PERFORMANCE ---
    if "plat_uploader" in df.columns:
        print("\n=== Brand-Specific Virality (Overall Score vs. log_vpd) ===")
        for brand, grp in df.groupby("plat_uploader"):
            sub = grp[["overall_score", "log_vpd"]].dropna()
            sub = sub[np.isfinite(sub["log_vpd"])]
            if len(sub) < 5: continue
            
            r, p = stats.pearsonr(sub["overall_score"], sub["log_vpd"])
            sig_star = "*" if p < 0.05 else ""
            print(f"  {brand:<20} | r={r:+.3f} | p={p:.3f}{sig_star} | n={len(sub)}")

if __name__ == "__main__":
    main()