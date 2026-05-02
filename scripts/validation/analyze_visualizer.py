import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Visualize TRIBE v2 Results")
    parser.add_argument("file_path", help="Path to the results.csv")
    args = parser.parse_args()

    # 1. Load and Clean (Same logic as before)
    try:
        df = pd.read_csv(args.file_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    df['plat_view_count'] = pd.to_numeric(df['plat_view_count'], errors='coerce')
    df['derived_age_days'] = pd.to_numeric(df['derived_age_days'], errors='coerce')
    df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
    
    # Calculate Target
    df["views_per_day"] = df["plat_view_count"] / df["derived_age_days"].replace(0, 1)
    df["log_vpd"] = np.log10(df["views_per_day"].where(df["views_per_day"] > 0))

    # Drop rows that don't have our axes
    plot_df = df.dropna(subset=["overall_score", "log_vpd", "plat_uploader"])

    if len(plot_df) < 5:
        print("Not enough valid data points to plot.")
        return

    # 2. Set the Style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # 3. Create the Plot
    # We use lmplot to get both the scatter points AND the regression trend line
    g = sns.lmplot(
        data=plot_df, 
        x="overall_score", 
        y="log_vpd", 
        hue="plat_uploader",
        height=7, 
        aspect=1.4,
        scatter_kws={'alpha':0.6, 's':100}
    )

    # 4. Refine Labels
    g.set_axis_labels("TRIBE v2 Overall Score", "Social Performance (log10 Views per Day)")
    plt.title("Neural Quality vs. Real-World Performance\n(Trendlines show within-brand correlation)", fontsize=14, pad=20)
    
    # 5. Save and Show
    output_name = "tribe_validation_chart.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Chart saved as: {output_name}")
    
    # Try to open it automatically (works on macOS)
    import os
    if sys.platform == "darwin":
        os.system(f"open {output_name}")

if __name__ == "__main__":
    main()