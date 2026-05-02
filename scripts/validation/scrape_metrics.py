import pandas as pd
import instaloader
import time
import random
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Scrape full marketing metrics for TRIBE v2")
    parser.add_argument("csv_path", help="Path to the results.csv file")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: File '{args.csv_path}' not found.")
        sys.exit(1)

    df = pd.read_csv(args.csv_path)
    
    # Ensure all target columns exist
    new_cols = ['plat_comment_count', 'plat_save_count', 'video_duration_seconds']
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    possible_cols = ['reel_url', 'post_url', 'url', 'link']
    url_col = next((c for c in possible_cols if c in df.columns), None)
    
    if not url_col:
        print(f"❌ Error: Could not find a URL column.")
        return

    L = instaloader.Instaloader()
    USER = "vebaxak634"
    
    try:
        L.load_session_from_file(USER)
        print(f"✅ Session loaded for {USER}")
    except FileNotFoundError:
        print(f"❌ Session not found. Run 'instaloader --load-cookies chrome' first.")
        return
    
    # Target rows where view count is missing or comment count is missing
    mask = (df['plat_view_count'].isna()) | (df['plat_comment_count'].isna())
    missing_rows = df[mask]
    
    print(f"Updating {len(missing_rows)} rows with full metrics...")

    count = 0
    for index, row in missing_rows.iterrows():
        url = str(row.get(url_col, ""))
        if not url or url == "nan": continue

        try:
            shortcode = [p for p in url.split('/') if p][-1].split('?')[0] 
            print(f"[{count+1}] Analyzing {shortcode}...", end="\r")
            
            post = instaloader.Post.from_shortcode(L.context, shortcode)
            
            # --- PRIMARY METRICS ---
            df.at[index, 'plat_view_count'] = post.video_view_count
            df.at[index, 'plat_like_count'] = post.likes
            
            # --- NEW MARKETING METRICS ---
            df.at[index, 'plat_comment_count'] = post.comments
            # Note: IG sometimes hides save_count via API unless you own the media, 
            # but we'll attempt to grab it if available.
            df.at[index, 'plat_save_count'] = getattr(post, 'save_count', None)
            df.at[index, 'video_duration_seconds'] = post.video_duration

            # --- UPDATED TIME LOGIC ---
            post_date = post.date_utc
            # Fixed the DeprecationWarning
            now = pd.Timestamp.now(tz='UTC').tz_localize(None)
            age_days = (now - post_date).days
            df.at[index, 'derived_age_days'] = max(age_days, 1)

            count += 1
            if count % 5 == 0:
                df.to_csv(args.csv_path, index=False)

            # Keep it human-like
            time.sleep(random.uniform(10, 15))

        except Exception as e:
            print(f"\n❌ Error on {url}: {e}")
            if "429" in str(e):
                print("🛑 Rate limited. Stopping.")
                break
            time.sleep(10)

    df.to_csv(args.csv_path, index=False)
    print(f"\n✅ Finished! Full metrics saved to {args.csv_path}")

if __name__ == "__main__":
    main()