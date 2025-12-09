import pandas as pd
import os
import requests
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---

# Set the full path to your ANNOTATIONS file
ANNOTATIONS_PATH = r'C:\Users\knkli\Downloads\SnapshotSerengeti_S1-11_v2_1\SnapshotSerengeti_v2_1_annotations.csv'

# Set the full path to your IMAGES file
IMAGES_PATH = r'C:\Users\knkli\Downloads\SnapshotSerengeti_S1-11_v2_1\SnapshotSerengeti_v2_1_images.csv'

# Set to a POSITIVE INTEGER to limit the total number of downloaded files, or -1 for all.
MAX_TOTAL_DOWNLOADS = 1000

# Filter pattern for captures (e.g., 'SER_S5#' for Season 5)
SEASON_PREFIX = 'SER_S5#' 

# Base URL for the Azure data download
BASE_URL = 'https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/'
DOWNLOAD_FOLDER = 'raw_captures'

# Columns
CAPTURE_ID_COL = 'capture_id'
PATH_COL = 'image_path_rel'

# Define the chunk size for reading large files
CHUNK_SIZE = 100000
# ---------------------

def download_file(url, local_path):
    """Downloads a file from a URL to a local path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False

def download_serengeti_images(annotations_path, images_path, total_limit, season_prefix):

    # 1. --- PASS 1: Collect Target Capture IDs ---
    target_captures = set()

    print(f"\n--- PASS 1: Collecting Capture IDs from Annotations File (Season: {season_prefix.strip('#')}) ---")

    dtype_map = {CAPTURE_ID_COL: 'object'}

    try:
        ann_reader = pd.read_csv(
            annotations_path,
            chunksize=CHUNK_SIZE,
            usecols=[CAPTURE_ID_COL],
            dtype=dtype_map
        )
    
        for i, chunk in enumerate(ann_reader):
            if total_limit != -1 and len(target_captures) >= total_limit:
                print(f"Total download limit of {total_limit} reached. Stopping Pass 1.")
                break

            chunk[CAPTURE_ID_COL] = chunk[CAPTURE_ID_COL].fillna('').astype(str)
            season_chunk = chunk[chunk[CAPTURE_ID_COL].str.contains(season_prefix)]
           
            # Add unique capture IDs from this chunk to the set
            for capture_id in season_chunk[CAPTURE_ID_COL].unique():
                if total_limit == -1 or len(target_captures) < total_limit:
                    target_captures.add(capture_id)
                else:
                    break
           
            print(f"Processed chunk {i+1}. Current captures matched: {len(target_captures)}")

    except Exception as e:
        print(f"\n❌ Annotation file processing error: {e}")
        return

    # 2. --- PASS 2: Collect Image Paths and Download ---

    if not target_captures:
        print(f"No matching captures found for the specified season ({season_prefix.strip('#')}). Exiting.")
        return

    total_captures_to_find = len(target_captures)
    print(f"\n--- PASS 2: Downloading Images (Total Unique Captures: {total_captures_to_find}) ---")

    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    
    downloaded_count = 0

    try:
        img_reader = pd.read_csv(
            images_path,
            chunksize=CHUNK_SIZE,
            usecols=[CAPTURE_ID_COL, PATH_COL],
            dtype={CAPTURE_ID_COL: 'object', PATH_COL: 'object'}
        )

        pbar = tqdm(total=total_captures_to_find, desc="Download Progress", unit="file", position=0)
    
        for i, chunk in enumerate(img_reader):
            target_ids = list(target_captures)
        
            filtered_paths = chunk[chunk[CAPTURE_ID_COL].isin(target_ids)]
        
            for _, row in filtered_paths.iterrows():
                capture_id = row[CAPTURE_ID_COL]
                rel_path = row[PATH_COL]
            
                if capture_id in target_captures:
                    full_url = BASE_URL + rel_path
                    local_path = os.path.join(DOWNLOAD_FOLDER, os.path.basename(rel_path))

                    # Resumable Check: Skip if file already exists
                    if os.path.exists(local_path):
                        target_captures.remove(capture_id)
                        pbar.update(1)  
                        continue

                    if download_file(full_url, local_path):
                        downloaded_count += 1
                        target_captures.remove(capture_id)
                        pbar.update(1)  
                        pbar.refresh()
            
            if not target_captures:
                print("\nAll target image paths found and processed. Stopping Pass 2.")
                break
        
        pbar.close()
        
    except Exception as e:
        print(f"\n❌ Images file processing error: {e}")
        pbar.close()
        return

    print("\n--- ✅ Download Complete ---")
    print(f"Total Download Limit: {'ALL' if total_limit == -1 else total_limit}")
    print(f"Season Filtered: {season_prefix.strip('#')}")
    print(f"New files downloaded: {downloaded_count}")
    print(f"\nAll files saved directly to the '{DOWNLOAD_FOLDER}' directory.")
    print("----------------------------")


if __name__ == '__main__':
    download_serengeti_images(
        ANNOTATIONS_PATH,
        IMAGES_PATH,
        MAX_TOTAL_DOWNLOADS,
        SEASON_PREFIX
    )