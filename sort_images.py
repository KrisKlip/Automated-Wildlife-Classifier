# sort_images.py
import os
import csv
import argparse
import shutil

def load_csv_data(csv_path):
    """Loads all records from the CSV file."""
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def sort_images_by_detection(input_dir, input_csv_path, output_dir):
    """Sorts and copies images based on the presence of detections."""
    
    all_records = load_csv_data(input_csv_path)
    if not all_records:
        print("Error: Input CSV is empty or cannot be read.")
        return

    # Determine which images are non-empty
    non_empty_files = set()
    for record in all_records:
        # MD_Class_ID 0, 1, or 2 indicates a detection (animal, person, vehicle)
        if int(record['MD_Class_ID']) != -1 and float(record['MD_Confidence']) > 0.1:
             non_empty_files.add(record['Image_Filename'])
    
    # Identify all processed files to find true 'empty' files
    processed_files = set(record['Image_Filename'] for record in all_records)
    empty_files = processed_files - non_empty_files

    # Create output directories
    non_empty_path = os.path.join(output_dir, 'non-empty')
    empty_path = os.path.join(output_dir, 'empty')
    os.makedirs(non_empty_path, exist_ok=True)
    os.makedirs(empty_path, exist_ok=True)

    copied_count = 0
    
    # Copy files
    for filename in processed_files:
        src_path = os.path.join(input_dir, filename)
        
        if filename in non_empty_files:
            dst_path = os.path.join(non_empty_path, filename)
        elif filename in empty_files:
            dst_path = os.path.join(empty_path, filename)
        else:
            continue

        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except FileNotFoundError:
            print(f"Warning: Source file not found: {src_path}")

    print(f"\n--- Image Sorting Complete ---")
    print(f"Total files copied: {copied_count}")
    print(f"Non-Empty: {len(non_empty_files)} | Empty: {len(empty_files)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Copies images to 'empty' or 'non-empty' folders based on CSV detection data.")
    parser.add_argument("input_dir", type=str, help="Directory containing source images.")
    parser.add_argument("input_csv_path", type=str, help="Path to the Master Detection CSV file.")
    parser.add_argument("output_dir", type=str, help="Parent directory for 'empty' and 'non-empty' subfolders.")
    args = parser.parse_args()
    sort_images_by_detection(args.input_dir, args.input_csv_path, args.output_dir)