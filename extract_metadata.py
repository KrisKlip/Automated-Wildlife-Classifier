# extract_metadata.py
import os
import csv
import argparse
from PIL import Image
from PIL.ExifTags import TAGS
import datetime

# Define the EXIF tag ID for 'DateTimeOriginal'
EXIF_TAG_DATETIME_ORIGINAL = 36867

def load_csv_data(csv_path):
    """Loads all records from the CSV file as strings."""
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []

def get_exif_data(img):
    """Safely extracts EXIF data, focusing on the original capture time."""
    exif_data = {}
    try:
        # Use _getexif() to retrieve the raw EXIF dictionary
        info = img._getexif()
        if info:
            timestamp_raw = info.get(EXIF_TAG_DATETIME_ORIGINAL)
            if timestamp_raw:
                exif_data['Timestamp'] = timestamp_raw
    except Exception:
        # Handles errors if the file is not a JPEG, corrupted, or lacks EXIF data
        pass
    return exif_data

def update_metadata(input_dir, input_csv_path, field_order_str):
    """Adds image width, height, and timestamp to the CSV records."""
    
    field_order = field_order_str.split(',')
    
    all_records = load_csv_data(input_csv_path)
    if not all_records:
        return

    print(f"Starting metadata extraction for {len(all_records)} records...")

    # Group records by image file for efficient image loading
    records_by_image = {}
    for record in all_records:
        records_by_image.setdefault(record['Image_Filename'], []).append(record)

    metadata_added_count = 0
    
    for filename, records in records_by_image.items():
        img_path = os.path.join(input_dir, filename)
        
        # Initialize default values
        width, height = 0, 0
        timestamp = ''
        
        try:
            with Image.open(img_path) as img_pil:
                # 1. Extract Dimensions
                width, height = img_pil.size
                
                # 2. Extract Timestamp
                exif = get_exif_data(img_pil)
                timestamp = exif.get('Timestamp', '')

        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping metadata extraction.")
        except Exception as e:
            print(f"Error processing image {filename}: {e}. Setting metadata to default.")
            
        # Update all records belonging to this image
        for record in records:
            record['Image_Width'] = width
            record['Image_Height'] = height
            record['Timestamp'] = timestamp
            metadata_added_count += 1
                
    # Re-Export the entire updated CSV file
    
    # Get the union of all keys from all records to ensure the header is complete
    all_keys = set()
    for record in all_records:
        all_keys.update(record.keys())
    
    fieldnames = list(all_keys)
    
    with open(input_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n--- Metadata Extraction Complete ---")
    print(f"Updated {metadata_added_count} records with image size and timestamp in: {input_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extracts and updates image metadata (dimensions and timestamp) into the CSV log.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("input_csv_path", type=str, help="Path to the Master Detection CSV file to be updated.")
    parser.add_argument("field_order", type=str, help="Comma-separated string defining the final CSV column order.")
    args = parser.parse_args()
    update_metadata(args.input_dir, args.input_csv_path, args.field_order)