# classify_data.py
import os
import csv
import argparse
import numpy as np
from PIL import Image
import torch
import supervision as sv
from PytorchWildlife.models import classification as pw_classification

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

def load_csv_data(csv_path):
    """Loads all records (as strings) from the CSV file."""
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)
        
def update_csv_data(input_dir, input_csv_path, field_order_str):
    """Runs the classifier and updates the CSV."""
    
    field_order = field_order_str.split(',')
    
    all_records = load_csv_data(input_csv_path)
    if not all_records:
        print("Error: Input CSV is empty or cannot be read.")
        return

    print(f"Initializing AI4G Serengeti Classifier on {DEVICE}...")
    classification_model = pw_classification.AI4GSnapshotSerengeti(device=DEVICE)

    # Group records by image file
    records_by_image = {}
    for record in all_records:
        records_by_image.setdefault(record['Image_Filename'], []).append(record)

    processed_records_count = 0
    
    for filename, records in records_by_image.items():
        img_path = os.path.join(input_dir, filename)
        
        try:
            input_img = np.array(Image.open(img_path).convert('RGB'))
        except FileNotFoundError:
            print(f"Warning: Image not found for classification: {img_path}. Skipping.")
            continue

        for record in records:
            # Only classify if an animal was detected (MD_Class_ID == 0)
            if int(record['MD_Class_ID']) == 0:
                xyxy = np.array([record['X_min'], record['Y_min'], record['X_max'], record['Y_max']], dtype=int)
                
                cropped_image = sv.crop_image(image=input_img, xyxy=xyxy)
                
                results_clf = classification_model.single_image_classification(cropped_image)
                
                record['Predicted_Species'] = results_clf["prediction"]
                record['Classification_Confidence'] = results_clf["confidence"]
                processed_records_count += 1
                
    # Re-Export the entire updated CSV file
    with open(input_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n--- Classification Complete ---")
    print(f"Updated {processed_records_count} animal records in: {input_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs species classification on detected animal crops and updates the CSV log.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("input_csv_path", type=str, help="Path to the Master Detection CSV file to be updated.")
    parser.add_argument("field_order", type=str, help="Comma-separated string defining the final CSV column order.")
    args = parser.parse_args()
    update_csv_data(args.input_dir, args.input_csv_path, args.field_order)