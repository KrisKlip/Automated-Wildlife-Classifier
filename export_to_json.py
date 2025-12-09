# export_to_json.py
import csv
import json
import argparse
import os

def load_csv_data(csv_path):
    """Loads all records from the CSV file."""
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {csv_path}")
        return []

def create_researcher_json(input_csv_path, output_json_path):
    """
    Groups CSV records by image, filters for classified animal detections,
    and converts the data to a standardized JSON format.
    """
    all_records = load_csv_data(input_csv_path)
    if not all_records:
        print("Error: Input CSV is empty or cannot be read.")
        return

    # 1. Group records by Image_Filename
    records_by_image = {}
    for record in all_records:
        filename = record['Image_Filename']
        if filename not in records_by_image:
            records_by_image[filename] = []
        records_by_image[filename].append(record)

    final_json_data = []

    # 2. Process and restructure the data for each image
    for filename, detections in records_by_image.items():
        # Skip if essential metadata is missing
        if not all(k in detections[0] for k in ['Image_Width', 'Image_Height', 'Timestamp']):
            continue

        try:
            image_record = {
                "file_name": filename,
                "width": int(detections[0]['Image_Width']),
                "height": int(detections[0]['Image_Height']),
                "datetime_original": detections[0]['Timestamp'],
                "annotations": []
            }
        except ValueError:
            print(f"Warning: Skipping {filename} due to invalid Image_Width/Height.")
            continue


        for record in detections:
            # Filter 1: Only process records that represent a valid animal detection (MD_Class_ID '0')
            try:
                md_class_id = int(record.get('MD_Class_ID', -1))
                if md_class_id != 0:
                    continue 
            except ValueError:
                continue
                
            # Filter 2: Only include classified animals
            predicted_species = record.get('Predicted_Species', '').strip()
            if not predicted_species or predicted_species.lower() in ['unknown', 'none', '']:
                continue

            try:
                # Convert bounding box coordinates to integers
                x_min = int(record['X_min'])
                y_min = int(record['Y_min'])
                x_max = int(record['X_max'])
                y_max = int(record['Y_max'])
                
                # Calculate [x_min, y_min, width, height] for the standardized bbox format
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                annotation = {
                    "detection_id": int(record['Detection_Index']),
                    "category": predicted_species,
                    "confidence": float(record['Classification_Confidence']),
                    "bbox": [x_min, y_min, bbox_width, bbox_height]
                }
                
                image_record['annotations'].append(annotation)

            except Exception:
                # Catch records with missing or invalid numeric data
                continue

        # Filter 3: Only include image records that have annotations
        if image_record['annotations']:
            final_json_data.append(image_record)

    # 3. Write the final data to JSON
    os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(final_json_data, f, indent=4)

    print(f"\n--- JSON Export Complete ---")
    print(f"Exported data for {len(final_json_data)} classified images to: {output_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exports the Master Detection CSV to a standardized JSON format.")
    parser.add_argument("input_csv_path", type=str, help="Path to the final classified Master Detection CSV file.")
    parser.add_argument("output_json_path", type=str, help="Path for the output standardized JSON file.")
    args = parser.parse_args()
    create_researcher_json(args.input_csv_path, args.output_json_path)