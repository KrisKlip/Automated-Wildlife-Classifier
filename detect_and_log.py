# detect_and_log.py
import os
import glob
import csv
import argparse
import torch
from PytorchWildlife.models import detection as pw_detection

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

def detect_and_create_csv(input_dir, output_csv_path, field_order_str):
    """Runs MegaDetector and logs bounding box data to a CSV."""
    
    field_order = field_order_str.split(',')
    
    print(f"Initializing MegaDetector V6 on {DEVICE}...")
    detection_model = pw_detection.MegaDetectorV6(
        device=DEVICE, 
        pretrained=True, 
        version="MDV6-yolov10-e"
    )

    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    if not image_paths:
        print(f"Error: No images found in {input_dir}")
        return

    all_detection_records = []
    print(f"Starting detection on {len(image_paths)} images...")

    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        
        # Runs detection
        results = detection_model.single_image_detection(img_path)
        
        # If no detections, create one 'empty' record for tracking
        if not results["detections"]:
            all_detection_records.append({
                'Image_Filename': img_filename,
                'Detection_Index': 0,
                'X_min': 0, 'Y_min': 0, 'X_max': 0, 'Y_max': 0,
                'MD_Class_ID': -1, # Using -1 to denote 'no detection'
                'MD_Confidence': 0.0,
                'Predicted_Species': 'empty',
                'Classification_Confidence': 0.0
            })
            continue

        # Loop through all detected objects
        for i, (xyxy, det_id) in enumerate(zip(results["detections"].xyxy, results["detections"].class_id)):
            det_conf = results["detections"].confidence[i]
            x_min, y_min, x_max, y_max = xyxy
            
            record = {
                'Image_Filename': img_filename,
                'Detection_Index': i,
                'X_min': int(x_min), 'Y_min': int(y_min), 
                'X_max': int(x_max), 'Y_max': int(y_max),
                'MD_Class_ID': int(det_id),
                'MD_Confidence': float(det_conf),
                'Predicted_Species': '', # Placeholder for later classification
                'Classification_Confidence': 0.0
            }
            all_detection_records.append(record)
    
    # Export all collected data to a single CSV file
    if all_detection_records:
        
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        
        with open(output_csv_path, 'w', newline='') as csvfile:
            # Fields not present (e.g., Image_Width, Timestamp) are left blank.
            writer = csv.DictWriter(csvfile, fieldnames=field_order) 
            writer.writeheader()
            writer.writerows(all_detection_records)
            
        print(f"\n--- Detection Log Complete ---")
        print(f"Data for {len(all_detection_records)} detections saved to: {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs MegaDetector on images and logs detection data to a CSV.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_csv_path", type=str, help="Path for the output Master Detection CSV file.")
    parser.add_argument("field_order", type=str, help="Comma-separated string defining the final CSV column order.")
    args = parser.parse_args()
    detect_and_create_csv(args.input_dir, args.output_csv_path, args.field_order)