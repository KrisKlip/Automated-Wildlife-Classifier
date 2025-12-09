# annotate_images.py
import os
import csv
import argparse
import numpy as np
from PIL import Image
import supervision as sv
from supervision.draw.utils import draw_text
import re

# --- CONFIGURATION ---
CLF_CONF_THRES = 0.8 # Confidence threshold for species prediction
# ---------------------

# MegaDetector Class Lookup
MD_CLASS_NAME_LUT = {
    1: 'Person',
    2: 'Vehicle',
    3: 'Empty' 
}
# -----------------------------------------------------------------

def load_csv_data(csv_path):
    """Loads all records from the CSV file."""
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def process_visual_outputs(input_dir, input_csv_path, annotated_output_dir, crop_output_dir):
    """Annotates images and performs cropping based on CSV data."""
    
    all_records = load_csv_data(input_csv_path)
    if not all_records:
        print("Error: Input CSV is empty or cannot be read.")
        return

    os.makedirs(annotated_output_dir, exist_ok=True)
    
    records_by_image = {}
    for record in all_records:
        records_by_image.setdefault(record['Image_Filename'], []).append(record)

    processed_count = 0
    
    for filename, records in records_by_image.items():
        img_path = os.path.join(input_dir, filename)
        
        try:
            input_img_np = np.array(Image.open(img_path).convert('RGB'))
            annotated_img = input_img_np.copy()
        except FileNotFoundError:
            continue

        xyxy_list = []
        label_list = []
        
        # Process Detections for Annotation and Cropping
        for record in records:
            md_class = int(record['MD_Class_ID'])
            
            # Skip if it's the 'empty' placeholder record (MD_Class_ID == -1)
            if md_class == -1:
                continue

            xyxy = [record['X_min'], record['Y_min'], record['X_max'], record['Y_max']]
            xyxy_list.append(xyxy)
            
            species = record.get('Predicted_Species', '')
            clf_conf = float(record.get('Classification_Confidence', 0.0))
            
            # Determine the annotation label
            if md_class == 0 and species and clf_conf > CLF_CONF_THRES:
                label = f"{species} {clf_conf:.2f}"
            elif md_class == 0 and species:
                # Use the first word of the species name if confidence is low
                label = f"Unknown ({species.split()[0]}) {clf_conf:.2f}"
            elif md_class == 0:
                # Use the generic animal label for md_class 0 (animal/unknown)
                label = f"Animal {float(record['MD_Confidence']):.2f}"
            else: # Person (1) or Vehicle (2)
                label = f"{MD_CLASS_NAME_LUT.get(md_class, 'Object')} {float(record['MD_Confidence']):.2f}"

            label_list.append(label)

            # Cropping Logic
            if md_class == 0:
                # Determine the folder name
                if species and clf_conf > CLF_CONF_THRES:
                    folder_name = species
                else:
                    folder_name = 'unknown' 

                # Sanitize folder name
                safe_folder_name = re.sub(r'\W+', '_', folder_name).strip('_').lower()
                
                species_crop_dir = os.path.join(crop_output_dir, safe_folder_name)
                os.makedirs(species_crop_dir, exist_ok=True)

                cropped_img = sv.crop_image(image=input_img_np, xyxy=np.array(xyxy, dtype=int))
                crop_name = f"{os.path.splitext(filename)[0]}_crop_{record['Detection_Index']}.jpg"
                
                Image.fromarray(cropped_img).save(os.path.join(species_crop_dir, crop_name))
        
        # Annotation Logic
        if xyxy_list:
            detections = sv.Detections(
                xyxy=np.array(xyxy_list, dtype=int),
                confidence=np.array([float(r.get('MD_Confidence', 0.0)) for r in records if int(r['MD_Class_ID']) != -1]),
                class_id=np.array([int(r['MD_Class_ID']) for r in records if int(r['MD_Class_ID']) != -1])
            )
            
            box_annotator = sv.BoxAnnotator(
                thickness=4, 
            )
            
            label_annotator = sv.LabelAnnotator(
                text_position=sv.Position.TOP_LEFT, 
                text_scale=1.2,
                text_thickness=2
            )
            
            annotated_img = box_annotator.annotate(
                scene=annotated_img,
                detections=detections,
            )
            
            annotated_img = label_annotator.annotate(
                scene=annotated_img,
                detections=detections,
                labels=label_list
            )
            
            Image.fromarray(annotated_img).save(os.path.join(annotated_output_dir, filename))
            processed_count += 1
            
    print(f"\n--- Visual Outputs Complete ---")
    print(f"Annotated {processed_count} images in: {annotated_output_dir}")
    print(f"Cropped images organized into species subfolders inside: {crop_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Node 4 & 5: Creates annotated images and cropped images using the classified CSV data.")
    parser.add_argument("input_dir", type=str, help="Directory containing source images.")
    parser.add_argument("input_csv_path", type=str, help="Path to the Master Detection CSV file (should be classified).")
    parser.add_argument("annotated_output_dir", type=str, help="Directory to save images with boundary boxes and labels.")
    parser.add_argument("crop_output_dir", type=str, help="Directory to save cropped images (will contain species subfolders).")
    args = parser.parse_args()
    process_visual_outputs(args.input_dir, args.input_csv_path, args.annotated_output_dir, args.crop_output_dir)