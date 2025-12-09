# run_pipeline.py
import argparse
import subprocess
import os
import sys

# Define default paths
DEFAULT_CSV = "data/main_detection_log.csv"
DEFAULT_SORTED = "output/sorted_images"
DEFAULT_ANNOTATED = "output/annotated_images"
DEFAULT_CROPS = "output/cropped_crops_by_species"
DEFAULT_JSON = "data/analyzed_data.json"

# MASTER LIST OF ALL CSV FIELDS IN DESIRED ORDER
MASTER_FIELD_ORDER = [
    'Image_Filename', 'Detection_Index', 
    'Image_Width', 'Image_Height', 'Timestamp',
    'MD_Class_ID', 'MD_Confidence', 
    'X_min', 'Y_min', 'X_max', 'Y_max',
    'Predicted_Species', 'Classification_Confidence'
]
FIELD_ORDER_STRING = ",".join(MASTER_FIELD_ORDER)

# Define the order of execution
PIPELINE_STEPS = {
    'detect': 'detect_and_log.py',
    'metadata': 'extract_metadata.py',
    'classify': 'classify_data.py',
    'sort': 'sort_images.py',
    'visualize': 'annotate_images.py',
    'json': 'export_to_json.py'
}

def execute_step(script_name, arguments):
    """Executes a single Python script via subprocess."""
    cmd = [sys.executable, script_name] + arguments
    print(f"\n--- Running Step: {script_name} ---")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {script_name}:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Script not found: {script_name}. Ensure all scripts are in the current directory.")
        return False
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the PytorchWildlife detection and classification pipeline.")
    
    parser.add_argument('input_dir', type=str, 
                        help="Directory containing input images (e.g., 'input_data').")
    parser.add_argument('--steps', nargs='+', default=list(PIPELINE_STEPS.keys()), 
                        choices=PIPELINE_STEPS.keys(),
                        help="Select which steps of the pipeline to run.")

    parser.add_argument('--csv', dest='csv', default=DEFAULT_CSV,
                        help=f"Master Detection Log CSV file path. (Default: {DEFAULT_CSV})")
    
    parser.add_argument('--json', dest='json_output', default=DEFAULT_JSON,
                        help=f"Output file path for the standardized JSON data. (Default: {DEFAULT_JSON})")
    
    parser.add_argument('--sorted', dest='sorted', default=DEFAULT_SORTED,
                        help=f"Output directory for sorted 'empty' and 'non-empty' images. (Default: {DEFAULT_SORTED})")
    parser.add_argument('--annotated', dest='annotated', default=DEFAULT_ANNOTATED,
                        help=f"Output directory for annotated images with bounding boxes and labels. (Default: {DEFAULT_ANNOTATED})")
    parser.add_argument('--crops', dest='crops', default=DEFAULT_CROPS,
                        help=f"Output directory for cropped images (with species subfolders). (Default: {DEFAULT_CROPS})")

    args = parser.parse_args()
    
    # Ensure base output directories exist
    os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.json_output) or '.', exist_ok=True)
    os.makedirs(args.sorted, exist_ok=True)
    os.makedirs(args.annotated, exist_ok=True)
    os.makedirs(args.crops, exist_ok=True)

    # Execute Pipeline
    for step in args.steps:
        script = PIPELINE_STEPS[step]
        success = False
        
        # Pass the field order string to scripts that write or update the CSV
        if step == 'detect':
            success = execute_step(script, [args.input_dir, args.csv, FIELD_ORDER_STRING])
        
        elif step == 'metadata':
            success = execute_step(script, [args.input_dir, args.csv, FIELD_ORDER_STRING])
            
        elif step == 'classify':
            success = execute_step(script, [args.input_dir, args.csv, FIELD_ORDER_STRING])
            
        elif step == 'sort':
            # sort_images.py only READS the CSV
            success = execute_step(script, [args.input_dir, args.csv, args.sorted])

        elif step == 'visualize':
            # annotate_images.py reads the CSV and needs both output dirs
            success = execute_step(script, [args.input_dir, args.csv, args.annotated, args.crops])
            
        elif step == 'json':
            # JSON export needs the final CSV path and the output JSON path
            success = execute_step(script, [args.csv, args.json_output])
            
        if not success:
            print(f"\nPipeline failed at step: {step}. Stopping execution.")
            sys.exit(1)

    print("\n\n✅ Pipeline finished successfully!")
    print(f"Final data exported to CSV: {args.csv}")
    print(f"Final data exported to JSON: {args.json_output}")