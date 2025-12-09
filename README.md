# PytorchWildlife-Pipeline: Automated Camera Trap Image Processing

This repository provides a complete pipeline to automatically process camera trap images using state-of-the-art computer vision models from the PytorchWildlife library (MegaDetector V6 and AI4G Classifier).

The pipeline detects objects, classifies species, extracts metadata, sorts images into 'empty' and 'non-empty' folders, and generates annotated visuals and a structured JSON output.

## 📁 Repository Structure

Your current, streamlined project structure looks like this:

```
PytorchWildlife-Pipeline/
├── raw_captures/            <- **Primary Input: Your raw images go here**
├── data/                    <- Final data products
│   └── main_detection_log.csv
│   └── analyzed_data.json
├── output/                  <- Generated visualizations and sorted files
│   ├── sorted_images/
│   ├── annotated_images/
│   └── cropped_crops_by_species/
├── run_pipeline.py          <- Main entry point for the entire workflow
├── requirements.txt
└── README.md
```

-----

## ⚙️ 1. Setup and Installation

### A. Prerequisites

  * **Python 3.8+**
  * **NVIDIA GPU (Recommended):** For best performance with Pytorch and the models, a CUDA-enabled GPU is highly recommended. The scripts automatically detect and use a GPU if available.

### B. Virtual Environment

Setting up a virtual environment is strongly recommended:

```bash
# Create a virtual environment
python -m venv venv 

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### C. Install Dependencies

You'll need Pytorch, PytorchWildlife, and several utility libraries. Ensure your `requirements.txt` file contains the following packages and install them:

```bash
pip install torch
pip install torch PytorchWildlife supervision pandas tqdm Pillow requests numpy lightning omegaconf
```

*(Note: If you have GPU issues, you may need to install the specific Pytorch version that matches your CUDA toolkit separately.)*

-----

## 🚀 2. Running the Pipeline

The pipeline is executed via the central script, **`run_pipeline.py`**. It requires one **positional argument** (the input image directory).

### A. Full Pipeline Run

To run the **entire workflow** using the default output paths:

1.  Place all your images in the **`raw_captures/`** folder.
2.  Execute the script from the root directory:

<!-- end list -->

```bash
python run_pipeline.py raw_captures/
```

### B. Selective Pipeline Steps

You can choose specific steps to execute using the optional **`--steps`** argument. This is useful for restarting from a failed step or running only visualization after re-classification.

| Step Argument | Script Executed | Description |
| :--- | :--- | :--- |
| `detect` | `detect_and_log.py` | Runs MegaDetector V6 and creates the initial CSV log. |
| `metadata` | `extract_metadata.py` | Adds image dimensions and EXIF timestamps to the CSV. |
| `classify` | `classify_data.py` | Runs the AI4G Species Classifier on animal detections. |
| `sort` | `sort_images.py` | Copies images into `output/sorted_images/empty` or `non-empty`. |
| `visualize` | `annotate_images.py` | Creates annotated images and species-specific crops. |
| `json` | `export_to_json.py` | Exports the final classified data to a standardized JSON file. |

**Example (Run only Detection and Metadata):**

```bash
python run_pipeline.py raw_captures/ --steps detect metadata
```

-----

## 🔬 3. Configuration: Confidence Thresholds

You can adjust the confidence thresholds for detection and classification by editing the two following files. Adjusting these values allows you to trade-off between **precision** (higher threshold) and **recall** (lower threshold).

### A. MegaDetector (MD) Confidence Threshold (`sort_images.py`)

This threshold controls which images are sorted into the **`non-empty`** folder.

  * **File:** `sort_images.py`
  * **Variable:** `MD_CONF_THRES`
  * **Effect:** Only detections with a confidence **higher** than this value will mark an image as "non-empty" for sorting purposes.

<!-- end list -->

```python
# sort_images.py

# --- CONFIGURATION ---
MD_CONF_THRES = 0.1 # Confidence threshold for detection
# ---------------------
```

### B. Classification (CLF) Confidence Threshold (`annotate_images.py`)

This threshold controls the labeling and cropping of classification results during visualization.

  * **File:** `annotate_images.py`
  * **Variable:** `CLF_CONF_THRES`
  * **Effect:** Predictions with a confidence **lower** than this value will typically be labeled as **'Unknown'** on the annotated images. It ensures only high-confidence species predictions are displayed.

<!-- end list -->

```python
# annotate_images.py

# --- CONFIGURATION ---
CLF_CONF_THRES = 0.8 # Confidence threshold for species prediction
# ---------------------
```

-----

## 🛠️ 4. Configuration: Changing Models

You can easily swap the detection and classification models used by PytorchWildlife by editing the respective Python files.

### A. Detection Model (`detect_and_log.py`)

Edit the **`detect_and_log.py`** file around line 15 to change the model class and version:

```python
# detect_and_log.py

# ... imports ...

def detect_and_create_csv(input_dir, output_csv_path, field_order_str):
    # ...
    print(f"Initializing MegaDetector V6 on {DEVICE}...")
    detection_model = pw_detection.MegaDetectorV6(
        device=DEVICE, 
        pretrained=True, 
        version="MDV6-yolov10-e" # <--- EDIT THIS VERSION
    )
    # ...
```

| Detection Model | Class | Available Versions (`version="..."`) |
| :--- | :--- | :--- |
| **MegaDetectorV5** | `pw_detection.MegaDetectorV5` | `"a"`, `"b"` |
| **MegaDetectorV6** | `pw_detection.MegaDetectorV6` | `"MDV6-yolov9-c"`, `"MDV6-yolov9-e"`, `"MDV6-yolov10-c"`, `"MDV6-yolov10-e"` |

### B. Classification Model (`classify_data.py`)

Edit the **`classify_data.py`** file around line 24 to change the model class:

```python
# classify_data.py

# ... imports ...

def update_csv_data(input_dir, input_csv_path, field_order_str):
    # ...
    print(f"Initializing AI4G Serengeti Classifier on {DEVICE}...")
    classification_model = pw_classification.AI4GSnapshotSerengeti(device=DEVICE) # <--- EDIT THIS CLASS
    # ...
```

| Classification Model | Class | Description |
| :--- | :--- | :--- |
| AI4G Opossum | `pw_classification.AI4GOpossum` | Opossum species classifier. |
| AI4G Amazon | `pw_classification.AI4GAmazonRainforest` | Amazon rainforest species classifier. |
| **AI4G Serengeti** | `pw_classification.AI4GSnapshotSerengeti` | **Currently used.** Snapshot Serengeti species classifier. |
| **Custom Model** | `pw_classification.CustomClassifier` | Use this class to load a custom Pytorch model from a local path. |

-----

## 5\. Overriding Default Paths

You can override the default output paths for the primary files and folders using optional arguments in `run_pipeline.py`:

| Argument | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **`input_dir`** | *Positional* | N/A | **Required.** Path to the folder containing your input images (e.g., `raw_captures/`). |
| `--csv` | *Optional* | `data/master_detection_log.csv` | Output path for the final **CSV detection log**. |
| `--json` | *Optional* | `data/research_data.json` | Output path for the final standardized **JSON data export**. |
| `--sorted` | *Optional* | `output/sorted_images` | Parent directory for the `empty/` and `non-empty/` subfolders. |
| `--annotated` | *Optional* | `output/annotated_images` | Directory to save images with bounding box and species labels. |
| `--crops` | *Optional* | `output/cropped_crops_by_species` | Directory to save cropped detections, organized by species. |

**Example (Using custom paths):**

```bash
python run_pipeline.py raw_captures/ --csv logs/final_run_dec.csv --json logs/final_run_dec.json
```