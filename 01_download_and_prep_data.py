import os
import urllib.request
import zipfile
import subprocess
import shutil

# --- Configuration ---
DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
ZIP_PATH = "data/DIV2K_valid_HR.zip"
EXTRACT_DIR = "data/raw_div2k"
OUT_MSTBIC_DIR = "data/MSTbic_100"

def main():
    """
    Step 1: Download exactly 100 high-resolution images from the official DIV2K valid set.
    These are all modern photos - absolutely no banned images (like Lenna) are in this dataset.
    """
    os.makedirs("data", exist_ok=True)
    
    print("\n--- Phase 1: Downloading 100 HR Images (DIV2K Validation Set) ---")
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading {DIV2K_URL} (This is ~700MB, it might take a moment...)")
        urllib.request.urlretrieve(DIV2K_URL, ZIP_PATH)
        print("Download complete!")
    else:
        print("DIV2K zip file already exists locally.")

    print("\n--- Phase 2: Extracting dataset ---")
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting images...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print(f"Extracted to {EXTRACT_DIR}")
        
        # Move images from the nested 'DIV2K_valid_HR' folder out directly to raw_div2k
        nested_dir = os.path.join(EXTRACT_DIR, "DIV2K_valid_HR")
        if os.path.exists(nested_dir):
            for file in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, file), os.path.join(EXTRACT_DIR, file))
            os.rmdir(nested_dir)
    else:
        print("Dataset already extracted.")

    print("\n--- Phase 3: Generating pseudo-LR pairs (MSTbic Training Format) ---")
    # Using your existing preparation script to generate the downsampled pairs
    # Because training a model on 100 images requires a lot of patches, we'll extract
    # 256x256 crops.
    
    if os.path.exists(os.path.join(OUT_MSTBIC_DIR, "train", "HR")):
        print(f"MSTbic dataset already prepared at {OUT_MSTBIC_DIR}. Skipping generation.")
        return

    # Path detection logic for Kaggle/Remote environments
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prep_script = os.path.join(base_dir, "src", "mikrosr", "dataprep", "prepare_mstbic_dataset.py")
    
    # Ensure all data paths are also relative to base_dir if needed, 
    # but for Kaggle, relative to CWD is usually fine if we cd in.
    
    # Run the existing data preparation tools
    cmd = [
        "python3", prep_script,
        "--src", EXTRACT_DIR, 
        "--out", OUT_MSTBIC_DIR,
        "--scale", "4",
        "--patch", "256",
        # Splitting the 100 images into train (85), val (10), test (5)
        "--splits", "0.85", "0.10", "0.05"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✅ Dataset successfully prepared in: {OUT_MSTBIC_DIR}")
    else:
        print("\n❌ Dataset compilation failed. Check the error log above.")

if __name__ == "__main__":
    main()
