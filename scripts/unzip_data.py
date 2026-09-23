"""Automated zip dataset extractor and importer."""
import os
import zipfile
from pathlib import Path

def extract_all_zip_files(search_dir="."):
    """Find and extract any zip file found in root or data directories into data/raw."""
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files_found = []
    for root, _, files in os.walk(search_dir):
        if ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".zip"):
                zip_path = Path(root) / file
                zip_files_found.append(zip_path)
                print(f"Found zip archive: {zip_path}")
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                    print(f"Successfully extracted {zip_path} into {target_dir}")
                    
    if not zip_files_found:
        print("No .zip files found in directory yet. Drop your dataset .zip file anywhere in the repo and run python scripts/unzip_data.py!")

if __name__ == "__main__":
    extract_all_zip_files()
