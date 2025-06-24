import requests
import zipfile
import os
import tempfile
import shutil
from pathlib import Path

def download_and_extract_polybox_data(url='https://polybox.ethz.ch/index.php/s/R99PSQwT9e9CjYy/download'):
    """
    Downloads a zip file from Polybox and extracts all files directly to the src/data directory.
    
    Args:
        url (str): URL to the zip file
    """
    # Define the data directory path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src/data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the zip file
    print(f"Downloading zip file from {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")
    
    # Save the zip file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
        temp_file_path = temp_file.name
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
    
    # Create a temporary directory for extraction
    temp_extract_dir = tempfile.mkdtemp()
    
    try:
        # Extract the zip file to the temporary directory
        print(f"Extracting zip file...")
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        
        # Move all files from any subdirectories directly to the data directory
        file_count = 0
        for root, _, files in os.walk(temp_extract_dir):
            for file in files:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(data_dir, file)
                shutil.copy2(source_path, dest_path)
                print(f"Copied: {file}")
                file_count += 1
        
        print(f"Extracted {file_count} files to {data_dir}")
    
    finally:
        # Clean up temporary files
        os.unlink(temp_file_path)
        shutil.rmtree(temp_extract_dir)
    
    print(f"All files have been extracted directly to {data_dir}")

# Example usage:
if __name__ == "__main__":
    download_and_extract_polybox_data()