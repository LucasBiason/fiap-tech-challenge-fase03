"""
Dataset Downloader for FIAP Tech Challenge Phase 3
Handles downloading and extracting the dataset from Google Drive
"""

import os
import zipfile
import gzip
import shutil
import gdown
from typing import Optional


class DatasetDownloader:
    """Handles dataset downloading and extraction from Google Drive."""

    def __init__(self, base_path: str, file_id: str,
                 dataset_url: Optional[str] = None):
        """
        Initialize downloader with base path and Google Drive file ID.

        Args:
            base_path: Base directory path for storing files
            file_id: Google Drive file ID
            dataset_url: Optional full URL (for reference)
        """
        self.base_path = base_path
        self.file_id = file_id
        self.dataset_url = dataset_url

        # Ensure base directory exists
        os.makedirs(base_path, exist_ok=True)

    def download_and_extract(self, output_json_path: str) -> bool:
        """
        Download ZIP file, extract, and decompress JSON.GZ to JSON.
        
        Args:
            output_json_path: Final path for the JSON dataset
            
        Returns:
            bool: True if successful, False otherwise
        """
        if os.path.exists(output_json_path):
            print(f"Dataset already exists: {output_json_path}")
            return self._verify_json_file(output_json_path)
        
        print("Starting dataset download and extraction process...")
        
        # Step 1: Download ZIP file
        zip_path = os.path.join(self.base_path, "dataset.zip")
        if not self._download_zip(zip_path):
            return False
        
        # Step 2: Extract ZIP file
        json_gz_path = self._extract_zip(zip_path)
        if not json_gz_path:
            return False
        
        # Step 3: Decompress JSON.GZ to JSON
        if not self._decompress_json_gz(json_gz_path, output_json_path):
            return False
        
        # Step 4: Clean up temporary files
        self._cleanup_temp_files(zip_path, json_gz_path)
        
        # Step 5: Verify final file
        return self._verify_json_file(output_json_path)

    def _download_zip(self, zip_path: str) -> bool:
        """Download ZIP file from Google Drive."""
        try:
            print("Downloading ZIP file from Google Drive...")
            download_url = (f"https://drive.google.com/uc?"
                           f"export=download&id={self.file_id}")
            gdown.download(download_url, zip_path, quiet=False)
            
            if os.path.exists(zip_path):
                size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                print(f"ZIP downloaded successfully: {size_mb:.1f} MB")
                return True
            else:
                print("ERROR: ZIP file not downloaded")
                return False
                
        except Exception as e:
            print(f"Error downloading ZIP: {e}")
            return False

    def _extract_zip(self, zip_path: str) -> Optional[str]:
        """Extract ZIP file and find JSON.GZ file."""
        try:
            extract_path = os.path.join(self.base_path, "temp_extract")
            os.makedirs(extract_path, exist_ok=True)
            
            print("Extracting ZIP file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
                file_list = zip_ref.namelist()
                print(f"Files in ZIP: {file_list}")
            
            # Find JSON.GZ files
            json_gz_files = [f for f in file_list if f.endswith('.json.gz')]
            print(f"JSON.GZ files found: {json_gz_files}")
            
            if json_gz_files:
                json_gz_path = os.path.join(extract_path, json_gz_files[0])
                print(f"Using JSON.GZ file: {json_gz_path}")
                return json_gz_path
            else:
                print("ERROR: No JSON.GZ files found!")
                return None
                
        except Exception as e:
            print(f"Error extracting ZIP: {e}")
            return None

    def _decompress_json_gz(self, json_gz_path: str, output_path: str) -> bool:
        """Decompress JSON.GZ file to JSON."""
        try:
            print("Decompressing JSON.GZ to JSON...")
            
            with gzip.open(json_gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"JSON decompressed successfully: {size_mb:.1f} MB")
                return True
            else:
                print("ERROR: Decompressed JSON not found")
                return False
                
        except Exception as e:
            print(f"Error decompressing JSON.GZ: {e}")
            return False

    def _cleanup_temp_files(self, zip_path: str, json_gz_path: str) -> None:
        """Clean up temporary files to save space."""
        try:
            # Remove ZIP file
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print("ZIP file cleaned up")
            
            # Remove extraction directory
            extract_dir = os.path.dirname(json_gz_path)
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
                print("Temporary extraction directory cleaned up")
                
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")

    def _verify_json_file(self, json_path: str) -> bool:
        """Verify that the file is a valid JSON."""
        try:
            print("Verifying JSON file format...")
            
            with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
                first_line = f.readline().strip()
                
                if first_line.startswith('{'):
                    print("Format: JSON Lines (JSONL) - each line is a JSON object")
                    return True
                elif first_line.startswith('['):
                    print("Format: JSON Array - single array with objects")
                    return True
                else:
                    print(f"ERROR: Not a valid JSON format. "
                          f"First line: {first_line[:50]}...")
                    return False
                    
        except Exception as e:
            print(f"Error verifying JSON: {e}")
            return False

    def get_file_info(self, file_path: str) -> dict:
        """Get detailed information about the dataset file."""
        if not os.path.exists(file_path):
            return {"exists": False}
        
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        
        # Count lines efficiently
        line_count = 0
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line_count += 1
        
        return {
            "exists": True,
            "size_mb": size_mb,
            "size_bytes": size_bytes,
            "total_lines": line_count,
            "path": file_path
        }


def download_dataset(base_path: str, output_json_path: str,
                     file_id: str,
                     dataset_url: Optional[str] = None) -> bool:
    """
    Convenience function to download and prepare dataset.
    
    Args:
        base_path: Base directory for temporary files
        output_json_path: Final path for JSON dataset
        file_id: Google Drive file ID
        dataset_url: Optional full URL (for reference)
        
    Returns:
        bool: True if successful, False otherwise
    """
    downloader = DatasetDownloader(base_path, file_id, dataset_url)
    return downloader.download_and_extract(output_json_path)

