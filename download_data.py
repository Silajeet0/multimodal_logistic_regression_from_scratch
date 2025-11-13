import os
import requests
import gzip
import shutil

# Define the folder to save in
SAVE_DIR = "Dataset/fashionmnist"

# List of files to download
FILES_TO_DOWNLOAD = [
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
]

# Create the directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)


def download_and_unzip():
    """
    Downloads and unzips the Fashion-MNIST dataset.
    """
    print(f"--- Data Downloader ---")
    print(f"Saving files to: {os.path.abspath(SAVE_DIR)}")

    for url in FILES_TO_DOWNLOAD:
        # Get the .gz filename (e.g., 'train-images-idx3-ubyte.gz')
        gz_filename = url.split('/')[-1]
        gz_filepath = os.path.join(SAVE_DIR, gz_filename)

        # Get the final unzipped filename (e.g., 'train-images')
        # rename them to match what the main.py expects
        if "train-images" in gz_filename:
            final_name = "train-images"
        elif "train-labels" in gz_filename:
            final_name = "train-labels"
        elif "t10k-images" in gz_filename:
            final_name = "t10k-images"
        elif "t10k-labels" in gz_filename:
            final_name = "t10k-labels"

        final_filepath = os.path.join(SAVE_DIR, final_name)

        # Check if the final file already exists
        if os.path.exists(final_filepath):
            print(f"Skipping '{gz_filename}', final file already exists.")
            continue

        # Download the file
        try:
            print(f"Downloading '{gz_filename}'...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(gz_filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            continue

        # Unzip the file
        try:
            print(f"Unzipping '{gz_filename}'...")
            with gzip.open(gz_filepath, 'rb') as f_in:
                with open(final_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print(f"Error unzipping {gz_filepath}: {e}")

        # Clean up the .gz file
        try:
            os.remove(gz_filepath)
        except Exception as e:
            print(f"Error removing {gz_filepath}: {e}")

    print("--- Download complete! ---")

if __name__ == "__main__":
    download_and_unzip()