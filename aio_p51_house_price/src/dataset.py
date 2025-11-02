from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import gdown  # Import gdown library to download files from Google Drive

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # Import predefined data directories

app = typer.Typer()  # Create a Typer CLI app instance

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR,  # Default input path for downloaded file
    #output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",  # Default output path for processed data
    file_id: str = "1Dh_y7gFDUa2sD72_cKIa209dhbMVoGEd",  # Google Drive file ID to download (default example)
):
    logger.info(f"Downloading file from Google Drive ID={file_id} to {input_path}")
    url = f"https://drive.google.com/uc?id={file_id}"  # Construct Google Drive download URL
    gdown.download(url, str(input_path) + "/", quiet=False)  # Download the file and save it at input_path

    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):  # Example progress bar for processing steps
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()  # Run the Typer CLI application