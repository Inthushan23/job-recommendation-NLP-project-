import os
from pathlib import Path 
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# S3 configuration
BUCKET_NAME = os.getenv("BUCKET_NAME", "s3-g3mg01")
S3_DATA_FOLDER = os.getenv("S3_DATA_FOLDER", "data/") 

# Default Excel filename
EXCEL_FILENAME = os.getenv("EXCEL_FILENAME", "my_data.xlsx")

# Define local project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
