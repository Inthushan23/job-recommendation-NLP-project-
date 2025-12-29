import os
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME", "s3-g3mg01")
# On définit un dossier racine pour les données dans S3
S3_DATA_FOLDER = os.getenv("S3_DATA_FOLDER", "data/") 
# Le nom du fichier Excel seul
EXCEL_FILENAME = os.getenv("EXCEL_FILENAME", "my_data.xlsx")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"