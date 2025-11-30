from pathlib import Path 


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Path to data
DATA_DIR = PROJECT_ROOT / "data"
EXCEL_PATH = DATA_DIR / "job_data.xlsx"

TASTES_EMBED_PATH = DATA_DIR / "tastes_embeddings.pkl"


