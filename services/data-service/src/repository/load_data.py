import pandas as pd 
import pickle

from ..config import EXCEL_PATH, TASTES_EMBED_PATH, DATA_DIR


def load_file():

    tastes, questions, skills = (pd.read_excel(EXCEL_PATH, sheet_name = 0), 
                                pd.read_excel(EXCEL_PATH, sheet_name = 1),
                                pd.read_excel(EXCEL_PATH, sheet_name = 2))
    return tastes, questions, skills


def load_embeddings_tastes():

    try:
        with open(TASTES_EMBED_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Fichier {TASTES_EMBED_PATH} non trouvé")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return None


def load_embeddings_skills(domain: str): 
    path = DATA_DIR /f"skills_embeddings_{domain}.pkl"

    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Fichier {path} non trouvé")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return None

