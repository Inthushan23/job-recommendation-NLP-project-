import pandas as pd 
import pickle
import boto3
from botocore.exceptions import ClientError
from io import BytesIO

# Import des nouvelles variables
from ..config import BUCKET_NAME, S3_DATA_FOLDER, EXCEL_FILENAME

s3 = boto3.client('s3')

def load_file():
    # Construction propre du chemin du fichier Excel
    key = f"{S3_DATA_FOLDER}{EXCEL_FILENAME}"
    print(f"Chargement du fichier Excel depuis S3: {key}") # Debug log
    
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    file = BytesIO(obj['Body'].read())

    tastes, questions, skills = (pd.read_excel(file, sheet_name=0), 
                                pd.read_excel(file, sheet_name=1),
                                pd.read_excel(file, sheet_name=2))
    return tastes, questions, skills

def load_from_s3(filename: str):
    # Construction propre du chemin pour les pickles
    key = f"{S3_DATA_FOLDER}{filename}"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pickle.loads(obj["Body"].read())
    except ClientError as e:
        print(f"ERREUR S3: Impossible de charger {key}. Erreur: {e}")
        return None # Attention: Si ton main n'attend pas None, ça plantera ici

def load_embeddings_tastes():
    return load_from_s3("tastes_embeddings.pkl")

def load_embeddings_skills(domain: str):
    filename = f"skills_embeddings_{domain.replace(' ', '_')}.pkl"
    return load_from_s3(filename)