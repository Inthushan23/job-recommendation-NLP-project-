import pandas as pd 
import pickle

import boto3
from botocore.exceptions import ClientError
from io import BytesIO

from ..config import BUCKET_NAME, DATA_KEY

s3 = boto3.client('s3')

def load_file():
    
    obj = s3.get_object(Bucket= BUCKET_NAME, Key= DATA_KEY)
    
    file = BytesIO(obj['Body'].read())

    tastes, questions, skills = (pd.read_excel(file, sheet_name = 0), 
                                pd.read_excel(file, sheet_name = 1),
                                pd.read_excel(file, sheet_name = 2))
    return tastes, questions, skills


def load_from_s3(filename: str):
    key = f"{DATA_KEY}{filename}"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pickle.loads(obj["Body"].read())
    except ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            print(f"Fichier {key} non trouvé dans S3")
        else:
            print(f"Erreur lors du chargement de {key}: {e}")
        return None

def load_embeddings_tastes():
    return load_from_s3("tastes_embeddings.pkl")

def load_embeddings_skills(domain: str):
    filename = f"skills_embeddings_{domain.replace(' ', '_')}.pkl"
    return load_from_s3(filename)