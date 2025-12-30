import pandas as pd 
import pickle
import boto3
from botocore.exceptions import ClientError
from io import BytesIO

from ..config import BUCKET_NAME, S3_DATA_FOLDER

# Initialize S3 client
s3 = boto3.client('s3')


# Load Excel files for tastes, questions, and skills
def load_file():
    obj_tastes = s3.get_object(Bucket=BUCKET_NAME, Key=f"{S3_DATA_FOLDER}Tastes.xlsx")
    tastes = pd.read_excel(BytesIO(obj_tastes['Body'].read()))
    
    obj_questions = s3.get_object(Bucket=BUCKET_NAME, Key=f"{S3_DATA_FOLDER}Questions.xlsx")
    questions = pd.read_excel(BytesIO(obj_questions['Body'].read()))
    
    obj_skills = s3.get_object(Bucket=BUCKET_NAME, Key=f"{S3_DATA_FOLDER}Skills.xlsx")
    skills = pd.read_excel(BytesIO(obj_skills['Body'].read()))
    
    return tastes, questions, skills


# Generic loader for pickled objects from S3
def load_from_s3(filename: str):
    key = f"{S3_DATA_FOLDER}{filename}"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return pickle.loads(obj["Body"].read())
    except ClientError as e:
        print(f"ERREUR S3: Impossible de charger {key}. Erreur: {e}")
        return None 


# Load tastes embeddings
def load_embeddings_tastes():
    return load_from_s3("tastes_embeddings.pkl")


# Load skills embeddings for a specific domain
def load_embeddings_skills(domain: str):
    filename = f"skills_embeddings_{domain.replace(' ', '_')}.pkl"
    return load_from_s3(filename)
