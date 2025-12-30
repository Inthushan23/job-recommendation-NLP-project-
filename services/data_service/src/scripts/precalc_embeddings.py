import pickle
from io import BytesIO
import boto3

from ..repository.load_data import load_file
from services.recommender_service.src.domain.encoder import Encoder
from ..domain.data_processing import ProcessData

from ..config import BUCKET_NAME, S3_DATA_FOLDER 

# Initialize S3 client
s3 = boto3.client("s3")


# Helper function to upload bytes data to S3
def upload_to_s3(data: bytes, key: str):
    """Upload an in-memory object to S3"""
    full_key = f"{S3_DATA_FOLDER}{key}"
    s3.put_object(Bucket=BUCKET_NAME, Key=full_key, Body=data)
    print(f"File uploaded to S3: {full_key}")


# Precompute embeddings for tastes and skills and store them in S3
def precalculate_embeddings():
    tastes_df, _, skills_df = load_file()  
    encoder = Encoder()
    p = ProcessData

    # Vectorize tastes
    print("Vectorizing Tastes...")
    tastes_embeddings = encoder.encode(
        tastes_df["Tastes"].apply(p.normalize).tolist()
    )

    tastes_data = {
        'embeddings': tastes_embeddings,
        'domains': tastes_df["Domain"].tolist(),
        'tastes_texts': tastes_df["Tastes"].tolist(),
        'dataframe': tastes_df.to_dict()
    }

    # Upload tastes embeddings
    upload_to_s3(pickle.dumps(tastes_data), "tastes_embeddings.pkl")

    # Vectorize skills for each domain
    print("Vectorizing Skills by domain...")
    for domain in skills_df["Domain"].unique():
        domain_skills = skills_df[skills_df["Domain"] == domain].reset_index(drop=True)
        
        print(f"Domain: {domain} - {len(domain_skills)} skills")
        
        skills_embeddings = encoder.encode(
            domain_skills["Skills"].apply(p.normalize).tolist()
        )
        
        skills_data = {
            'embeddings': skills_embeddings,
            'domain': domain,
            'dataframe': domain_skills.to_dict(),
            'jobs': domain_skills["Job"].tolist(),
            'competencies': domain_skills["Competency"].tolist(),
            'skills_texts': domain_skills["Skills"].tolist(),
            'weights': domain_skills["Weight"].tolist() if "Weight" in domain_skills.columns else [1] * len(domain_skills)
        }
        
        filename = f'skills_embeddings_{domain.replace(" ", "_")}.pkl'
        upload_to_s3(pickle.dumps(skills_data), filename)
        
        print(f"{domain}: {len(domain_skills)} skills vectorized → {S3_DATA_FOLDER}{filename}")
    
    print(f"Processed domains: {list(skills_df['Domain'].unique())}")
