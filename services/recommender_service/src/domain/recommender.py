import pandas as pd
import numpy as np

from services.data_service.src.repository.load_data import load_file, load_embeddings_tastes, load_embeddings_skills
from services.data_service.src.domain.data_processing import ProcessData
from .encoder import Encoder

class Recommender:

    # Initialize and load data/embeddings
    def __init__(self):
        self._tastes_df, self._questions_df, self._skills_df = load_file()  
        self._embeddings_tastes = load_embeddings_tastes()
        self._domain = None

    # Properties for easy access to dataframes and embeddings
    @property
    def tastes_df(self):
        return self._tastes_df if not self._tastes_df.empty else pd.DataFrame()
    
    @property
    def questions_df(self):
        return self._questions_df if not self._questions_df.empty else pd.DataFrame()
    
    @property
    def skills_df(self):
        return self._skills_df if not self._skills_df.empty else pd.DataFrame()
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def embeddings_tastes(self):
        return self._embeddings_tastes

    # Part 1: Determine domain based on user input
    def get_domain(self, user_input1, user_input2, encoder: Encoder):
        p = ProcessData
        emb_tastes = self.embeddings_tastes
        embedded_tastes = emb_tastes["embeddings"]

        # Clean and embed user inputs
        embedded_user1 = encoder.encode(p.normalize(user_input1), conv_to_tensor=True)
        embedded_user2 = encoder.encode(p.normalize(user_input2), conv_to_tensor=True)
        
        # Compute cosine similarities
        similarities1 = encoder.cosine_similarity(embedded_user1, embedded_tastes)
        similarities2 = encoder.cosine_similarity(embedded_user2, embedded_tastes)

        df_tastes = self.tastes_df.copy()
        df_tastes["sim1"] = similarities1
        df_tastes["sim2"] = np.abs(similarities2)
        df_tastes["Score"] = df_tastes["sim1"] - df_tastes["sim2"]
        df_tastes = df_tastes.sort_values(by="Score", ascending=False).reset_index(drop=True)

        return df_tastes, df_tastes.loc[0, "Domain"]
    
    # Part 2: Compute similarity based on user responses to questions
    def question_based_sim(self, domain, user_input3, user_input4, user_input5, encoder: Encoder):
        p = ProcessData
        df_skills = self.skills_df.copy()
        skills_domain = df_skills[df_skills["Domain"] == domain] 
        skills_competency = skills_domain[["Competency", "Skills", "Weight"]].reset_index(drop=True)

        skills_data = load_embeddings_skills(domain)
        if skills_data is None or "embeddings" not in skills_data:
            print(f"⚠️ Warning: Embeddings for {domain} not found in S3.")
            return pd.DataFrame(), pd.DataFrame()
        
        embedded_skills = skills_data["embeddings"]

        # Encode user responses
        embedded_user3 = encoder.encode(p.normalize(user_input3), conv_to_tensor=True)
        embedded_user4 = encoder.encode(p.normalize(user_input4), conv_to_tensor=True)
        embedded_user5 = encoder.encode(p.normalize(user_input5), conv_to_tensor=True)
        
        similarities3 = encoder.cosine_similarity(embedded_user3, embedded_skills)
        similarities4 = encoder.cosine_similarity(embedded_user4, embedded_skills)
        similarities5 = encoder.cosine_similarity(embedded_user5, embedded_skills)
        
        # Compute weighted scores per skill
        skills_competency["sim3"] = similarities3
        skills_competency["sim4"] = similarities4
        skills_competency["sim5"] = similarities5
        
        total_similarity = similarities3 + similarities4 + similarities5
        weighted_scores = total_similarity * skills_competency["Weight"]
        total_weight = skills_competency["Weight"].sum()
        skills_competency["Score"] = weighted_scores / total_weight
        
        # Sort skills by score
        skills_competency = (
            skills_competency
            .sort_values(by="Score", ascending=False)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        
        # Aggregate scores by job
        jobs_competencies = skills_domain.groupby("Job").apply(
            lambda x: pd.Series({
                'Competency': ", ".join(x['Competency'].astype(str)),
                'Weights': dict(zip(x['Competency'], x['Weight']))
            })
        ).reset_index()
    
        jobs_competencies["Score"] = 0.0
        for competency, score in zip(skills_competency["Competency"], skills_competency["Score"]):
            for index, row in jobs_competencies.iterrows():
                job_skills = row['Competency'].split(', ')
                if competency in job_skills:
                    jobs_competencies.at[index, 'Score'] += score
    
        jobs_competencies = jobs_competencies.sort_values(by="Score", ascending=False).reset_index(drop=True)
        
        return jobs_competencies, skills_competency
