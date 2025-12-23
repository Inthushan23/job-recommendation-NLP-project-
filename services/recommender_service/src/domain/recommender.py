import pandas as pd
import numpy as np

from ....data_service.src.repository.load_data import load_file, load_embeddings_tastes, load_embeddings_skills
from ....data_service.src.domain.data_processing import ProcessData
from .encoder import Encoder

 
class Recommender:

    def __init__(self):
        self._tastes_df, self._questions_df, self._skills_df = load_file()  
        self._embeddings_tastes = load_embeddings_tastes()
        self._domain = None

    @property
    def tastes_df(self):
        if self._tastes_df.empty:
            return pd.DataFrame()
        return self._tastes_df
    
    @property
    def questions_df(self):
        if self._questions_df.empty:
            return pd.DataFrame()
        return self._questions_df
    
    @property
    def skills_df(self):
        if self._skills_df.empty:
            return pd.DataFrame()
        return self._skills_df
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def embeddings_tastes(self):
        return self._embeddings_tastes

    # Fonction pour la partie 1 (choix du domaine)
    def get_domain(self, user_input1, user_input2, encoder: Encoder):
        p = ProcessData
        emb_tastes = self.embeddings_tastes
        embedded_tastes = emb_tastes["embeddings"]

        # Clean and embed user inputs
        cleaned_user_input1 = p.normalize(user_input1)
        cleaned_user_input2 = p.normalize(user_input2)

        embedded_user1 = encoder.encode(cleaned_user_input1, conv_to_tensor=True)
        embedded_user2 = encoder.encode(cleaned_user_input2, conv_to_tensor=True)
        
        # Compute cosine similarity
        similarities1 = encoder.cosine_similarity(embedded_user1, embedded_tastes)
        similarities2 = encoder.cosine_similarity(embedded_user2, embedded_tastes)

        df_tastes = self.tastes_df.copy()
        df_tastes["sim1"] = similarities1
        df_tastes["sim2"] = np.abs(similarities2)
        df_tastes["Score"] = df_tastes["sim1"] - df_tastes["sim2"]
        df_tastes = df_tastes.sort_values(by="Score", ascending=False).reset_index(drop=True)

        # self.domain(df_tastes.loc[0, "Domain"])
        return df_tastes, df_tastes.loc[0, "Domain"]
    

    # Partie 2 réponses aux questions
    def question_based_sim(self, domain, user_input3, user_input4, user_input5, encoder: Encoder):
        p = ProcessData
        df_skills = self.skills_df.copy()
        skills_domain = df_skills[df_skills["Domain"] == domain] 
        skills_competency = skills_domain[["Competency", "Skills", "Weight"]].reset_index(drop=True)

        skills_data = load_embeddings_skills(domain)
        embedded_skills = skills_data["embeddings"]

        cleaned_user_input3 = p.normalize(user_input3)
        cleaned_user_input4 = p.normalize(user_input4)
        cleaned_user_input5 = p.normalize(user_input5)
        
        embedded_user3 = encoder.encode(cleaned_user_input3, conv_to_tensor=True)
        embedded_user4 = encoder.encode(cleaned_user_input4, conv_to_tensor=True)
        embedded_user5 = encoder.encode(cleaned_user_input5, conv_to_tensor=True)
        
        similarities3 = encoder.cosine_similarity(embedded_user3, embedded_skills)
        similarities4 = encoder.cosine_similarity(embedded_user4, embedded_skills)
        similarities5 = encoder.cosine_similarity(embedded_user5, embedded_skills)
        
        # Weighted score per skill
        skills_competency["sim3"] = similarities3
        skills_competency["sim4"] = similarities4
        skills_competency["sim5"] = similarities5
        
        total_similarity = similarities3 + similarities4 + similarities5
        weighted_scores = total_similarity * skills_competency["Weight"]
        total_weight = skills_competency["Weight"].sum()
        skills_competency["Score"] = weighted_scores / total_weight
        
        # Sort by score
        skills_competency = (
            skills_competency
            .sort_values(by="Score", ascending=False)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        
        # Aggregate results by job
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
