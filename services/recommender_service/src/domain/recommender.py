import numpy as np

from ....data_service.src.repository.load_data import load_file


class Recommender:

    def __init__(self):
        self._tastes_df, self._questions_df, self._skills_df = load_file()  
        self._domain = None

    @property
    def tastes_df(self):
        return self._tastes_df
    
    @property
    def questions_df(self):
        return self._questions_df
    
    @property
    def skills_df(self):
        return self._skills_df
    
    @property
    def domain(self):
        return self._domain
    @domain.setter
    def domain(self, val):
        self._domain =  val
    

    # Fonction pour la partie 1 (choix du domaine)
    def get_domain(self, sim_like, sim_dislike):

        df_tastes = self.tastes_df.copy()
        df_tastes["sim1"] = sim_like
        df_tastes["sim2"] = np.where(sim_dislike > 0, sim_dislike, - sim_dislike)
        df_tastes["Score"] = df_tastes["sim1"] - df_tastes["sim2"]
        df_tastes["Score"] = df_tastes["sim1"] - df_tastes["sim2"]
        df_tastes = df_tastes.sort_values(by="Score", ascending=False).reset_index(drop=True)

        self.domain(df_tastes.loc[0, "Domain"])
        return df_tastes
    
    def filter_domain(self):
        df_skills = self.skills_df
        Skills_domain = df_skills[df_skills["Domain"] == st.session_state["domain"]]
        
    

