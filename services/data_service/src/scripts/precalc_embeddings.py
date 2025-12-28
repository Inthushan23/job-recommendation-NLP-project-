import pickle

from ..repository.load_data import load_file
from services.recommender_service.src.domain.encoder import Encoder
from ..domain.data_processing import ProcessData

from ..config import DATA_DIR


# Le lancer qu'une fois au début
def precalculate_embeddings():
    tastes_df, _, skills_df = load_file()  
    encoder = Encoder()
    p = ProcessData

    print("Vectorisation des Tastes...")
    tastes_embeddings = encoder.encode(
        tastes_df["Tastes"].apply(p.normalize).tolist()
    )

    tastes_data = {
        'embeddings': tastes_embeddings,
        'domains': tastes_df["Domain"].tolist(),
        'tastes_texts': tastes_df["Tastes"].tolist(),
        'dataframe': tastes_df.to_dict()
    }

    with open(DATA_DIR / 'tastes_embeddings.pkl', 'wb') as f:
        pickle.dump(tastes_data, f)

    print("Vectorisation des Skills par domaine...")
    for domain in skills_df["Domain"].unique():
        domain_skills = skills_df[skills_df["Domain"] == domain].reset_index(drop=True)
        
        print(f"Domain: {domain} - {len(domain_skills)} compétences")
        
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
        
        # Sauvegarder
        filename = DATA_DIR / f'skills_embeddings_{domain.replace(" ", "_")}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(skills_data, f)
        
        print(f"{domain}: {len(domain_skills)} skills vectorisés → {filename}")
    
    print(f"Domaines traités: {list(skills_df['Domain'].unique())}")
