from services.data_service.src.repository.load_data import load_embeddings_tastes, load_embeddings_skills


# Check if the tastes embeddings file is loaded correctly
def test_load_embeddings_tastes():
    emb_tastes = load_embeddings_tastes()

    assert emb_tastes is not None 
    assert "embeddings" in emb_tastes
    assert len(emb_tastes["embeddings"])>0
    
    
# Check if the skills embeddings file is loaded correctly
def test_load_embeddings_skills():
    domains_list = ["Space", "Art", "Financial", "Nature", "Medicine", "IT"]

    for domain in domains_list:
        emb_skills = load_embeddings_skills(domain)
        
        assert emb_skills is not None 
        assert "embeddings" in emb_skills
        assert len(emb_skills["embeddings"])>0

