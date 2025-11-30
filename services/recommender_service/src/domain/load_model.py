from sentence_transformers import SentenceTransformer, util

class LoadModel:
    _model = None 
    
    def get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer("usc-isi/sbert-roberta-large-anli-mnli-snli")
        return cls._model
