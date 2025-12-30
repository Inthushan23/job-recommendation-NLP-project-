from sentence_transformers import SentenceTransformer, util
from ..config import ENCODER_MODEL

class LoadModel:
    _model = None 
    
    # Return the model instance, load if not already loaded
    def get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(ENCODER_MODEL)
        return cls._model