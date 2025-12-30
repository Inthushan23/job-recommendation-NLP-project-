from .load_model import LoadModel
from sentence_transformers import util

class Encoder:

    # Load the sentence transformer model
    def __init__(self):
        self.model = LoadModel().get_model()

    # Encode input text(s) into embeddings
    def encode(self, ipt: str, conv_to_tensor: bool = True):
        return self.model.encode(ipt, convert_to_tensor=conv_to_tensor)

    # Compute cosine similarity between two embeddings
    def cosine_similarity(self, a, b):
        return util.cos_sim(a, b)[0].numpy()
