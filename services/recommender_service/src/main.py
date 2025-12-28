from .api.routes import router

from .domain.recommender import Recommender
from .domain.encoder import Encoder

from fastapi import FastAPI
from services.data_service.src.scripts.precalc_embeddings import precalculate_embeddings


app = FastAPI()

precalculate_embeddings()

recommender = Recommender()
app.state.recommender = recommender

encoder = Encoder()
app.state.encoder = encoder

app.include_router(router)
