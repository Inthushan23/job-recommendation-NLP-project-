from contextlib import asynccontextmanager
from fastapi import FastAPI
from .api.routes import router
from .domain.recommender import Recommender
from .domain.encoder import Encoder

# Lifespan context to initialize encoder and recommender
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize encoder
    app.state.encoder = Encoder()
    
    # Load recommender system
    try:
        app.state.recommender = Recommender()
    except Exception as e:
        print(f"{e}")
    
    yield
    print("SHUTDOWN")

# Create FastAPI app and include routes
app = FastAPI(lifespan=lifespan)
app.include_router(router)
