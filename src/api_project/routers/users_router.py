import httpx 
from fastapi import APIRouter
from config import MICROSERVICES

router = APIRouter(prefix="/users")


@router.get("/")
async def get_users():
    async with httpx.AsyncClient() as client:
        result = await client.get(f"{MICROSERVICES['users']}/users")
        return result.json()
    
@router.post("/")
async def add_user(user: dict):
    async with httpx.AsyncClient() as client: 
        result = await client.post(f"{MICROSERVICES['users']}/users", json=user)
        return result.json()