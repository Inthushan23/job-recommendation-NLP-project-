from fastapi import APIRouter
import httpx 
from config import MICROSERVICES

router = APIRouter(prefix="/loans")


@router.get("/")
async def get_loan():
    async with httpx.AsyncClient() as client:
        result = await client.get(f"{MICROSERVICES['loans']}/loans")
        return result.json()

@router.post("/")
async def create_loans(user_id:int, book_id: int):
    async with httpx.AsyncClient() as client:
        result = await client.post(f"{MICROSERVICES['loans']}/loans", params={"user_id": user_id, "book_id": book_id})
        return result.json()
