from fastapi import APIRouter
import httpx 
from config import MICROSERVICES

router = APIRouter(prefix="/books")

@router.get("/")
async def get_books():
    async with httpx.AsyncClient() as client:
        result = await client.get(f"{MICROSERVICES['books']}/books")
        return result.json()
    
@router.post("/")  
async def add_book(book: dict):
    async with httpx.AsyncClient() as client:
        result = await client.post(f"{MICROSERVICES['books']}/books", json= book)
        return result.json()

@router.put("/{id}")
async def modif_book(id: int, title: str, writer: str, year: int):
    async with httpx.AsyncClient() as client:
        result = await client.put(f"{MICROSERVICES['books']}/books/{id}", params={ "title": title, "writer": writer, "year": year})
        return result.json()