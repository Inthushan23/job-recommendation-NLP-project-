from fastapi import FastAPI
import httpx
from config import MICROSERVICES
from models.loan import Loan

books_service = FastAPI()

loans = []

@books_service.get("/loans")
def get_loans():
    return loans


@books_service.post("/loans")
async def create_loans(user_id: int, book_id: int):
    book = None 
    user = None
    
    async with httpx.AsyncClient() as client:
        res1 = await client.get(f"{MICROSERVICES['books']}/books")
        res2 = await client.get(f"{MICROSERVICES['users']}/users")
        books = res1.json()
        users = res2.json()


    for bk in books:
        if bk["id"] == book_id:
            book = bk 
            break 

    for ur in users:
        if ur["id"] == user_id:
            user = ur 
            break

    if book and user:
        l = {"user_id": user_id, "book_id": book_id, "title": book["title"], "writer": book["writer"], "year": book["year"], "user_name": user["name"], "email": user["email"]}
        loans.append(l)
        return l 
    return {"Error": "This book has not been borrowed !"}





    
    
    
    
