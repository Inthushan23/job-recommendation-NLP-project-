from fastapi import FastAPI
from models.book import Book


books_service = FastAPI()

books = [{"id": 1, "title": "test", "writer": "Inthu", "year": 2025}]

@books_service.get("/books")
def get_books():
    return books

@books_service.post("/books")
def add_book(book: Book):

    book = {"id": book.id, "title" : book.title, "writer": book.writer, "year": book.year}
    books.append(book)
    
    return book 

@books_service.put("/books/{id}")
def modif_book(id: int, title: str, writer: str, year: int):
    for elt in books:
        if elt["id"] == id:
            new = {"id": id, "title": title, "writer": writer, "year": year}
            books.remove(elt)
            books.append(new)
            return new 
    return {"Error": "The given id doesnt exist"}