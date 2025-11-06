from pydantic import BaseModel

class Book(BaseModel):
    id: int
    title: str 
    writer: str 
    year: int