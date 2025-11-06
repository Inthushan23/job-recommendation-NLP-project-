from pydantic import BaseModel

class Loan(BaseModel):
    title: str 
    writer: str 
    year: int
    name: str 
    email: str 
