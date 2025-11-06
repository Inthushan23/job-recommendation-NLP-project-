from fastapi import FastAPI 
from routers import books_router, users_router, loan_router

app = FastAPI()

app.include_router(books_router.router)
app.include_router(users_router.router)
app.include_router(loan_router.router)
