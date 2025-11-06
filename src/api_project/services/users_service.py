from fastapi import FastAPI
from models.user import User 


books_service = FastAPI()

users = []


@books_service.get("/users")
def get_users():
    return users

@books_service.post("/users")
def add_user(user: User):
    user = {"id": user.id, "name": user.name, "email": user.email}
    users.append(user)
    return user