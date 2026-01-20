from typing import Optional

from pydantic import BaseModel


class CreatePlane(BaseModel):
    name: str
    price: float
    user_limit: int
    role_limit: int
    description: Optional[str] = None


class ReadPlane(BaseModel):
    id: int
    name: str
    price: float
    user_limit: int
    role_limit: int
    description: Optional[str] = None

    class Config:
        from_attributes = True
