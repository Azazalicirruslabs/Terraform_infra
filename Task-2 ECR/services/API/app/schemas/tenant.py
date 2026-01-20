from pydantic import BaseModel


class CreateTenant(BaseModel):
    name: str
    email: str
    mobile_no: str
    domain: str


class ReadTenant(BaseModel):
    id: int
    name: str
    email: str
    mobile_no: str
    domain: str

    class Config:
        from_attributes = True
