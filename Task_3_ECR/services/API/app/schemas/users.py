import re

from pydantic import BaseModel, EmailStr, validator

# Regex to validate cirruslabs.io email addresses
EMAIL_REGEX = r"^[a-zA-Z0-9._-]+@cirruslabs\.io$"


class UserCreate(BaseModel):

    name: str
    username: str
    email: EmailStr
    password: str
    tenant_id: int

    @validator("name")
    def name_must_be_alpha(cls, v):
        if not v or not re.match(r"^[A-Za-z]+(?: [A-Za-z]+)*$", v.strip()):
            raise ValueError("Name must contain only alphabets and single spaces between names")
        return v.strip()

    @validator("username")
    def username_must_be_valid(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Username is required and cannot be empty.")

        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError(
                "Username must contain only letters, numbers, and underscores (no spaces or special characters)."
            )

        return v

    @validator("tenant_id")
    def tenant_id_must_be_digit(cls, v):
        if not isinstance(v, int) or v < 0:
            raise ValueError("Tenant ID must be a positive integer")
        return v

    @validator("email")
    def validate_cirrus_email(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Email is required and cannot be empty.")

        if not re.match(EMAIL_REGEX, v):
            raise ValueError(
                "Email must be a valid cirruslabs.io address and can only contain letters, numbers, dots (.), underscores (_), and hyphens (-) before the '@'. Special characters like '+', '%', or '#' are not allowed."
            )
        return v

    @validator("password")
    def password_must_be_valid(cls, v):
        if not v or len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserRead(BaseModel):
    id: int
    email: EmailStr
    username: str
    tenant_id: int

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    email: str
    password: str
