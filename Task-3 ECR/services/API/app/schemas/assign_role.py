from typing import List

from pydantic import BaseModel


class AssignRolesRequest(BaseModel):
    user_id: int
    role_ids: List[int]
