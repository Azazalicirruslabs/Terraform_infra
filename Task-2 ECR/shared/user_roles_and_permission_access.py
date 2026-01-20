from fastapi import Depends, HTTPException

from shared.auth import get_current_user

# This function checks if the current user has the required permission
# It can be used as a dependency in FastAPI routes to enforce permission checks.


def require_permission(permission: str):
    def wrapper(current_user=Depends(get_current_user)):
        if permission not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="You don't have enough permissions to perform this operation")
        return current_user

    return wrapper


# for multiple permission
# def require_all_permissions(*required_permissions):
#     def checker(current_user=Depends(get_current_user)):
#         user_permissions = current_user.get("permissions", [])
#         if not all(p in user_permissions for p in required_permissions):
#             raise HTTPException(status_code=403, detail="Permission denied")
#         return current_user
#     return checker


def require_role(role: str):
    def wrapper(current_user=Depends(get_current_user)):
        if role not in current_user.get("roles", []):
            raise HTTPException(status_code=403, detail="Role required")
        return current_user

    return wrapper
