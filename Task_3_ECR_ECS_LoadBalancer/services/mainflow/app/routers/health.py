from fastapi import APIRouter

router = APIRouter(prefix="/mainflow", tags=["health"])


@router.get("/health", status_code=200)
def health_check():
    """Health check endpoint for Mainflow service."""
    return {"status": "Mainflow-healthy"}
