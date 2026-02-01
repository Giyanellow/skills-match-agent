"""
FastAPI application entry point for the Resume Analyzer API.

This module configures and runs the FastAPI application that provides resume-job
matching analysis through RESTful API endpoints.
"""

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes.analyze import analysis_router

app = FastAPI(
    title="Resume Analyzer API",
    description="AI-powered resume-job matching and skill analysis",
    version="0.1.0",
)
router = APIRouter()

# Configure CORS to allow cross-origin requests
# Note: In production, restrict allow_origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=analysis_router, prefix="/analysis", tags=["analysis"])


@app.post("/health-check")
def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        Dictionary with status indicator
    """
    return {"status": "healthy"}


@app.get("/")
def root():
    """
    Root endpoint providing API information.

    Returns:
        Dictionary with welcome message
    """
    return {"message": "Welcome to Resume Agent API"}


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application with uvicorn
    # Access API docs at: http://localhost:8000/docs
    uvicorn.run(app, host="0.0.0.0", port=8000)
