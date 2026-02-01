"""
FastAPI routes for resume-job analysis API endpoints.

This module defines the HTTP API for uploading and analyzing resumes against
job descriptions.
"""

from typing import Annotated

from fastapi import File, HTTPException, UploadFile
from fastapi.routing import APIRouter
from loguru import logger
from pydantic_ai import AgentRunResult

from src.core.agent_setup import ResumeAgent, ResumeAgentInput, ResumeAgentOutput

analysis_router = APIRouter()


def validate_txt_file(file: UploadFile = File(...)) -> UploadFile:
    """
    Validate that uploaded file is a .txt file with a valid filename.

    Args:
        file: Uploaded file from FastAPI

    Returns:
        The same file if validation passes

    Raises:
        HTTPException: 400 if file has no name or wrong extension
    """
    if not file.filename:
        raise HTTPException(400, "File must have a filename")
    if not file.filename.endswith(".txt"):
        raise HTTPException(400, f"Must be .txt file, got: {file.filename}")
    return file


@analysis_router.post("/")
async def analyze_resume(
    job_description_file: Annotated[UploadFile, File(...)],
    resume_file: Annotated[UploadFile, File(...)],
) -> ResumeAgentOutput:
    """
    Analyze resume against job description and return structured match results.

    This endpoint:
    1. Accepts two file uploads (job description and resume as .txt files)
    2. Extracts text content from both files
    3. Runs AI-powered skill extraction and matching analysis
    4. Returns structured results with match score and insights

    Args:
        job_description_file: Job posting/description as UTF-8 encoded .txt file
        resume_file: Candidate resume as UTF-8 encoded .txt file

    Returns:
        ResumeAgentOutput containing:
            - top_keywords: Most important skills from job description
            - matched_keywords: Skills found in both documents
            - missing_keywords: Skills candidate lacks
            - match_score: Percentage match (0-100)
            - confidence_notes: Technical notes about analysis quality
            - summary: Professional assessment of the match

    Raises:
        HTTPException 400: If files cannot be decoded as UTF-8
        HTTPException 500: If analysis fails for any reason

    Example:
        ```bash
        curl -X POST http://localhost:8000/analysis/ \\
          -F "job_description_file=@job.txt" \\
          -F "resume_file=@resume.txt"
        ```

        Response:
        ```json
        {
          "top_keywords": ["Python", "Docker", "AWS", "PostgreSQL"],
          "matched_keywords": ["Python", "Docker"],
          "missing_keywords": ["AWS", "PostgreSQL"],
          "match_score": 50.0,
          "confidence_notes": "Resume has clear skills section. Good match quality.",
          "summary": "50% match. Strong in Python and Docker. Missing cloud (AWS) and database (PostgreSQL) experience."
        }
        ```
    """
    try:
        logger.info(
            f"Received files - Job: {job_description_file.filename}, Resume: {resume_file.filename}"
        )

        job_str = (await job_description_file.read()).decode("utf-8")
        resume_str = (await resume_file.read()).decode("utf-8")

        agent = ResumeAgent()
        input_data = ResumeAgentInput(job_description=job_str, resume_text=resume_str)
        result = await agent.analyze_invoke_async(input=input_data)

        return result.output

    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Could not decode file as UTF-8: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
