"""
PydanticAI tool for skill extraction and matching.

This module provides the tool function that the AI agent uses to extract and compare
skills from job descriptions and resumes.
"""

from typing import Dict

from loguru import logger
from pydantic_ai import RunContext


def analyze_skills(ctx: RunContext, job_description: str, resume_text: str) -> Dict:
    """
    Extract and match skills between job description and resume.

    This is a PydanticAI tool that performs single-step skill analysis:
    1. Extracts skills from both job description and resume using NLP
    2. Compares the extracted skills to find matches and gaps
    3. Calculates a deterministic match score

    The tool accesses the NLP extractor and matcher through the RunContext's
    dependency injection system (ctx.deps).

    Args:
        ctx: PydanticAI RunContext providing access to dependencies (extractor, matcher)
        job_description: Full text of the job posting
        resume_text: Full text of the candidate's resume

    Returns:
        Dictionary containing:
            - job_skills: List of skills extracted from job description
            - resume_skills: List of skills extracted from resume
            - matched_keywords: Skills found in both documents
            - missing_keywords: Skills in job but not in resume
            - score: Match percentage (0-100)
            - match_ratio: String like "5/8" showing matched/total

    Raises:
        Exception: If skill extraction or matching fails

    Example:
        When called by the agent:
        ```
        result = analyze_skills(ctx, job_desc, resume)
        # result = {
        #     "job_skills": ["Python", "Docker", "AWS"],
        #     "resume_skills": ["Python", "Docker", "JavaScript"],
        #     "matched_keywords": ["Python", "Docker"],
        #     "missing_keywords": ["AWS"],
        #     "score": 66.67,
        #     "match_ratio": "2/3"
        # }
        ```

    Note:
        This tool is designed to be called by PydanticAI agents and expects
        ctx.deps to have 'extractor' and 'matcher' attributes.
    """
    logger.info("Skills analysis tool is being called")

    try:
        # Extract skills
        job_skills = ctx.deps.extractor.extract_best_format(job_description)
        resume_skills = ctx.deps.extractor.extract_best_format(resume_text)

        logger.info(
            f"Extracted job skills: {job_skills} \n Extracted resume skills: {resume_skills}"
        )

        # Match skills
        match_result = ctx.deps.matcher.match_skills(resume_text, job_description)

        # Return combined result
        return {
            "job_skills": job_skills,
            "resume_skills": resume_skills,
            "matched_keywords": match_result["matched_keywords"],
            "missing_keywords": match_result["missing_keywords"],
            "score": match_result["score"],
            "match_ratio": match_result["match_ratio"],
        }
    except Exception as e:
        logger.error(f"Skills analysis failed: {e}")
        raise
