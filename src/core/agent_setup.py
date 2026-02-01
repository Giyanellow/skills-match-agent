"""
Resume-job matching agent configuration and execution logic.

This module defines the PydanticAI agent that orchestrates skill extraction and matching
between resumes and job descriptions, providing structured output for API consumers.
"""

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import AgentRunResult

from src.config.paths import PROMPTS_DIR
from src.core.agent_factory import AgentFactory
from src.core.extractor import NLPSkillExtractor, ResumeJobMatcher
from src.core.tools import analyze_skills

with open(PROMPTS_DIR / "agent_prompts.yaml") as f:
    prompts = yaml.safe_load(f)


class ResumeAgentInput(BaseModel):
    """
    Input data for resume-job matching analysis.

    Attributes:
        job_description: Full text of the job posting/description
        resume_text: Full text of the candidate's resume
    """

    job_description: str
    resume_text: str


class ResumeAgentOutput(BaseModel):
    """
    Structured output for resume-job matching analysis.

    This schema ensures consistent, machine-readable results from the AI agent.
    All fields are deterministic and based on NLP skill extraction + matching.

    Attributes:
        top_keywords: Most important skills/requirements from job description,
                     ranked by criticality and frequency (limited to 10-15)
        matched_keywords: Skills found in both resume and job description
        missing_keywords: Required skills from job not found in resume
        match_score: Percentage match (0-100), calculated as (matched/total) × 100
        confidence_notes: Technical notes about extraction quality and limitations
        summary: Professional 2-3 sentence assessment for stakeholders
    """

    # Core matching data
    top_keywords: List[str] = Field(
        description="Top extracted keywords from job description, ranked by importance"
    )
    matched_keywords: List[str] = Field(
        description="Keywords found in both resume and job description"
    )
    missing_keywords: List[str] = Field(
        description="Required keywords from job description not found in resume"
    )

    # Scoring
    match_score: float = Field(
        description="Deterministic match score: (matched/total) * 100", ge=0, le=100
    )

    # Analysis
    confidence_notes: str = Field(
        description="Technical notes about extraction quality and matching confidence"
    )
    summary: str = Field(
        description="Professional, concise summary of the match results"
    )


class ResumeAgentDeps:
    """
    Dependency injection container for the resume matching agent.

    Provides the agent with access to NLP extraction and matching capabilities
    without hardcoding these dependencies in the agent itself.

    Attributes:
        extractor: NLPSkillExtractor instance for skill extraction
        matcher: ResumeJobMatcher instance for skill comparison
    """

    def __init__(self) -> None:
        """Initialize extractor and matcher instances."""
        self.extractor = NLPSkillExtractor()
        self.matcher = ResumeJobMatcher(skill_extractor=self.extractor)


class ResumeAgent:
    """
    Main agent class for resume-job matching analysis.

    This agent uses Claude Haiku 4.5 with PydanticAI to:
    1. Extract skills from job descriptions and resumes using NLP
    2. Match and compare the extracted skills
    3. Generate structured, actionable insights

    The agent has access to the `analyze_skills` tool which encapsulates
    the NLP extraction and matching logic.

    Attributes:
        factory: AgentFactory instance for creating PydanticAI agents
        instruction_prompt: System prompt loaded from YAML configuration
        agent: Configured PydanticAI agent instance
    """

    def __init__(self):
        """
        Initialize the resume matching agent.

        Sets up the agent with Claude Haiku 4.5, skill analysis tools,
        and structured output schema.
        """
        self.factory = AgentFactory(
            model_name="claude-haiku-4-5-20251001", tools=[analyze_skills]
        )
        self.instruction_prompt = prompts["resume_agent_prompt"]
        self.agent = self.factory.create_agent(
            instruction_prompt=self.instruction_prompt,
            output_type=ResumeAgentOutput,
            deps_type=ResumeAgentDeps,
            retries=3,
        )

    def analyze_invoke(
        self, input: ResumeAgentInput
    ) -> AgentRunResult[ResumeAgentOutput]:
        """
        Synchronous analysis of resume against job description.

        Formats the input into a prompt, runs the agent synchronously, and returns
        structured results.

        Args:
            input: ResumeAgentInput containing job_description and resume_text

        Returns:
            AgentRunResult containing ResumeAgentOutput with match analysis

        Note:
            This is the synchronous version. Use analyze_invoke_async for FastAPI routes.
        """
        formatted_input = f"""
        Job Description:
            {input.job_description}

        Resume:
            {input.resume_text}
        """
        result = self.agent.run_sync(
            user_prompt=formatted_input, deps=ResumeAgentDeps()
        )
        return result

    async def analyze_invoke_async(
        self, input: ResumeAgentInput
    ) -> AgentRunResult[ResumeAgentOutput]:
        """
        Asynchronous analysis of resume against job description.

        Formats the input into a prompt, runs the agent asynchronously, and returns
        structured results. This is the preferred method for FastAPI endpoints.

        Args:
            input: ResumeAgentInput containing job_description and resume_text

        Returns:
            AgentRunResult containing ResumeAgentOutput with match analysis

        Note:
            Use this method in async contexts (FastAPI routes, async workflows).
        """
        formatted_input = f"""
        Job Description:
            {input.job_description}
        Resume:
            {input.resume_text}
        """
        result = await self.agent.run(
            user_prompt=formatted_input, deps=ResumeAgentDeps()
        )
        return result


if __name__ == "__main__":
    agent = ResumeAgent()
    JOB_DESCRIPTION = """
    Senior Full Stack Engineer
    We are seeking a Senior Full Stack Engineer to join our platform team building scalable web applications.
    Required Skills:
    - 5+ years of professional software development experience
    - Strong proficiency in Python and modern web frameworks (Django or Flask)
    - Experience with React.js and TypeScript for frontend development
    - Solid understanding of RESTful API design and microservices architecture
    - Proficiency with SQL databases (PostgreSQL preferred) and ORM frameworks
    - Experience with containerization using Docker and orchestration with Kubernetes
    - Strong knowledge of Git version control and CI/CD pipelines
    - Understanding of cloud platforms (AWS, GCP, or Azure)
    Nice to Have:
    - Experience with Redis or other caching solutions
    - Knowledge of message queues (RabbitMQ, Kafka)
    - Familiarity with monitoring tools (Prometheus, Grafana, DataDog)
    - Experience with GraphQL APIs
    - Contributions to open-source projects
    Responsibilities:
    - Design and implement scalable backend services and APIs
    - Build responsive user interfaces with modern JavaScript frameworks
    - Collaborate with product and design teams on feature development
    - Participate in code reviews and maintain high code quality standards
    - Optimize application performance and scalability
    """
    RESUME_TEXT = """
    John Smith
    Full Stack Software Engineer
    john.smith@email.com | LinkedIn: /in/johnsmith | GitHub: github.com/jsmith
    SUMMARY
    Results-driven Full Stack Engineer with 6 years of experience building web applications and distributed systems. 
    Passionate about clean code, scalable architecture, and modern development practices.
    TECHNICAL SKILLS
    Languages: Python, JavaScript, TypeScript, SQL, Bash
    Backend: Django, Flask, FastAPI, Node.js, RESTful APIs
    Frontend: React, Redux, HTML5, CSS3, Material-UI
    Databases: PostgreSQL, MySQL, MongoDB, Redis
    DevOps: Docker, Kubernetes, Jenkins, GitHub Actions, AWS (EC2, S3, RDS, Lambda)
    Tools: Git, Jira, Postman, VS Code
    PROFESSIONAL EXPERIENCE
    Senior Software Engineer | TechCorp Inc. | Jan 2021 - Present
    - Architected and deployed microservices using Python/Django and Docker, serving 2M+ daily active users
    - Built responsive admin dashboards with React and TypeScript, improving operational efficiency by 40%
    - Implemented CI/CD pipelines with GitHub Actions, reducing deployment time from 2 hours to 15 minutes
    - Designed RESTful APIs consumed by web and mobile clients with 99.9% uptime
    - Optimized PostgreSQL queries and implemented Redis caching, reducing API response time by 60%
    - Led code reviews and mentored 3 junior engineers on best practices
    Software Engineer | StartupXYZ | Jun 2018 - Dec 2020
    - Developed full-stack features using Flask backend and React frontend
    - Integrated third-party payment APIs (Stripe, PayPal) processing $5M+ annually
    - Migrated monolithic application to microservices architecture on AWS
    - Implemented automated testing achieving 85% code coverage
    - Collaborated with product team in Agile/Scrum environment
    EDUCATION
    B.S. Computer Science | State University | 2018
    Relevant Coursework: Data Structures, Algorithms, Database Systems, Web Development
    PROJECTS
    - Open Source Contributor: Active contributor to Django REST framework (200+ stars on GitHub)
    - Personal: Built a real-time chat application using WebSockets, Redis pub/sub, and React
    """
    input = ResumeAgentInput(job_description=JOB_DESCRIPTION, resume_text=RESUME_TEXT)
    print(agent.analyze_invoke(input=input))
