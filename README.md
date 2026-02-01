# Skills Match Agent

AI-powered resume-job matching system that uses NLP and Claude Haiku 4.5 to analyze skill alignment between resumes and job descriptions.

## Overview

This application provides intelligent resume analysis by:

- **Extracting technical skills** from job descriptions and resumes using spaCy NLP
- **Matching skills** to identify gaps and overlaps
- **Calculating deterministic match scores** based on skill coverage
- **Generating actionable insights** through AI-powered analysis

### Key Features

- ✅ **NLP-Based Extraction**: Uses spaCy for accurate skill recognition (not regex)
- ✅ **Format Preservation**: Maintains proper skill formatting (e.g., "Node.js", "C++", ".NET")
- ✅ **Variation Handling**: Recognizes skill variations (e.g., "nodejs" → "node.js")
- ✅ **Multi-word Skills**: Detects compound skills like "Google Cloud", "GitHub Actions"
- ✅ **Deterministic Scoring**: Objective match percentage based on skill overlap
- ✅ **RESTful API**: FastAPI backend with automatic OpenAPI documentation

## Table of Contents

- [Installation](#installation)
- [How to Run](#how-to-run)
- [API Usage](#api-usage)
- [Example Inputs and Outputs](#example-inputs-and-outputs)
- [Scoring Logic](#scoring-logic)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)
- [Testing](#testing)

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager (recommended) or pip
- Anthropic API key for Claude

### Setup Steps

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd skills-match-agent
   ```

2. **Install uv (if not already installed)**

   Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/):

   ```bash
   # macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Create virtual environment and install dependencies**

   Using `uv` (recommended):

   ```bash
   uv sync
   ```

   This will:
   - Create a `.venv` virtual environment
   - Install all dependencies from `pyproject.toml`
   - Set up the project in editable mode

   Using `pip` (alternative):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

4. **Download spaCy model**

   ```bash
   # With uv
   uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

   # With pip
   python -m spacy download en_core_web_md
   ```

5. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   # Optional: OPENAI_API_KEY for GPT models
   ```

## How to Run

### Running the API Server

Start the FastAPI application:

```bash
# Using uv
uv run python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- **Base URL**: <http://localhost:8000>
- **API Docs**: <http://localhost:8000/docs> (Swagger UI)
- **Alternative Docs**: <http://localhost:8000/redoc>

### Running the Agent Standalone

You can also run the agent directly without the API:

```python
from src.core.agent_setup import ResumeAgent, ResumeAgentInput

agent = ResumeAgent()

job_description = """
Senior Backend Developer
Required Skills:
- Python (5+ years)
- Django or Flask
- PostgreSQL
- Docker & Kubernetes
- AWS
"""

resume_text = """
John Doe - Software Engineer
Skills: Python, Django, PostgreSQL, Docker, Git, React
Experience: 6 years backend development
"""

input_data = ResumeAgentInput(
    job_description=job_description,
    resume_text=resume_text
)

result = agent.analyze_invoke(input=input_data)
print(result.output)
```

## API Usage

### Endpoint: POST `/analysis/`

Analyze a resume against a job description.

**Request:**

```bash
curl -X POST "http://localhost:8000/analysis/" \
  -F "job_description_file=@job.txt" \
  -F "resume_file=@resume.txt"
```

**Python Example:**

```python
import requests

with open("job.txt", "rb") as job_file, open("resume.txt", "rb") as resume_file:
    files = {
        "job_description_file": job_file,
        "resume_file": resume_file
    }
    response = requests.post("http://localhost:8000/analysis/", files=files)
    result = response.json()
    print(f"Match Score: {result['match_score']}%")
    print(f"Summary: {result['summary']}")
```

## Example Inputs and Outputs

### Example 1: Strong Match

**Job Description (job.txt):**

```
Senior Full Stack Engineer

Required Skills:
- Python programming (5+ years)
- React.js and TypeScript
- PostgreSQL database
- Docker containerization
- AWS cloud services
- RESTful API design
```

**Resume (resume.txt):**

```
Jane Smith - Full Stack Developer

Technical Skills:
Languages: Python, TypeScript, JavaScript, SQL
Frontend: React, Redux, HTML5, CSS3
Backend: Django, Flask, FastAPI
Databases: PostgreSQL, MongoDB, Redis
DevOps: Docker, Kubernetes, AWS (EC2, S3, Lambda)
Tools: Git, GitHub Actions, Postman

Experience:
Senior Software Engineer | TechCorp (2019-Present)
- Built scalable APIs using Python/Django serving 1M+ users
- Developed React/TypeScript frontends with responsive design
- Deployed microservices on AWS using Docker containers
```

**API Response:**

```json
{
  "top_keywords": [
    "Python",
    "React",
    "TypeScript",
    "PostgreSQL",
    "Docker",
    "AWS",
    "REST"
  ],
  "matched_keywords": [
    "Python",
    "React",
    "TypeScript",
    "PostgreSQL",
    "Docker",
    "AWS"
  ],
  "missing_keywords": [],
  "match_score": 100.0,
  "confidence_notes": "Resume has comprehensive skills section with clear technical expertise. All required skills are present with additional complementary skills (Kubernetes, MongoDB, Redis).",
  "summary": "Perfect 100% match. Candidate possesses all required skills including Python (5+ years evident), React/TypeScript frontend, PostgreSQL database, Docker containerization, and AWS cloud. Strong full-stack profile with microservices architecture experience."
}
```

### Example 2: Partial Match

**Job Description (job.txt):**

```
DevOps Engineer Position

Requirements:
- Kubernetes orchestration
- Terraform infrastructure as code
- Jenkins CI/CD
- Python scripting
- AWS cloud platform
- Monitoring (Prometheus, Grafana)
```

**Resume (resume.txt):**

```
Alex Johnson - Systems Engineer

Skills:
- Python automation scripts
- Docker containers
- AWS (EC2, S3, CloudFormation)
- Git version control
- Linux administration
- Bash scripting
```

**API Response:**

```json
{
  "top_keywords": [
    "Kubernetes",
    "Terraform",
    "Jenkins",
    "Python",
    "AWS",
    "Prometheus",
    "Grafana"
  ],
  "matched_keywords": ["Python", "AWS"],
  "missing_keywords": [
    "Kubernetes",
    "Terraform",
    "Jenkins",
    "Prometheus",
    "Grafana"
  ],
  "match_score": 28.57,
  "confidence_notes": "Resume lacks structured DevOps tools section. Candidate has foundational skills (Docker, AWS) but missing critical DevOps-specific tools. CloudFormation experience may partially substitute for Terraform.",
  "summary": "28.57% match - weak alignment. Candidate has Python scripting and AWS basics but lacks core DevOps tooling: Kubernetes orchestration, Terraform IaC, Jenkins CI/CD, and monitoring stack (Prometheus/Grafana). Consider as junior DevOps role or with training plan."
}
```

### Example 3: No Match

**Job Description:**

```
Data Scientist Role
Required: Python, R, TensorFlow, PyTorch, Pandas, Scikit-learn, SQL
```

**Resume:**

```
Marketing Manager
Skills: Google Analytics, SEO, Content Strategy, Social Media, Email Marketing
```

**API Response:**

```json
{
  "top_keywords": [
    "Python",
    "R",
    "TensorFlow",
    "PyTorch",
    "Pandas",
    "Scikit-learn",
    "SQL"
  ],
  "matched_keywords": [],
  "missing_keywords": [
    "Python",
    "R",
    "TensorFlow",
    "PyTorch",
    "Pandas",
    "Scikit-learn",
    "SQL"
  ],
  "match_score": 0.0,
  "confidence_notes": "No technical skills overlap detected. Resume focused on marketing competencies while job requires data science technical stack.",
  "summary": "0% match. No skill overlap between resume and job requirements. Candidate's marketing background does not align with data science technical requirements."
}
```

## Scoring Logic

### How the Match Score is Calculated

The match score is **deterministic and objective**, based purely on skill overlap:

```
Match Score = (Number of Matched Skills / Total Required Skills) × 100
```

### Step-by-Step Process

1. **Skill Extraction (NLP-based)**

   ```
   Job Description → spaCy tokenization → ["Python", "Docker", "AWS", "PostgreSQL"]
   Resume          → spaCy tokenization → ["Python", "Docker", "JavaScript", "Git"]
   ```

2. **Case-Insensitive Matching**

   ```
   Job Skills (lowercase):    {"python", "docker", "aws", "postgresql"}
   Resume Skills (lowercase): {"python", "docker", "javascript", "git"}

   Matched: {"python", "docker"}
   Missing: {"aws", "postgresql"}
   ```

3. **Score Calculation**

   ```
   Matched: 2 skills
   Required: 4 skills
   Score = (2 / 4) × 100 = 50.0%
   ```

4. **Format Preservation**
   - Output uses the **job description's formatting** for consistency
   - Example: Job has "Node.js", resume has "nodejs" → Output shows "Node.js"

### Match Quality Thresholds

The system classifies matches into categories:

| Score Range | Classification  | Interpretation                           |
| ----------- | --------------- | ---------------------------------------- |
| 80-100%     | ✓ Strong match  | Candidate meets most/all requirements    |
| 60-79%      | ~ Good match    | Candidate meets majority of requirements |
| 40-59%      | △ Partial match | Candidate has some relevant skills       |
| 0-39%       | ✗ Weak match    | Significant skill gaps exist             |

### What Makes This Scoring Accurate

1. **NLP Tokenization**: Uses spaCy's linguistic models instead of regex
   - Correctly handles punctuation: "C++" is one token, not "C" + "+" + "+"
   - Recognizes compound terms: "Google Cloud" as single skill
   - Respects word boundaries: "JavaScript" doesn't match "Java"

2. **Skill Taxonomy**: Built from 1000+ verified skills
   - GitHub Linguist programming languages
   - Curated frameworks, tools, methodologies
   - Common variations (nodejs ↔ node.js, .net ↔ dotnet)

3. **Variation Handling**: Normalizes equivalent terms

   ```
   "Node.js" = "nodejs" = "node"
   ".NET" = "dotnet"
   "Kubernetes" = "k8s"
   ```

4. **No Subjective Weighting**: All skills weighted equally
   - Prevents bias toward specific technologies
   - Provides clear, explainable results
   - Allows AI agent to add context in summary

## Known Limitations

### 1. Equal Skill Weighting

**Issue**: All skills treated with equal importance.

**Example**:

```
Job: "Python, Excel, Email etiquette"
Resume: "Python, Excel, Git"
Score: 66.67% (2/3)
```

"Python" and "Email etiquette" both count as 1 skill, despite different significance.

**Workaround**: The AI agent's `summary` field provides context about critical vs. nice-to-have skills.

### 2. No Years of Experience Tracking

**Issue**: Cannot distinguish between 1 year and 10 years of experience.

**Example**:

```
Job: "Python (5+ years)"
Resume: "Python (6 months)"
Result: Both extracted as "Python" → counted as match
```

**Future Enhancement**: Could extract experience duration with enhanced NLP patterns.

### 3. Soft Skills Not Detected

**Issue**: Only technical skills are extracted. Soft skills like "leadership", "communication", "problem-solving" are ignored.

**Design Decision**: Focused on objective, verifiable technical skills rather than subjective soft skills.

### 4. Context-Free Matching

**Issue**: No understanding of skill context or proficiency level.

**Example**:

```
Job: "Expert in machine learning model deployment"
Resume: "Completed online ML course"
Result: Both have "machine learning" → match
```

**Mitigation**: The AI agent's `confidence_notes` can flag ambiguities.

### 5. Acronym Ambiguity

**Issue**: Acronyms with multiple meanings may cause false matches.

**Example**:

- "ML" could mean Machine Learning or Markup Language
- "AI" could mean Artificial Intelligence or Adobe Illustrator

**Partial Solution**: Variation mapping handles known cases, but context-dependent acronyms remain challenging.

### 6. Taxonomy Coverage

**Issue**: Newer or niche technologies may not be in the skill database.

**Example**: Very new frameworks released after the taxonomy was built might not be recognized.

**Solution**: The system supports taxonomy updates via `NLPSkillExtractor(auto_update=True)` to fetch latest GitHub Linguist data.

### 7. Multi-Language Resumes

**Issue**: Only English-language resumes are supported.

**Limitation**: spaCy model is English-only (`en_core_web_md`).

### 8. Format Dependency

**Issue**: Currently only accepts plain text (.txt) files.

**Enhancement Needed**: Future versions could support PDF, DOCX parsing.

## Project Structure

```
skills-match-agent/
├── src/
│   ├── core/
│   │   ├── agent_factory.py      # PydanticAI agent factory with provider detection
│   │   ├── agent_setup.py        # Resume matching agent configuration
│   │   ├── extractor.py          # NLP skill extraction and matching logic
│   │   ├── tools.py              # AI agent tool functions
│   │   └── prompts/
│   │       └── agent_prompts.yaml # System prompts for AI agent
│   ├── config/
│   │   ├── config.py             # Application settings and API keys
│   │   └── paths.py              # Project path constants
│   └── routes/
│       └── analyze.py            # FastAPI route handlers
├── tests/
│   ├── conftest.py               # Pytest configuration and fixtures
│   └── test_extractor.py         # Comprehensive extractor tests
├── main.py                       # FastAPI application entry point
├── pyproject.toml                # Project dependencies and metadata
├── combined_skills.json          # Cached skill taxonomy (auto-generated)
└── README.md                     # This file
```

## Testing

### Run All Tests

```bash
# Using uv
uv run pytest

# Using pytest directly
pytest

# With coverage report
pytest --cov=src --cov-report=html
```

### Test Coverage

The project includes comprehensive tests for:

- ✅ Skill extraction (simple, multi-word, special characters)
- ✅ Format preservation and quality scoring
- ✅ Skill variation recognition
- ✅ Match calculation and scoring
- ✅ Edge cases (empty inputs, no matches, perfect matches)
- ✅ Real-world scenarios

### Running Specific Tests

```bash
# Test only the extractor
pytest tests/test_extractor.py

# Test a specific class
pytest tests/test_extractor.py::TestNLPSkillExtractor

# Test a specific function
pytest tests/test_extractor.py::TestNLPSkillExtractor::test_extract_simple_skills

# Verbose output
pytest -v
```

## Architecture

### How It Works

```
┌─────────────┐
│  User Input │
│ (Job + Resume)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  FastAPI Route  │  ← Handles file uploads, validation
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  ResumeAgent     │  ← PydanticAI agent with Claude Haiku 4.5
│  (AI Orchestrator)│
└────────┬─────────┘
         │
         ▼
┌────────────────────┐
│  analyze_skills()  │  ← Tool function called by agent
└─────────┬──────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌──────────┐  ┌────────┐
│ Extractor│  │Matcher │  ← NLP extraction + comparison
└──────────┘  └────────┘
    │              │
    └──────┬───────┘
           ▼
   ┌───────────────┐
   │ Structured    │  ← ResumeAgentOutput (Pydantic model)
   │ JSON Response │
   └───────────────┘
```

### Key Components

1. **NLPSkillExtractor** (`src/core/extractor.py`)
   - Loads spaCy language model
   - Builds skill taxonomy from GitHub + curated lists
   - Extracts skills using tokenization (not regex)
   - Handles variations and format preservation

2. **ResumeJobMatcher** (`src/core/extractor.py`)
   - Compares extracted skill sets
   - Calculates match score
   - Generates human-readable explanations

3. **ResumeAgent** (`src/core/agent_setup.py`)
   - Orchestrates extraction + matching via AI agent
   - Uses Claude Haiku 4.5 for intelligent analysis
   - Produces structured output (Pydantic validation)

4. **AgentFactory** (`src/core/agent_factory.py`)
   - Creates PydanticAI agents with configurable LLM providers
   - Supports Anthropic (Claude) and OpenAI (GPT) models
   - Manages tool mounting and dependency injection

## Configuration

### Environment Variables

| Variable            | Required | Description                                         |
| ------------------- | -------- | --------------------------------------------------- |
| `ANTHROPIC_API_KEY` | Yes      | API key for Claude models                           |
| `OPENAI_API_KEY`    | No       | API key for GPT models (if using)                   |
| `APP_NAME`          | No       | Application name (default: "Resume Analyzer Agent") |
| `ENVIRONMENT`       | No       | Environment: development/staging/production         |

### Agent Configuration

The agent's behavior is controlled by `src/core/prompts/agent_prompts.yaml`:

```yaml
resume_agent_prompt: |
  You are a resume-job matching analysis assistant...

  ## Your Workflow
  1. Extract Skills
  2. Match Skills
  3. Synthesize Results

  ## Output Requirements
  - top_keywords: Ranked list (10-15 max)
  - matched_keywords: Skills in both documents
  - missing_keywords: Skills only in job description
  - match_score: Use EXACT score from tool (don't recalculate)
  - confidence_notes: Technical observations
  - summary: Professional 2-3 sentence assessment
```

## Contributing

### Adding New Skills

To update the skill taxonomy:

```python
from src.core.extractor import NLPSkillExtractor

# Rebuild taxonomy from latest sources
extractor = NLPSkillExtractor(auto_update=True)
```

This will:

1. Fetch latest languages from GitHub Linguist
2. Include curated tech skills
3. Generate variations
4. Cache to `combined_skills.json`

### Customizing Match Logic

Edit `src/core/extractor.py` → `ResumeJobMatcher.match_skills()` to:

- Change scoring formula
- Add skill weighting
- Include experience duration
- Filter by skill categories

## Support

For issues, questions, or contributions:

- Open an issue in the repository
- Review the inline documentation in source files
- Check the test suite for usage examples

---

**Built with**: Python 3.12, FastAPI, spaCy, PydanticAI, Claude Haiku 4.5
