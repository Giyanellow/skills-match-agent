"""
NLP-based skill extraction and resume-job matching system.

This module provides intelligent skill extraction using spaCy NLP and comprehensive
matching capabilities for resume analysis against job descriptions.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import requests
import spacy
from loguru import logger


class NLPSkillExtractor:
    """
    Intelligent skill extractor using spaCy NLP for token-based skill recognition.

    This class builds and maintains a comprehensive skill taxonomy from multiple sources
    (GitHub Linguist, curated tech stacks) and uses NLP tokenization to extract skills
    from text while preserving their original formatting (e.g., "Node.js", "C++", ".NET").

    The extractor handles:
    - Multi-word skills (e.g., "Google Cloud", "GitHub Actions")
    - Skill variations (e.g., "nodejs" → "node.js", ".net" → "dotnet")
    - Format preservation (prefers mixed case like "TypeScript" over "typescript")
    - Special characters in skill names (C++, C#, .NET)

    Attributes:
        CACHE_FILE: Path to JSON cache file containing the skill taxonomy
        nlp: Loaded spaCy language model for tokenization
        skills: Set of lowercase skill names from the taxonomy
        skill_variations: Mapping of canonical skill names to their variations
    """

    CACHE_FILE = "combined_skills.json"

    def __init__(self, auto_update: bool = False):
        """
        Initialize the NLP skill extractor with spaCy model and skill taxonomy.

        Args:
            auto_update: If True, rebuild skill taxonomy from external sources.
                        If False, load from cache file if it exists.

        Raises:
            OSError: If spaCy model 'en_core_web_md' is not installed.
        """
        try:
            self.nlp = spacy.load("en_core_web_md")
            logger.info("✓ Loaded spaCy model")
        except OSError:
            logger.error(
                "spaCy model not found. Run: python -m spacy download en_core_web_md"
            )
            raise
        self.skills = self._load_or_build_skills(auto_update)
        self.skill_variations = self._build_variation_map()

    def _load_or_build_skills(self, auto_update: bool) -> Set[str]:
        """
        Load skill taxonomy from cache or build from multiple sources.

        If auto_update=False and cache exists, loads from cache for fast initialization.
        Otherwise, fetches skills from GitHub Linguist API and curated lists, then caches.

        Args:
            auto_update: Whether to rebuild the taxonomy from external sources

        Returns:
            Set of lowercase skill names

        Note:
            The cache file is written as sorted JSON for version control friendliness.
        """
        cache_path = Path(self.CACHE_FILE)

        if not auto_update and cache_path.exists():
            with open(cache_path, "r") as f:
                data = json.load(f)
                logger.info(f"✓ Loaded {len(data)} skills from cache")
                return set(s.lower() for s in data)

        logger.info("Building skill taxonomy from multiple sources...")
        skills = set()

        try:
            skills.update(self._fetch_github_languages())
            logger.info("  + GitHub languages")
        except Exception as e:
            logger.error(f"  ✗ GitHub failed: {e}")

        skills.update(self._get_curated_tech_skills())
        logger.info("  + Curated tech skills")
        skills.update(self._add_variations(skills))

        # Filter single-letter skills except 'R' (the programming language)
        skills = {s for s in skills if len(s) > 1 or s.lower() == "r"}

        with open(cache_path, "w") as f:
            json.dump(sorted(skills), f, indent=2)

        logger.info(f"✓ Built taxonomy with {len(skills)} skills")

        return skills

    def _build_variation_map(self) -> Dict[str, Set[str]]:
        """
        Build a mapping of canonical skill names to their common variations.

        This enables matching skills written in different formats:
        - "node.js" → {"nodejs", "node.js", "node"}
        - ".net" → {".net", "dotnet", "net"}
        - "some-tool" → {"some-tool", "sometool"}

        Returns:
            Dictionary mapping lowercase canonical forms to sets of variations

        Example:
            >>> variations = extractor._build_variation_map()
            >>> variations["node.js"]
            {"node.js", "nodejs", "node"}
        """
        variations = {}

        for skill in self.skills:
            canonical = skill.lower()
            variations[canonical] = {canonical}

            # Handle .js framework variations (node.js, vue.js, etc.)
            if ".js" in skill:
                base = skill.replace(".js", "")
                variations[canonical].add(base)
                variations[canonical].add(base + "js")

            # Handle dotted names (.net, asp.net)
            if "." in skill:
                variations[canonical].add(skill.replace(".", ""))
                if skill.startswith("."):
                    variations[canonical].add("dot" + skill[1:])

            # Handle hyphenated names
            if "-" in skill:
                variations[canonical].add(skill.replace("-", ""))

        return variations

    def _fetch_github_languages(self) -> Set[str]:
        """
        Fetch programming language names from GitHub Linguist repository.

        Parses the languages.yml file to extract language names for the skill taxonomy.

        Returns:
            Set of lowercase language names

        Raises:
            requests.HTTPError: If the GitHub API request fails

        Note:
            This method is called during taxonomy building if auto_update=True.
        """
        url = "https://raw.githubusercontent.com/github/linguist/master/lib/linguist/languages.yml"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        languages = set()

        for line in response.text.split("\n"):
            if line and not line.startswith(" ") and ":" in line:
                lang = line.split(":")[0].strip()
                if lang and not lang.startswith("#"):
                    languages.add(lang.lower())

        return languages

    def _get_curated_tech_skills(self) -> Set[str]:
        """
        Return curated list of important frameworks, tools, and methodologies.

        This hand-curated set ensures critical modern tech skills are included
        in the taxonomy, covering:
        - Web frameworks (React, Vue.js, Django, Flask, etc.)
        - Databases (PostgreSQL, MongoDB, Redis, etc.)
        - Cloud platforms (AWS, Azure, GCP)
        - DevOps tools (Docker, Kubernetes, Terraform, etc.)
        - Methodologies (Agile, Scrum, CI/CD, microservices)

        Returns:
            Set of lowercase skill names
        """
        return {
            # Frameworks
            "react",
            "vue.js",
            "angular",
            "svelte",
            "next.js",
            "django",
            "flask",
            "fastapi",
            "spring",
            "express",
            "nest.js",
            ".net",
            "asp.net",
            "laravel",
            "rails",
            # Databases
            "postgresql",
            "mysql",
            "mongodb",
            "redis",
            "elasticsearch",
            "cassandra",
            "dynamodb",
            "sql",
            "nosql",
            # Cloud
            "aws",
            "azure",
            "gcp",
            "google cloud",
            "heroku",
            "vercel",
            # DevOps
            "docker",
            "kubernetes",
            "k8s",
            "terraform",
            "ansible",
            "jenkins",
            "github actions",
            "gitlab ci",
            "ci/cd",
            # Methodologies
            "agile",
            "scrum",
            "devops",
            "tdd",
            "rest",
            "graphql",
            "microservices",
            "api",
        }

    def _add_variations(self, skills: Set[str]) -> Set[str]:
        """
        Generate common variations of existing skills.

        Adds bidirectional variations for JavaScript frameworks:
        - "vue.js" → "vue"
        - "react" → "react.js"

        Args:
            skills: Existing skill set

        Returns:
            Set of additional skill variations to add to the taxonomy
        """
        variations = set()

        for skill in skills:
            if skill.endswith(".js"):
                variations.add(skill[:-3])  # vue.js → vue
            elif skill in ["react", "vue", "next", "nest"]:
                variations.add(f"{skill}.js")  # react → react.js

        return variations

    def extract_best_format(self, text: str) -> List[str]:
        """
        Extract skills from text using NLP tokenization with format preservation.

        This is the main extraction method that:
        1. Tokenizes text using spaCy (better than regex for word boundaries)
        2. Matches tokens against the skill taxonomy
        3. Handles multi-word skills (e.g., "Google Cloud", "GitHub Actions")
        4. Recognizes skill variations (e.g., "nodejs" → "node.js")
        5. Preserves the best formatting (mixed case, special characters)

        The method uses two matching strategies:
        - Single-token matching for simple skills
        - Sliding window (1-3 tokens) for multi-word skills

        Args:
            text: Input text (resume, job description, etc.)

        Returns:
            Sorted list of extracted skills in their best format
            (e.g., ["C++", "Node.js", "PostgreSQL", "Python"])

        Example:
            >>> extractor.extract_best_format("I know Python, nodejs, and C++")
            ["C++", "Node.js", "Python"]
        """
        if not text:
            return []

        doc = self.nlp(text)
        found_skills = {}

        # Method 1: Single-token matching
        for token in doc:
            token_lower = token.text.lower()
            if token_lower in self.skills:
                if token_lower not in found_skills:
                    found_skills[token_lower] = []
                found_skills[token_lower].append(token.text)

        # Method 2: Multi-word skills using sliding window
        # Captures skills like "Node.js", "Google Cloud", "CI/CD"
        for i in range(len(doc)):
            for length in range(1, 4):
                if i + length > len(doc):
                    break

                span = doc[i : i + length]
                span_text = span.text.lower()

                if span_text in self.skills:
                    if span_text not in found_skills:
                        found_skills[span_text] = []
                    found_skills[span_text].append(span.text)

                # Check against skill variations
                for canonical, vars in self.skill_variations.items():
                    if span_text in vars:
                        if canonical not in found_skills:
                            found_skills[canonical] = []
                        found_skills[canonical].append(span.text)

        # Select the best format for each skill
        result = []
        for skill_lower, formats in found_skills.items():
            best_format = self._pick_best_format(formats)
            result.append(best_format)

        return sorted(result)

    def _pick_best_format(self, formats: List[str]) -> str:
        """
        Choose the best format when a skill appears multiple times in different cases.

        Selection criteria:
        1. If one format appears more frequently, prefer it
        2. Otherwise, rank by quality score (mixed case > special chars > uppercase > lowercase)

        Args:
            formats: List of format variations seen in text (e.g., ["python", "Python", "PYTHON"])

        Returns:
            The best format string

        Example:
            >>> extractor._pick_best_format(["python", "Python", "PYTHON"])
            "Python"  # Mixed case wins
        """
        if len(formats) == 1:
            return formats[0]

        # Count occurrences
        counts = Counter(formats)
        most_common = counts.most_common(1)[0]

        # If one format appears significantly more, use it
        if most_common[1] > 1:
            return most_common[0]

        # Otherwise, prefer higher quality formatting
        ranked = sorted(formats, key=self._format_quality_score, reverse=True)
        return ranked[0]

    def _format_quality_score(self, format_str: str) -> int:
        """
        Calculate quality score for a skill format string.

        Scoring hierarchy (higher is better):
        - Mixed case (e.g., "TypeScript", "PostgreSQL"): +10 points
        - Special characters (e.g., "C++", ".NET", "CI/CD"): +5 points
        - All uppercase (e.g., "AWS", "SQL"): +2 points
        - All lowercase (e.g., "python", "docker"): +1 point

        Args:
            format_str: A skill name in specific formatting

        Returns:
            Integer score (higher = better format quality)

        Example:
            >>> extractor._format_quality_score("Node.js")  # Mixed + special
            15
            >>> extractor._format_quality_score("nodejs")   # Lowercase
            1
        """
        score = 0

        # Mixed case is preferred (TypeScript, PostgreSQL)
        if any(c.isupper() for c in format_str) and any(
            c.islower() for c in format_str
        ):
            score += 10

        # Special characters indicate proper formatting (C++, .NET, CI/CD)
        if any(c in format_str for c in ["+", "#", ".", "-", "/"]):
            score += 5

        # All uppercase is acceptable (AWS, SQL, API)
        if format_str.isupper() and len(format_str) > 1:
            score += 2

        # All lowercase is least preferred
        if format_str.islower():
            score += 1

        return score


class ResumeJobMatcher:
    """
    Match resume skills against job description requirements.

    This class compares skills extracted from a resume and job description to:
    - Identify matching skills (candidate has the skill)
    - Identify missing skills (candidate lacks the skill)
    - Calculate a match score as a percentage
    - Generate human-readable explanations

    The matching is case-insensitive and uses the job description's formatting
    in the output for consistency.

    Attributes:
        skill_extractor: NLPSkillExtractor instance for extracting skills
    """

    def __init__(self, skill_extractor: NLPSkillExtractor):
        """
        Initialize the matcher with a skill extractor.

        Args:
            skill_extractor: Configured NLPSkillExtractor instance
        """
        self.skill_extractor = skill_extractor

    def match_skills(self, resume_text: str, job_desc: str) -> dict:
        """
        Compare resume skills against job description requirements.

        Extracts skills from both documents, identifies matches and gaps, and
        calculates a deterministic match score.

        Args:
            resume_text: Full resume text
            job_desc: Full job description text

        Returns:
            Dictionary containing:
                - matched_keywords: List of skills found in both (using job's format)
                - missing_keywords: List of required skills not in resume (job's format)
                - score: Percentage match score (0-100)
                - match_ratio: String like "5/8" (matched/total required)
                - explanation: Human-readable analysis with match verdict

        Example:
            >>> result = matcher.match_skills(resume_text, job_text)
            >>> result["score"]
            75.0
            >>> result["matched_keywords"]
            ["Python", "Docker", "PostgreSQL"]
            >>> result["missing_keywords"]
            ["Kubernetes"]
        """

        # Extract skills with format preservation
        job_skills = self.skill_extractor.extract_best_format(job_desc)
        resume_skills = self.skill_extractor.extract_best_format(resume_text)

        # Create lowercase maps for case-insensitive matching
        job_map = {s.lower(): s for s in job_skills}
        resume_map = {s.lower(): s for s in resume_skills}

        job_keys = set(job_map.keys())
        resume_keys = set(resume_map.keys())

        # Find matches and gaps
        matched_keys = job_keys & resume_keys
        missing_keys = job_keys - resume_keys

        if not job_keys:
            return self._empty_result("No skills found in job description")

        # Use job description's formatting in output
        matched_display = [job_map[k] for k in sorted(matched_keys)]
        missing_display = [job_map[k] for k in sorted(missing_keys)]

        # Calculate deterministic score
        score = (len(matched_keys) / len(job_keys)) * 100

        return {
            "matched_keywords": matched_display,
            "missing_keywords": missing_display,
            "score": round(score, 2),
            "match_ratio": f"{len(matched_keys)}/{len(job_keys)}",
            "explanation": self._explain(matched_display, missing_display, score),
        }

    def _explain(self, matched: List[str], missing: List[str], score: float) -> str:
        """
        Generate human-readable explanation of match results.

        Provides a verdict based on score thresholds and lists matched/missing skills.

        Verdict thresholds:
        - ≥80%: "Strong match"
        - 60-79%: "Good match"
        - 40-59%: "Partial match"
        - <40%: "Weak match"

        Args:
            matched: List of matched skill names
            missing: List of missing skill names
            score: Match percentage

        Returns:
            Multi-line explanation string
        """
        explanation = []

        # Provide verdict based on score
        if score >= 80:
            explanation.append(f"✓ Strong match ({score}%)")
        elif score >= 60:
            explanation.append(f"~ Good match ({score}%)")
        elif score >= 40:
            explanation.append(f"△ Partial match ({score}%)")
        else:
            explanation.append(f"✗ Weak match ({score}%)")

        if matched:
            explanation.append(f"\nMatched skills ({len(matched)}):")
            explanation.append("  " + ", ".join(matched))

        if missing:
            explanation.append(f"\nMissing skills ({len(missing)}):")
            explanation.append("  " + ", ".join(missing))

        return "\n".join(explanation)

    def _empty_result(self, message: str) -> dict:
        """
        Return empty result structure with custom message.

        Used for edge cases like empty job descriptions or no skills found.

        Args:
            message: Explanation message to include

        Returns:
            Dictionary with zero values and the provided message
        """
        return {
            "matched_keywords": [],
            "missing_keywords": [],
            "score": 0.0,
            "match_ratio": "0/0",
            "explanation": message,
        }


if __name__ == "__main__":
    # Example usage demonstrating the extractor and matcher
    extractor = NLPSkillExtractor()
    matcher = ResumeJobMatcher(extractor)

    job_desc = """
    Requirements:
    - C++ programming
    - Node.js backend
    - .NET framework
    - MySQL
    - Postgresql
    """

    resume = """
    SKILLS: C++, Nodejs, Python, SQL, SQLite
    """

    result = matcher.match_skills(resume, job_desc)
    from pprint import pprint

    pprint(result)
    print(result["matched_keywords"])  # Uses job's format: ['C++', 'Node.js']
    print(result["missing_keywords"])  # ['.NET', 'PostgreSQL']
