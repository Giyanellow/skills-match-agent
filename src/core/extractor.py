import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import requests
import spacy
from loguru import logger


class NLPSkillExtractor:
    """Use spaCy NLP for intelligent skill extraction"""

    CACHE_FILE = "combined_skills.json"

    def __init__(self, auto_update: bool = False):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_md")
            logger.info("✓ Loaded spaCy model")
        except OSError:
            logger.error(
                "spaCy model not found. Run: python -m spacy download en_core_web_md"
            )
            raise
        self.skills = self._load_or_build_skills(auto_update)

        # Build a mapping: lowercase → variations seen
        self.skill_variations = self._build_variation_map()

    def _load_or_build_skills(self, auto_update: bool) -> Set[str]:
        """Load from cache or build from multiple sources"""
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

        # Filter out problematic single-letter skills
        skills = {s for s in skills if len(s) > 1 or s.lower() == "r"}

        with open(cache_path, "w") as f:
            json.dump(sorted(skills), f, indent=2)

        logger.info(f"✓ Built taxonomy with {len(skills)} skills")

        return skills

    def _build_variation_map(self) -> Dict[str, Set[str]]:
        """
        Map canonical forms to their variations
        e.g., 'node.js' → {'nodejs', 'node.js', 'node'}
        """
        variations = {}

        for skill in self.skills:
            canonical = skill.lower()
            variations[canonical] = {canonical}

            # Add variations
            if ".js" in skill:
                # node.js → nodejs, node
                base = skill.replace(".js", "")
                variations[canonical].add(base)
                variations[canonical].add(base + "js")

            if "." in skill:
                # .net → dotnet, net
                variations[canonical].add(skill.replace(".", ""))
                if skill.startswith("."):
                    variations[canonical].add("dot" + skill[1:])

            if "-" in skill:
                # some-tool → sometool
                variations[canonical].add(skill.replace("-", ""))

        return variations

    def _fetch_github_languages(self) -> Set[str]:
        """Fetch from GitHub linguist"""
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
        """Curated list of frameworks, tools, methodologies"""
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
        """Add common variations"""
        variations = set()

        for skill in skills:
            if skill.endswith(".js"):
                variations.add(skill[:-3])  # vue.js → vue
            elif skill in ["react", "vue", "next", "nest"]:
                variations.add(f"{skill}.js")  # react → react.js

        return variations

    def extract_best_format(self, text: str) -> List[str]:
        """
        Extract skills using NLP tokens instead of regex
        """
        if not text:
            return []

        # Process text with spaCy
        doc = self.nlp(text)
        found_skills = {}  # skill_lowercase → original_format
        # Method 1: Token-based matching (better than regex)

        for token in doc:
            token_lower = token.text.lower()
            # Check if token matches any skill
            if token_lower in self.skills:
                if token_lower not in found_skills:
                    found_skills[token_lower] = []

                found_skills[token_lower].append(token.text)

        # Method 2: Multi-word skills (using noun chunks and sliding windows)
        # Check for skills like "Node.js", "Google Cloud", "CI/CD"
        for i in range(len(doc)):
            for length in range(1, 4):  # Check 1-3 token spans
                if i + length > len(doc):
                    break

                span = doc[i : i + length]
                span_text = span.text.lower()

                # Direct match
                if span_text in self.skills:
                    if span_text not in found_skills:
                        found_skills[span_text] = []
                    found_skills[span_text].append(span.text)

                # Check variations
                for canonical, vars in self.skill_variations.items():
                    if span_text in vars:
                        if canonical not in found_skills:
                            found_skills[canonical] = []
                        found_skills[canonical].append(span.text)
        # Pick best format for each skill
        result = []

        for skill_lower, formats in found_skills.items():
            best_format = self._pick_best_format(formats)
            result.append(best_format)

        return sorted(result)

    def _pick_best_format(self, formats: List[str]) -> str:
        """Choose the best format when multiple versions exist"""
        if len(formats) == 1:
            return formats[0]
        # Count occurrences
        counts = Counter(formats)
        most_common = counts.most_common(1)[0]
        # If one format appears more often, use it
        if most_common[1] > 1:
            return most_common[0]
        # Otherwise, prefer "nice" formatting
        ranked = sorted(formats, key=self._format_quality_score, reverse=True)
        return ranked[0]

    def _format_quality_score(self, format_str: str) -> int:
        """Score format quality - higher is better"""
        score = 0
        # Mixed case (e.g., "Node.js", "TypeScript") = best
        if any(c.isupper() for c in format_str) and any(
            c.islower() for c in format_str
        ):
            score += 10
        # Has special chars (e.g., "C++", ".NET") = good
        if any(c in format_str for c in ["+", "#", ".", "-", "/"]):
            score += 5
        # All uppercase = okay
        if format_str.isupper() and len(format_str) > 1:
            score += 2
        # All lowercase = least preferred
        if format_str.islower():
            score += 1
        return score


class ResumeJobMatcher:
    """Simple matcher using NLPSkillExtractor"""

    def __init__(self, skill_extractor: NLPSkillExtractor):
        self.skill_extractor = skill_extractor

    def match(self, resume_text: str, job_desc: str) -> dict:
        """Match resume against job description"""

        # Extract with best formatting
        job_skills = self.skill_extractor.extract_best_format(job_desc)
        resume_skills = self.skill_extractor.extract_best_format(resume_text)

        # Create lowercase maps for matching
        job_map = {s.lower(): s for s in job_skills}
        resume_map = {s.lower(): s for s in resume_skills}

        job_keys = set(job_map.keys())
        resume_keys = set(resume_map.keys())

        # Find matches
        matched_keys = job_keys & resume_keys
        missing_keys = job_keys - resume_keys

        if not job_keys:
            return self._empty_result("No skills found in job description")

        # Display using job description's format
        matched_display = [job_map[k] for k in sorted(matched_keys)]
        missing_display = [job_map[k] for k in sorted(missing_keys)]

        # Calculate score
        score = (len(matched_keys) / len(job_keys)) * 100

        return {
            "matched_keywords": matched_display,
            "missing_keywords": missing_display,
            "score": round(score, 2),
            "match_ratio": f"{len(matched_keys)}/{len(job_keys)}",
            "explanation": self._explain(matched_display, missing_display, score),
        }

    def _explain(self, matched: List[str], missing: List[str], score: float) -> str:
        explanation = []

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
        return {
            "matched_keywords": [],
            "missing_keywords": [],
            "score": 0.0,
            "match_ratio": "0/0",
            "explanation": message,
        }


if __name__ == "__main__":
    # Simple usage - one extractor, one matcher
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
    result = matcher.match(resume, job_desc)
    from pprint import pprint

    pprint(result)
    print(result["matched_keywords"])  # ['C++', 'Node.js'] - uses job's format!
    print(result["missing_keywords"])  # ['.NET']
