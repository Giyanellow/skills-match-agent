"""
Comprehensive test suite for src/core/extractor.py

Tests both NLPSkillExtractor and ResumeJobMatcher classes.
Uses real spaCy tokenization with mocked I/O and external APIs.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest
import responses
import spacy

from src.core.extractor import NLPSkillExtractor, ResumeJobMatcher


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_skills_data():
    """Sample skills data for mocking cache"""
    return [
        "python",
        "javascript",
        "java",
        "c++",
        "c#",
        "node.js",
        "react",
        "vue.js",
        ".net",
        "asp.net",
        "postgresql",
        "mysql",
        "mongodb",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "google cloud",
        "github actions",
        "ci/cd",
        "typescript",
        "r",  # Single letter exception
    ]


@pytest.fixture
def mock_cache_file(mock_skills_data, tmp_path):
    """Mock the combined_skills.json cache file"""
    cache_file = tmp_path / "combined_skills.json"
    cache_file.write_text(json.dumps(mock_skills_data))
    return cache_file


@pytest.fixture
def extractor_with_cache(mock_cache_file, monkeypatch):
    """Create NLPSkillExtractor with mocked cache file"""
    # Patch the cache file location
    monkeypatch.setattr(
        "src.core.extractor.NLPSkillExtractor.CACHE_FILE", str(mock_cache_file)
    )

    # Use real spaCy for tokenization
    with patch("src.core.extractor.spacy.load") as mock_load:
        mock_load.return_value = spacy.blank("en")
        extractor = NLPSkillExtractor(auto_update=False)
    
    return extractor


@pytest.fixture
def matcher(extractor_with_cache):
    """Create ResumeJobMatcher with mocked extractor"""
    return ResumeJobMatcher(extractor_with_cache)


# ============================================================================
# TEST CLASS: NLPSkillExtractor
# ============================================================================


class TestNLPSkillExtractor:
    """Tests for the NLPSkillExtractor class"""

    # ------------------------------------------------------------------------
    # Test Group: Skill Extraction Basics
    # ------------------------------------------------------------------------

    def test_extract_simple_skills(self, extractor_with_cache):
        """Extract common skills from text"""
        text = "I have experience with Python, JavaScript, and Java"
        skills = extractor_with_cache.extract_best_format(text)
        
        # Case-insensitive check
        skills_lower = [s.lower() for s in skills]
        assert "python" in skills_lower
        assert "javascript" in skills_lower
        assert "java" in skills_lower

    def test_extract_cpp_preserves_format(self, extractor_with_cache):
        """C++ should be extracted as a single skill with special chars"""
        text = "Expert in C++ programming and algorithms"
        skills = extractor_with_cache.extract_best_format(text)
        
        # Should extract C++ (or c++ depending on format)
        assert any("c++" in s.lower() for s in skills)

    def test_extract_nodejs_variations(self, extractor_with_cache):
        """Node.js and nodejs should be recognized as same skill"""
        text1 = "Experience with Node.js backend"
        text2 = "Experience with nodejs backend"
        
        skills1 = extractor_with_cache.extract_best_format(text1)
        skills2 = extractor_with_cache.extract_best_format(text2)
        
        # Both should extract node.js skill (normalized)
        skills1_lower = [s.lower() for s in skills1]
        skills2_lower = [s.lower() for s in skills2]
        
        assert "node.js" in skills1_lower or "nodejs" in skills1_lower
        assert "node.js" in skills2_lower or "nodejs" in skills2_lower

    def test_extract_dotnet_formats(self, extractor_with_cache):
        """Extract .NET with proper formatting"""
        text = "Experience with .NET framework and ASP.NET"
        skills = extractor_with_cache.extract_best_format(text)
        
        skills_lower = [s.lower() for s in skills]
        assert ".net" in skills_lower or "dotnet" in skills_lower
        assert "asp.net" in skills_lower

    def test_extract_multi_word_skills(self, extractor_with_cache):
        """Extract multi-word skills like 'Google Cloud', 'GitHub Actions'"""
        text = "Deployed on Google Cloud using GitHub Actions for CI/CD"
        skills = extractor_with_cache.extract_best_format(text)
        
        skills_lower = [s.lower() for s in skills]
        assert "google cloud" in skills_lower
        assert "github actions" in skills_lower
        assert "ci/cd" in skills_lower

    def test_extract_case_insensitive(self, extractor_with_cache):
        """Extraction should be case-insensitive"""
        text = "PYTHON, python, Python - all should work"
        skills = extractor_with_cache.extract_best_format(text)
        
        # Should only extract Python once (deduplicated)
        python_count = sum(1 for s in skills if s.lower() == "python")
        assert python_count == 1

    def test_extract_mixed_case_quality(self, extractor_with_cache):
        """When skill appears in multiple cases, prefer mixed case"""
        text = "Python and PYTHON and python"
        skills = extractor_with_cache.extract_best_format(text)
        
        # Should pick "Python" (mixed case) over "PYTHON" or "python"
        assert "Python" in skills

    # ------------------------------------------------------------------------
    # Test Group: Format Quality Scoring
    # ------------------------------------------------------------------------

    def test_format_quality_mixed_case_highest(self, extractor_with_cache):
        """Mixed case format should score highest"""
        formats = ["python", "PYTHON", "Python"]
        best = extractor_with_cache._pick_best_format(formats)
        assert best == "Python"

    def test_format_quality_special_chars(self, extractor_with_cache):
        """Special characters in format should score high"""
        formats = ["cpp", "c++", "C++"]
        best = extractor_with_cache._pick_best_format(formats)
        assert best == "C++"  # Mixed case + special chars

    def test_format_quality_frequency_wins(self, extractor_with_cache):
        """When one format appears more frequently, it wins"""
        formats = ["nodejs", "nodejs", "Node.js"]
        best = extractor_with_cache._pick_best_format(formats)
        assert best == "nodejs"  # Appears twice

    def test_format_quality_score_calculation(self, extractor_with_cache):
        """Test the _format_quality_score function directly"""
        score_mixed = extractor_with_cache._format_quality_score("Node.js")
        score_upper = extractor_with_cache._format_quality_score("NODEJS")
        score_lower = extractor_with_cache._format_quality_score("nodejs")
        
        # Mixed case + special chars should score highest
        assert score_mixed > score_upper > score_lower

    # ------------------------------------------------------------------------
    # Test Group: Skill Loading & Caching
    # ------------------------------------------------------------------------

    def test_load_from_cache(self, mock_cache_file, monkeypatch):
        """Should load skills from cache when auto_update=False"""
        monkeypatch.setattr(
            "src.core.extractor.NLPSkillExtractor.CACHE_FILE", str(mock_cache_file)
        )
        
        with patch("src.core.extractor.spacy.load") as mock_load:
            mock_load.return_value = spacy.blank("en")
            extractor = NLPSkillExtractor(auto_update=False)
        
        # Should have loaded skills from cache
        assert len(extractor.skills) > 0
        assert "python" in extractor.skills
        assert "javascript" in extractor.skills

    @responses.activate
    def test_build_skills_from_github_api(self, tmp_path, monkeypatch):
        """Should fetch skills from GitHub API when cache missing"""
        # Mock GitHub API response
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/github/linguist/master/lib/linguist/languages.yml",
            body="Python:\n  type: programming\nJavaScript:\n  type: programming\nRuby:\n  type: programming",
            status=200,
        )
        
        # Use non-existent cache file
        cache_file = tmp_path / "new_cache.json"
        monkeypatch.setattr(
            "src.core.extractor.NLPSkillExtractor.CACHE_FILE", str(cache_file)
        )
        
        with patch("src.core.extractor.spacy.load") as mock_load:
            mock_load.return_value = spacy.blank("en")
            extractor = NLPSkillExtractor(auto_update=True)
        
        # Should have built taxonomy from API + curated skills
        assert len(extractor.skills) > 0
        assert "python" in extractor.skills
        
        # Cache file should be created
        assert cache_file.exists()

    @responses.activate
    def test_github_api_failure_fallback(self, tmp_path, monkeypatch):
        """Should fall back to curated skills if GitHub API fails"""
        # Mock GitHub API to fail
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/github/linguist/master/lib/linguist/languages.yml",
            body="Error",
            status=500,
        )
        
        cache_file = tmp_path / "new_cache.json"
        monkeypatch.setattr(
            "src.core.extractor.NLPSkillExtractor.CACHE_FILE", str(cache_file)
        )
        
        with patch("src.core.extractor.spacy.load") as mock_load:
            mock_load.return_value = spacy.blank("en")
            extractor = NLPSkillExtractor(auto_update=True)
        
        # Should still have curated skills
        assert "react" in extractor.skills
        assert "docker" in extractor.skills

    def test_curated_skills_present(self, extractor_with_cache):
        """Verify important curated skills are in taxonomy"""
        curated_expected = [
            "react",
            "docker",
            "postgresql",
            "aws",
            "kubernetes",
            "ci/cd",
        ]
        
        for skill in curated_expected:
            assert skill in extractor_with_cache.skills

    # ------------------------------------------------------------------------
    # Test Group: Variation Mapping
    # ------------------------------------------------------------------------

    def test_variation_map_nodejs(self, extractor_with_cache):
        """nodejs should map to node.js variations"""
        variations = extractor_with_cache.skill_variations.get("node.js", set())
        
        # Should have variations
        assert len(variations) > 0
        # At minimum, should have the canonical form
        assert "node.js" in variations

    def test_variation_map_dotnet(self, extractor_with_cache):
        """dotnet should map to .net"""
        variations = extractor_with_cache.skill_variations.get(".net", set())
        
        assert ".net" in variations

    def test_extract_with_variations(self, extractor_with_cache):
        """Variations should be recognized during extraction"""
        # Test both variations
        text1 = "nodejs developer"
        text2 = "Node.js developer"
        
        skills1 = extractor_with_cache.extract_best_format(text1)
        skills2 = extractor_with_cache.extract_best_format(text2)
        
        # Both should extract something related to node
        assert len(skills1) > 0
        assert len(skills2) > 0

    # ------------------------------------------------------------------------
    # Test Group: Edge Cases
    # ------------------------------------------------------------------------

    def test_extract_empty_text(self, extractor_with_cache):
        """Empty text should return empty list"""
        skills = extractor_with_cache.extract_best_format("")
        assert skills == []

    def test_extract_none_text(self, extractor_with_cache):
        """None text should return empty list"""
        skills = extractor_with_cache.extract_best_format(None)
        assert skills == []

    def test_single_letter_filtering(self, extractor_with_cache):
        """Single letter 'C' should be filtered out, but 'R' should stay"""
        text = "I know C and R programming languages"
        skills = extractor_with_cache.extract_best_format(text)
        
        skills_lower = [s.lower() for s in skills]
        
        # 'R' should be present (it's an exception)
        assert "r" in skills_lower
        
        # Standalone 'C' should not be present (unless part of C++)
        standalone_c = [s for s in skills if s.lower() == "c"]
        assert len(standalone_c) == 0

    def test_no_skills_in_text(self, extractor_with_cache):
        """Text with no skills should return empty list"""
        text = "I love cooking and playing tennis"
        skills = extractor_with_cache.extract_best_format(text)
        assert skills == []


# ============================================================================
# TEST CLASS: ResumeJobMatcher
# ============================================================================


class TestResumeJobMatcher:
    """Tests for the ResumeJobMatcher class"""

    # ------------------------------------------------------------------------
    # Test Group: Matching Logic
    # ------------------------------------------------------------------------

    def test_perfect_match(self, matcher):
        """All required skills present should give 100% score"""
        job_desc = "Requirements: Python, JavaScript, Docker"
        resume = "Skills: Python, JavaScript, Docker, AWS"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 100.0
        assert len(result["matched_keywords"]) == 3
        assert len(result["missing_keywords"]) == 0
        assert result["match_ratio"] == "3/3"

    def test_partial_match(self, matcher):
        """Partial skill match should calculate correct percentage"""
        job_desc = "Requirements: Python, JavaScript, Docker, Kubernetes"
        resume = "Skills: Python, JavaScript"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 50.0  # 2 out of 4
        assert len(result["matched_keywords"]) == 2
        assert len(result["missing_keywords"]) == 2
        assert result["match_ratio"] == "2/4"

    def test_no_match(self, matcher):
        """No matching skills should give 0% score"""
        job_desc = "Requirements: Python, JavaScript"
        resume = "Skills: Marketing, Sales"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 0.0
        assert len(result["matched_keywords"]) == 0
        assert len(result["missing_keywords"]) == 2

    def test_case_insensitive_matching(self, matcher):
        """Matching should be case-insensitive"""
        job_desc = "Requirements: python, javascript"
        resume = "Skills: PYTHON, JAVASCRIPT"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 100.0
        assert len(result["matched_keywords"]) == 2

    def test_extra_resume_skills_ignored(self, matcher):
        """Extra skills in resume shouldn't affect score negatively"""
        job_desc = "Requirements: Python"
        resume = "Skills: Python, JavaScript, Docker, Kubernetes, AWS, React"
        
        result = matcher.match(resume, job_desc)
        
        # Should still be 100% (1/1 required skills matched)
        assert result["score"] == 100.0
        assert result["match_ratio"] == "1/1"

    # ------------------------------------------------------------------------
    # Test Group: Format Preservation
    # ------------------------------------------------------------------------

    def test_uses_job_description_format(self, matcher):
        """Output should use job description's format, not resume's"""
        job_desc = "Requirements: C++ and Python"
        resume = "Skills: c++ and python"
        
        result = matcher.match(resume, job_desc)
        
        # Should use job's format (C++ not c++)
        assert "C++" in result["matched_keywords"]
        assert "Python" in result["matched_keywords"]
        
        # Should NOT use resume's lowercase format
        assert "c++" not in result["matched_keywords"]

    def test_missing_skills_use_job_format(self, matcher):
        """Missing skills should also use job description's format"""
        job_desc = "Requirements: Python, .NET, PostgreSQL"
        resume = "Skills: python"
        
        result = matcher.match(resume, job_desc)
        
        # Missing skills should use job's format
        assert ".NET" in result["missing_keywords"]
        assert "PostgreSQL" in result["missing_keywords"]

    # ------------------------------------------------------------------------
    # Test Group: Score Calculation
    # ------------------------------------------------------------------------

    def test_score_calculation_one_third(self, matcher):
        """1 out of 3 skills should be 33.33%"""
        job_desc = "Requirements: Python, JavaScript, Docker"
        resume = "Skills: Python"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 33.33  # Rounded to 2 decimals
        assert result["match_ratio"] == "1/3"

    def test_score_calculation_two_thirds(self, matcher):
        """2 out of 3 skills should be 66.67%"""
        job_desc = "Requirements: Python, JavaScript, Docker"
        resume = "Skills: Python, JavaScript"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 66.67
        assert result["match_ratio"] == "2/3"

    def test_score_rounding(self, matcher):
        """Score should be rounded to 2 decimal places"""
        job_desc = "Requirements: Python, JavaScript, Docker, AWS, React, Vue.js, TypeScript"
        resume = "Skills: Python, JavaScript"
        
        result = matcher.match(resume, job_desc)
        
        # 2/7 = 28.571428...
        assert result["score"] == 28.57
        assert isinstance(result["score"], float)

    # ------------------------------------------------------------------------
    # Test Group: Explanation Text
    # ------------------------------------------------------------------------

    def test_explanation_strong_match(self, matcher):
        """Score >= 80% should show 'Strong match'"""
        job_desc = "Requirements: Python, JavaScript, Docker, AWS, React"
        resume = "Skills: Python, JavaScript, Docker, AWS"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 80.0
        assert "✓ Strong match" in result["explanation"]

    def test_explanation_good_match(self, matcher):
        """60% <= score < 80% should show 'Good match'"""
        job_desc = "Requirements: Python, JavaScript, Docker"
        resume = "Skills: Python, JavaScript"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 66.67
        assert "~ Good match" in result["explanation"]

    def test_explanation_partial_match(self, matcher):
        """40% <= score < 60% should show 'Partial match'"""
        job_desc = "Requirements: Python, JavaScript"
        resume = "Skills: Python"
        
        result = matcher.match(resume, job_desc)
        
        # 1/2 = 50% (partial match range)
        assert result["score"] == 50.0
        assert "△ Partial match" in result["explanation"]

    def test_explanation_weak_match(self, matcher):
        """Score < 40% should show 'Weak match'"""
        job_desc = "Requirements: Python, JavaScript, Docker, AWS, Kubernetes"
        resume = "Skills: Python"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 20.0
        assert "✗ Weak match" in result["explanation"]

    def test_explanation_includes_matched_skills(self, matcher):
        """Explanation should list matched skills"""
        job_desc = "Requirements: Python, JavaScript"
        resume = "Skills: Python, JavaScript"
        
        result = matcher.match(resume, job_desc)
        
        assert "Matched skills" in result["explanation"]
        assert "Python" in result["explanation"]
        assert "JavaScript" in result["explanation"]

    def test_explanation_includes_missing_skills(self, matcher):
        """Explanation should list missing skills"""
        job_desc = "Requirements: Python, JavaScript, Docker"
        resume = "Skills: Python"
        
        result = matcher.match(resume, job_desc)
        
        assert "Missing skills" in result["explanation"]
        assert "JavaScript" in result["explanation"]
        assert "Docker" in result["explanation"]

    # ------------------------------------------------------------------------
    # Test Group: Edge Cases
    # ------------------------------------------------------------------------

    def test_empty_job_description(self, matcher):
        """Empty job description should return appropriate message"""
        job_desc = ""
        resume = "Skills: Python, JavaScript"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 0.0
        assert result["match_ratio"] == "0/0"
        assert "No skills found in job description" in result["explanation"]

    def test_empty_resume(self, matcher):
        """Empty resume should match 0 skills"""
        job_desc = "Requirements: Python, JavaScript"
        resume = ""
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 0.0
        assert len(result["matched_keywords"]) == 0
        assert len(result["missing_keywords"]) == 2

    def test_both_empty(self, matcher):
        """Both empty should return empty result"""
        result = matcher.match("", "")
        
        assert result["score"] == 0.0
        assert result["match_ratio"] == "0/0"
        assert len(result["matched_keywords"]) == 0
        assert len(result["missing_keywords"]) == 0

    def test_special_characters_in_text(self, matcher):
        """Should handle special characters gracefully"""
        job_desc = "Requirements: C++, C#, .NET, Node.js"
        resume = "Skills: C++, C#, .NET, Node.js"
        
        result = matcher.match(resume, job_desc)
        
        assert result["score"] == 100.0
        assert len(result["matched_keywords"]) == 4

    def test_realistic_scenario(self, matcher):
        """Test with realistic job description and resume"""
        job_desc = """
        Senior Backend Developer
        
        Required Skills:
        - Python programming
        - Node.js and JavaScript
        - PostgreSQL or MySQL
        - Docker and Kubernetes
        - AWS cloud services
        - REST API design
        - CI/CD experience
        """
        
        resume = """
        John Doe - Software Engineer
        
        Technical Skills:
        - Expert in Python and JavaScript
        - Backend development with Node.js
        - Database: PostgreSQL, MongoDB
        - Containerization: Docker
        - Cloud: AWS (EC2, S3, Lambda)
        - API development (REST, GraphQL)
        """
        
        result = matcher.match(resume, job_desc)
        
        # Should find matches for: Python, Node.js, JavaScript, PostgreSQL, Docker, AWS
        assert result["score"] > 50.0  # Good match
        assert len(result["matched_keywords"]) >= 5
        assert "Python" in str(result["matched_keywords"])
        assert "Docker" in str(result["matched_keywords"])


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for the full workflow"""

    def test_end_to_end_matching_workflow(self, extractor_with_cache):
        """Test complete workflow from extraction to matching"""
        matcher = ResumeJobMatcher(extractor_with_cache)
        
        job_desc = "Looking for Python, Docker, and AWS expertise"
        resume = "I have 5 years of Python and Docker experience"
        
        result = matcher.match(resume, job_desc)
        
        assert isinstance(result, dict)
        assert "score" in result
        assert "matched_keywords" in result
        assert "missing_keywords" in result
        assert "match_ratio" in result
        assert "explanation" in result
        
        # Should match Python and Docker
        assert result["score"] == 66.67  # 2 out of 3
        assert len(result["matched_keywords"]) == 2
        assert "AWS" in result["missing_keywords"]

    def test_multiple_resumes_against_same_job(self, matcher):
        """Test matching multiple resumes against same job description"""
        job_desc = "Requirements: Python, Docker, AWS"
        
        resume1 = "Skills: Python, Docker, AWS"
        resume2 = "Skills: Python, Docker"
        resume3 = "Skills: Python"
        
        result1 = matcher.match(resume1, job_desc)
        result2 = matcher.match(resume2, job_desc)
        result3 = matcher.match(resume3, job_desc)
        
        # Scores should be descending
        assert result1["score"] > result2["score"] > result3["score"]
        assert result1["score"] == 100.0
        assert result2["score"] == 66.67
        assert result3["score"] == 33.33
