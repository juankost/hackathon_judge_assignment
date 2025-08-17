"""
LLM-powered preprocessing of judges and participants data into JSON formats required
by the judge assignment algorithm.

This module uses Pydantic models to define the structured output format and leverages
Google's Gemini API for intelligent extraction of information from raw text inputs.

HACKATHON CONTEXT:
- Problems are sponsored by companies (e.g., "Google", "Microsoft", "Apple")
- Specialized judges work at the sponsoring companies and judge their company's problems
- Flexible judges work at universities/academic institutions and can judge any problem

Expected JSON formats produced by this module:

- participants.json:
  {
    "<participant_id>": {"problem": "<company_name>", "name": "<optional name>", ...metadata}
  }

- judges.json:
  {
    "<judge_id>": {"name": "<optional name>", "problem": "<company_name_or_null>"}
  }
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Structured outputs for the LLM calls
class ParticipantInfo(BaseModel):
    """Information about a single participant."""

    problem: str = Field(description="The name of this participant's project")
    sponsoring_company: Optional[str] = Field(
        default=None, description="The sponsoring company name for this participant's project"
    )
    name: Optional[str] = Field(
        default=None, description="Optional display/full name of the participant"
    )


class JudgeInfo(BaseModel):
    """Information about a single judge."""

    name: Optional[str] = Field(default=None, description="Optional display/full name of the judge")
    company: Optional[str] = Field(
        default=None,
        description="Company name for specialized judges (judges working at sponsoring companies), null for flexible judges (university/academic judges)",
    )


class ParticipantsData(BaseModel):
    """Collection of all participants with their IDs as keys."""

    participants: Dict[str, ParticipantInfo] = Field(
        description="Dictionary mapping participant IDs to participant information"
    )


class JudgesData(BaseModel):
    """Collection of all judges with their IDs as keys."""

    judges: Dict[str, JudgeInfo] = Field(
        description="Dictionary mapping judge IDs to judge information"
    )


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""

    model: str = "gemini-2.5-pro"
    temperature: float = 0.0


# Utility functions
def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _ensure_dirs(inputs_dir: str, processed_dir: str) -> None:
    """Ensure required directories exist (relative to project root if needed)."""
    base = _project_root()
    inputs_dir_abs = inputs_dir if os.path.isabs(inputs_dir) else os.path.join(base, inputs_dir)
    processed_dir_abs = (
        processed_dir if os.path.isabs(processed_dir) else os.path.join(base, processed_dir)
    )
    os.makedirs(inputs_dir_abs, exist_ok=True)
    os.makedirs(processed_dir_abs, exist_ok=True)


def _get_gemini_client():
    """Get Google Gemini client using API key from environment."""
    try:
        import google.genai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please set your Gemini API key in the .env file or environment."
            )

        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        raise ImportError(
            "google-genai package not found. " "Please install it with: uv add google-genai"
        )


def _maybe_load_existing_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to load text as JSON if it's already in the correct format."""
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_problems_from_participants(participants_data: ParticipantsData) -> List[str]:
    """Extract unique company names from participants data."""
    companies = set()
    for participant in participants_data:
        if participant.sponsoring_company is not None:
            companies.add(participant.sponsoring_company)
    return list(companies)


# Function to extract structured data from text using Gemini LLM with Pydantic schema
def _extract_with_llm(
    client,
    text: str,
    data_type: str,
    response_schema: type[BaseModel],
    config: PreprocessingConfig,
    known_companies: Optional[list] = None,
) -> BaseModel:
    """Extract structured data from text using Gemini LLM with Pydantic schema."""

    if data_type == "company_to_problem":
        prompt = f"""
        You are an expert at extracting structured information from raw text. The text describes a list of companies in a hackathon, also giving information on the problem that they are tackling (and potentially the sponsoring company).
        
        Given the following text containing company descriptions, extract the following structured information 
        for each company. 
        """
        # TODO: Implement this

    if data_type == "participants":
        prompt = f"""
        You are an expert at extracting structured information from raw text. The text describes a list of participants in a hackathon, also giving information on the problem that they are tackling (and potentially the sponsoring company).
        
        Given the following text containing participant descriptions, extract the following structured information 
        for each participant. 
        
        - problem: the name of the problem that the participant is tackling
        - sponsoring_company: the sponsoring company of the problem (if any)
        - name: the name of the participant (if any) --> If not Name is provided, assign a generic sequential ID to the participant (e.g. p1, p2, p3, etc.)
        
        CONTEXT: In this hackathon, problems can be sponsored by companies. Each participant works on 
        a problem/project that may be associated with a company, but also has a freedom to work on his/her own problem, not related to any company.

        Raw text:
        {text}
        
        Return the data in JSON format.
        """
    else:  # judges

        assert (
            known_companies is not None
        ), "known_companies must be provided for the judges extraction"

        prompt = f"""
        You are an expert at extracting structured information from raw text.
        
        The text describes a list of judges in a hackathon. For some of them it also mentions the company they work for. Other judges, on the other hand, are not associated with any company, but rather academic institutions/NGOs/or nothing at all.

        TASK: Given the following text containing judge descriptions, extract the following structured information 
        for each judge. 
        - name: the name of the judge (REQUIRED - extract from text, never use generic names)
        - company: The company name if they work at one of the sponsoring companies, or null if they are university/academic/independent judges
        
        CONTEXT: In this hackathon, problems are sponsored by companies. Judges can be either:
        1. SPECIALIZED: Work at one of the sponsoring companies and judge their company's problem
        2. FLEXIBLE: Work at universities or other non-sponsoring organizations and can judge any problem
        
        List of sponsoring companies: {', '.join(known_companies)}
        Raw text: {text}
        
        RULES for determining specialization:
        - If the judge works at a company that matches one of the known sponsoring companies, set that company as their problem. USE THE EXACT NAME OF THE COMPANY as given in the list of known companies.
        - If the judge works at a university, is described as academic, or works at a non-sponsoring organization, set company to null
        - If they are described as "flexible", "generalist", "can judge any category", or similar, set company to null
        - If no company affiliation is mentioned or the company doesn't match known sponsors, set company to null
        
        Return the data in JSON format.
        """

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "temperature": config.temperature,
            },
        )
        return response.parsed

    except Exception as e:
        raise ValueError(f"Failed to extract {data_type} data using LLM: {str(e)}")


# Main function to preprocess files
def preprocess_files(
    judges_input_path: str,
    participants_input_path: str,
    processed_dir: str = "data/processed",
    config: Optional[PreprocessingConfig] = None,
) -> Tuple[str, str]:  # judges_out_path, participants_out_path
    """
    Preprocess judges and participants files using LLM extraction with Pydantic models.

    Implementation follows a two-stage approach:
    1. First extract participants to get the list of all sponsoring companies
    2. Then extract judges using the known companies list to determine specialization
       (company employees = specialized, university/academic = flexible)

    Args:
        judges_input_path: Path to raw judges file (text or JSON)
        participants_input_path: Path to raw participants file (text or JSON)
        processed_dir: Directory to save processed JSON files
        config: Optional preprocessing configuration

    Returns:
        Tuple of (judges_out_path, participants_out_path)
    """
    if config is None:
        config = PreprocessingConfig()

    # Ensure directories exist
    base = _project_root()
    inputs_dir = os.path.dirname(judges_input_path) or os.path.join(base, "data/inputs")
    _ensure_dirs(inputs_dir, processed_dir)

    # Read input files
    with open(judges_input_path, "r", encoding="utf-8") as f:
        judges_raw = f.read()
    with open(participants_input_path, "r", encoding="utf-8") as f:
        participants_raw = f.read()

    # Check if inputs are already valid JSON
    judges_existing = _maybe_load_existing_json(judges_raw)
    participants_existing = _maybe_load_existing_json(participants_raw)

    # Get Gemini client for LLM processing
    client = _get_gemini_client()

    # STAGE 1: Process participants first to extract company list
    if participants_existing is not None:
        # Already valid JSON, validate and normalize format
        participants_data = ParticipantsData(participants=participants_existing)
        known_companies = _extract_problems_from_participants(participants_existing)
    else:
        # Extract using LLM
        participants_data = _extract_with_llm(
            client, participants_raw, "participants", ParticipantsData, config
        )
        known_companies = _extract_problems_from_participants(participants_data.participants)

    # STAGE 2: Process judges using the known company list
    if judges_existing is not None:
        # Already valid JSON, validate and normalize format
        judges_data = JudgesData(judges=judges_existing)
    else:
        # Extract using LLM with known companies context
        judges_data = _extract_with_llm(
            client, judges_raw, "judges", JudgesData, config, known_companies
        )

    # Validate that we have data
    if not participants_data.participants:
        raise ValueError("No participants were extracted from the input")
    if not judges_data.judges:
        raise ValueError("No judges were extracted from the input")

    # Save processed files
    participants_out_path = os.path.join(processed_dir, "participants.json")
    judges_out_path = os.path.join(processed_dir, "judges.json")

    with open(participants_out_path, "w", encoding="utf-8") as f:
        json.dump(participants_data, f, indent=2)

    with open(judges_out_path, "w", encoding="utf-8") as f:
        json.dump(judges_data, f, indent=2)

    # Return the paths to the processed files
    return judges_out_path, participants_out_path


def main():
    """Command-line interface for preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess judges and participants data using LLM extraction"
    )
    parser.add_argument("--judges", required=True, help="Path to judges raw text or JSON file")
    parser.add_argument(
        "--participants", required=True, help="Path to participants raw text or JSON file"
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Output directory for processed JSON files",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-pro", help="Gemini model to use for extraction"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (0.0 for deterministic)",
    )

    args = parser.parse_args()

    config = PreprocessingConfig(model=args.model, temperature=args.temperature)

    try:
        # Ensure directories exist
        _ensure_dirs("data/inputs", args.processed_dir)

        # Process files
        judges_data, participants_data = preprocess_files(
            args.judges, args.participants, args.processed_dir, config
        )

        # Output result paths as JSON for easy consumption
        result = {
            "judges": judges_data.judges,
            "participants": participants_data.participants,
            "status": "success",
        }
        print(json.dumps(result, indent=2))

    except Exception as e:
        # Output error information
        error_result = {"error": str(e), "status": "failed"}
        print(json.dumps(error_result, indent=2))
        exit(1)


if __name__ == "__main__":
    main()
