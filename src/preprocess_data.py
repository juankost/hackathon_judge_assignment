"""
LLM-powered preprocessing of judges and participants data into JSON formats required
by the judge assignment algorithm.

This module uses Pydantic models to define the structured output format and leverages
Google's Gemini API for intelligent extraction of information from raw text inputs.

HACKATHON CONTEXT:
- Problems are sponsored by companies (e.g., "Google", "Microsoft", "Apple")
- Specialized judges work at the sponsoring companies and judge their company's problems
- Flexible judges work at universities/academic institutions and can judge any problem

"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# Structured outputs for the LLM calls
class ParticipantInfo(BaseModel):
    """Information about a single participant."""

    full_name: Optional[str] = Field(
        default=None, description="Optional display/full name of the participant"
    )
    participant_id: str = Field(
        description="The unique ID for the participant, using P1, P2, P3, etc."
    )
    problem: str = Field(description="The name of this participant's project")
    problem_id: str = Field(description="The unique ID for the problem")
    sponsoring_company: Optional[str] = Field(
        default=None, description="The sponsoring company name for this participant's project"
    )


class JudgeInfo(BaseModel):
    """Information about a single judge."""

    full_name: Optional[str] = Field(
        default=None, description="Optional display/full name of the judge"
    )
    judge_id: str = Field(description="The unique ID for the judge")
    company: Optional[str] = Field(
        default=None,
        description="Company name for specialized judges (judges working at sponsoring companies), null for flexible judges (university/academic judges)",
    )
    problem: Optional[str] = Field(
        default=None,
        description="The name of the problem that the judge is judging, if he is a specialized judge, null for flexible judges",
    )
    problem_id: str = Field(description="The unique ID for the problem")


class ParticipantsData(BaseModel):
    """Collection of all participants."""

    participants: list[ParticipantInfo] = Field(description="List of participants")


class JudgesData(BaseModel):
    """Collection of all judges."""

    judges: list[JudgeInfo] = Field(description="List of judges")


class CompanyToProblemMapping(BaseModel):

    company: str = Field(description="The name of the company")
    problem: str = Field(description="The name of the problem that the company is tackling")
    problem_id: str = Field(description="The unique ID for the problem")


class CompanyToProblemData(BaseModel):
    """Collection of all company to problem mappings."""

    mapping: list[CompanyToProblemMapping] = Field(
        description="List of company to problem mappings"
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
    logger.debug(
        "Ensuring directories exist: inputs_dir=%s, processed_dir=%s",
        inputs_dir_abs,
        processed_dir_abs,
    )
    os.makedirs(inputs_dir_abs, exist_ok=True)
    os.makedirs(processed_dir_abs, exist_ok=True)
    logger.debug("Directories ensured")


def _get_gemini_client():
    """Get Google Gemini client using API key from environment."""
    try:
        logger.info("Initializing Gemini client")
        import google.genai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please set your Gemini API key in the .env file or environment."
            )

        client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized")
        return client
    except ImportError:
        raise ImportError(
            "google-genai package not found. " "Please install it with: uv add google-genai"
        )


# Function to extract structured data from text using Gemini LLM with Pydantic schema
def _extract_with_llm(
    client,
    text: str,
    data_type: str,
    response_schema: type[BaseModel],
    config: PreprocessingConfig,
    known_companies: Optional[list] = None,
    company_to_problem_data: Optional[CompanyToProblemData] = None,
) -> BaseModel:
    """Extract structured data from text using Gemini LLM with Pydantic schema."""

    logger.info("Extracting %s data with LLM", data_type)
    logger.debug("Input text length: %d characters", len(text) if text is not None else 0)

    if data_type == "company_to_problem":
        prompt = f"""
        You are an expert at extracting structured information from raw text. The text describes a list of companies in a hackathon, also giving information on the problem that they are sponsoring / proposing.

        Given the following text containing company descriptions, extract the following structured information 
        for each company. 

        - company: the name of the company
        - problem: the name of the problem that the company is sponsoring / proposing
        - problem_id: create a unique ID for each problem, using X1, X2, X3, etc.

        Raw text: 
        {text}

        Return the data in JSON format.
        """

    elif data_type == "participants":

        assert (
            known_companies is not None
        ), "known_companies must be provided for the participants extraction"

        assert (
            company_to_problem_data is not None
        ), "company_to_problem_data must be provided for the participants extraction"

        # Prepare a readable mapping string to guide the model to use exact names
        mapping_str = None
        if company_to_problem_data is not None:
            try:
                mapping_items = [
                    f"- {m.company}: {m.problem} (ID: {m.problem_id})"
                    for m in company_to_problem_data.mapping
                ]
                mapping_str = "\n".join(mapping_items)
            except Exception:
                mapping_str = str(company_to_problem_data)

        known_companies_str = ", ".join(known_companies) if known_companies else ""

        prompt = f"""
        You are an expert at extracting structured information from raw text. The text describes a list of participants in a hackathon, also giving information on the problem that they are tackling (and potentially the sponsoring company).
        
        Given the following text containing participant descriptions, extract the following structured information for each participant. 
        
        - full_name: the name of the participant (if any) --> If not Name is provided, assign a generic sequential ID to the participant (e.g. p1, p2, p3, etc.)
        - participant_id: create a unique ID for each participant, using P1, P2, P3, etc.
        - problem: the name of the problem that the participant is tackling
        - problem_id: the unique ID for the problem, use the same ID as the one in the mapping for the problem. If the problem is not sponsored by a company, create a new ID for the problem that is not in the mapping.
        - sponsoring_company: the sponsoring company of the problem (if any)
                
        CONTEXT: In this hackathon, problems can be sponsored by companies. Each participant works on 
        a problem/project that may be associated with a company, but can also work on their own problem not related to any company.

        IMPORTANT RULES:
        - If the participant is working on a sponsored problem and a sponsoring company can be inferred, use the EXACT company and problem names from the mapping below.
        - If the participant is working on their own/independent problem, set sponsoring_company to null and set problem based on the raw text as written. Create a new problem ID for the problem that is not in the mapping.
        - List of sponsoring companies (for reference): {known_companies_str}
        - Mapping company -> problem (use exact strings):\n{mapping_str}

        Raw text:
        {text}
        
        Return the data in JSON format.
        """
    elif data_type == "judges":

        assert (
            known_companies is not None
        ), "known_companies must be provided for the judges extraction"

        assert (
            company_to_problem_data is not None
        ), "company_to_problem_data must be provided for the judges extraction"

        # Prepare a readable mapping string
        mapping_str = None
        if company_to_problem_data is not None:
            try:
                mapping_items = [
                    f"- {m.company}: {m.problem} (ID: {m.problem_id})"
                    for m in company_to_problem_data.mapping
                ]
                mapping_str = "\n".join(mapping_items)
            except Exception:
                mapping_str = str(company_to_problem_data)

        prompt = f"""
        You are an expert at extracting structured information from raw text.
        
        The text describes a list of judges in a hackathon. For some of them it also mentions the company they work for. Other judges, on the other hand, are not associated with any company, but rather academic institutions/NGOs/or nothing at all.

        TASK: Given the following text containing judge descriptions, extract the following structured information 
        for each judge. 
        - full_name: the name of the judge (REQUIRED - extract from text, never use generic names)
        - judge_id: create a unique ID for each judge, using J1, J2, J3, etc.
        - company: The company name if they work at one of the sponsoring companies, or null if they are university/academic/independent judges
        - problem: If company is set (specialized judge), set to the EXACT problem name from the mapping for that company; otherwise set to null
        - problem_id: the unique ID for the problem, use the same ID as the one in the mapping for the problem. If the problem is null, set also the problem_id to null.

        CONTEXT: In this hackathon, problems are sponsored by companies. Judges can be either:
        1. SPECIALIZED: Work at one of the sponsoring companies and judge their company's problem
        2. FLEXIBLE: Work at universities or other non-sponsoring organizations and can judge any problem
        
        List of sponsoring companies: {', '.join(known_companies)}
        Mapping of companies to the problems they are sponsoring (use exact strings):
        {mapping_str}
        Raw text: 
        {text}
        
        RULES for determining specialization:
        - If the judge works at a company that matches one of the known sponsoring companies, set that company as their problem. USE THE EXACT NAME OF THE COMPANY as given in the list of known companies.
        - If the judge works at a university, is described as academic, or works at a non-sponsoring organization, set company to null
        - If they are described as "flexible", "generalist", "can judge any category", or similar, set company to null
        - If no company affiliation is mentioned or the company doesn't match known sponsors, set company to null
        
        Return the data in JSON format.
        """
    else:
        logger.error("Invalid data type: %s", data_type)
        raise ValueError(f"Invalid data type: {data_type}")

    try:
        logger.debug("Prompt: %s", prompt)
        response = client.models.generate_content(
            model=config.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "temperature": config.temperature,
            },
        )
        logger.debug("Response: %s", response.parsed.model_dump_json(indent=2))
        logger.info("LLM extraction for %s completed", data_type)
        return response.parsed

    except Exception as e:
        logger.exception("LLM extraction for %s failed", data_type)
        raise ValueError(f"Failed to extract {data_type} data using LLM: {str(e)}")


# Main function to preprocess files
def preprocess_files(
    judges_input_path: str,
    participants_input_path: str,
    company_to_problem_input_path: str,
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
        company_to_problem_input_path: Path to raw company to problem mapping file (text or JSON)
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
    logger.info("Starting preprocessing")
    logger.debug(
        "Input paths: judges=%s, participants=%s, company_to_problem=%s; processed_dir=%s",
        judges_input_path,
        participants_input_path,
        company_to_problem_input_path,
        processed_dir,
    )

    # Read input files
    with open(judges_input_path, "r", encoding="utf-8") as f:
        judges_raw = f.read()
    with open(participants_input_path, "r", encoding="utf-8") as f:
        participants_raw = f.read()
    with open(company_to_problem_input_path, "r", encoding="utf-8") as f:
        company_to_problem_raw = f.read()
    logger.debug(
        "Loaded inputs: judges=%d chars, participants=%d chars, company_to_problem=%d chars",
        len(judges_raw),
        len(participants_raw),
        len(company_to_problem_raw),
    )

    # Get Gemini client for LLM processing
    client = _get_gemini_client()
    logger.info("Gemini client ready")

    # STAGE 1: Process company to problem mapping
    logger.info("Stage 1: Extracting company-to-problem mapping")
    company_to_problem_data = _extract_with_llm(
        client=client,
        text=company_to_problem_raw,
        data_type="company_to_problem",
        response_schema=CompanyToProblemData,
        config=config,
    )

    # Derive known companies directly from the mapping extraction
    known_companies = [m.company for m in company_to_problem_data.mapping]
    logger.info(
        "Extracted %d company-to-problem mappings; %d known companies",
        len(company_to_problem_data.mapping),
        len(known_companies),
    )
    logger.debug(
        "Full company-to-problem mapping: %s", company_to_problem_data.model_dump_json(indent=2)
    )

    # STAGE 2: Process participants using the mapping and known companies
    logger.info("Stage 2: Extracting participants")
    participants_data = _extract_with_llm(
        client,
        participants_raw,
        "participants",
        ParticipantsData,
        config,
        known_companies=known_companies,
        company_to_problem_data=company_to_problem_data,
    )
    logger.info(
        "Extracted %d participants",
        len(participants_data.participants),
    )
    logger.debug("Full participants data: %s", participants_data.model_dump_json(indent=2))

    # STAGE 3: Process judges using the known company list and mapping
    logger.info("Stage 3: Extracting judges")
    judges_data = _extract_with_llm(
        client,
        judges_raw,
        "judges",
        JudgesData,
        config,
        known_companies=known_companies,
        company_to_problem_data=company_to_problem_data,
    )
    logger.info("Extracted %d judges", len(judges_data.judges))
    logger.debug("Full judges data: %s", judges_data.model_dump_json(indent=2))

    # Validate that we have data
    if not participants_data.participants:
        logger.error("No participants were extracted from the input")
        raise ValueError("No participants were extracted from the input")
    if not judges_data.judges:
        logger.error("No judges were extracted from the input")
        raise ValueError("No judges were extracted from the input")

    # Save processed files
    participants_out_path = os.path.join(processed_dir, "participants.json")
    judges_out_path = os.path.join(processed_dir, "judges.json")
    logger.info(
        "Saving processed files: judges=%s, participants=%s",
        judges_out_path,
        participants_out_path,
    )

    with open(participants_out_path, "w", encoding="utf-8") as f:
        json.dump(participants_data.model_dump(), f, indent=2)

    with open(judges_out_path, "w", encoding="utf-8") as f:
        json.dump(judges_data.model_dump(), f, indent=2)

    # Return the paths to the processed files
    logger.info("Preprocessing completed successfully")
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
        "--company-to-problem",
        required=True,
        help="Path to company-to-problem raw text or JSON file",
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config = PreprocessingConfig(model=args.model, temperature=args.temperature)

    try:
        # Ensure directories exist
        _ensure_dirs("data/inputs", args.processed_dir)

        # Process files
        judges_out_path, participants_out_path = preprocess_files(
            args.judges,
            args.participants,
            args.company_to_problem,
            processed_dir=args.processed_dir,
            config=config,
        )

        # Output result paths as JSON for easy consumption
        result = {
            "judges_path": judges_out_path,
            "participants_path": participants_out_path,
            "status": "success",
        }
        print(json.dumps(result, indent=2))

    except Exception as e:
        logger.exception("Preprocessing failed")
        # Output error information
        error_result = {"error": str(e), "status": "failed"}
        print(json.dumps(error_result, indent=2))
        exit(1)


if __name__ == "__main__":
    main()
