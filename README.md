# Hackathon Judge Assignment

This project provides a robust solution for assigning judges to participant groups in a hackathon setting. The core of the solution is an algorithm that automates the assignment process, considering various constraints to ensure fair and effective evaluations.

## Key Features

- **Automated Judge Assignment**: Automatically assigns specialized and flexible judges to participant groups.
- **Constraint-Based Optimization**: Minimizes constraint violations to ensure high-quality judging assignments.
- **Flexible Configuration**: Allows customization of algorithm parameters to suit different hackathon needs.
- **Built-in Examples**: Includes predefined examples for testing and demonstration purposes.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `uv` package manager (recommended)
- Gemini API key (for LLM-powered preprocessing)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/hackathon-judge-assignment.git
   cd hackathon-judge-assignment
   ```

2. **Install dependencies**:

   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with your Gemini API key:

   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
   ```

   Or create the file manually with:

   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

   Get your API key from: https://ai.google.dev/

## Quickstart with Raw Text Inputs

The system can intelligently extract structured data from raw text descriptions of judges and participants using AI. Here's how to get started quickly:

### 1. Prepare your raw text files

Create text files describing your judges and participants in natural language:

**`data/inputs/judges_raw.txt`**:

```
Dr. Alan Turing - Expert in Algorithm Design and theoretical computer science
Prof. Donald Knuth - Specialist in Data Structures and algorithms
Dr. Yann LeCun - Machine Learning and deep learning expert
Tim Berners-Lee - Web Development pioneer
Dr. Flexible Judge 1 - Can judge any category, generalist background
Dr. Flexible Judge 2 - Experienced in multiple domains
```

**`data/inputs/participants_raw.txt`**:

```
Alice Johnson - Working on Algorithm Design project, intermediate level
Bob Smith - Algorithm Design category
Charlie Brown - Data Structures implementation
Diana Prince - Data Structures project
Eve Wilson - Machine Learning classifier
Frank Miller - ML model optimization
Grace Hopper - Algorithm Design compiler project
Henry Ford - Data Structures database
Iris Chang - Web Development frontend
Jack Ryan - Web application backend
```

### 2. Run the assignment algorithm

The system will automatically detect that your inputs are raw text and use AI to extract structured data:

```bash
python src/judge_assignment_algorithm.py \
  --judges data/inputs/judges_raw.txt \
  --participants data/inputs/participants_raw.txt \
  --group-size 5 \
  --seed 42
```

### 3. View results

The algorithm will:

1. Use Gemini AI to extract structured data from your text files
2. Save processed JSON files to `data/processed/`
3. Run the assignment algorithm
4. Save results to `data/outputs/assignment_results.json`
5. Display a summary of the assignments in the console

## Usage

Run the algorithm with example data:

```bash
python src/judge_assignment_algorithm.py --example
```

Run with your own JSON or raw text files (automatically uses AI for extraction):

```bash
python src/judge_assignment_algorithm.py \
  --judges data/inputs/judges_raw.txt \
  --participants data/inputs/participants_raw.txt \
  --group-size 5 \
  --seed 42
```

Manual preprocessing (if you want to process files separately):

```bash
python src/preprocess_data.py \
  --judges data/inputs/judges_raw.txt \
  --participants data/inputs/participants_raw.txt \
  --processed-dir data/processed
```

### Arguments

- `--participants`: Path to JSON or raw text file containing participant data
- `--judges`: Path to JSON or raw text file containing judge data
- `--batch-size`: The desired number of participants per group (default: 5)
- `--seed`: Random seed for reproducible results (optional)
- `--no-optimization`: Disables simulated annealing for faster but potentially less optimal results
- `--example`: Runs the algorithm with built-in example data
- `--config`: Path to a JSON file for custom algorithm configuration

Example `participants.json` and `judges.json` files are available in the `data/` directory.

### Running with example data

To see the algorithm in action with predefined data, run:

```bash
python src/judge_assignment_algorithm.py --example
```

## Running tests

To ensure the algorithm is working correctly, you can run the suite of unit tests:

```bash
python -m unittest discover .
```

This command will automatically discover and run all tests in the `tests/` directory.
