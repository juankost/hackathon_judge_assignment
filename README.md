## Hackathon Judge Assignment

This repository assigns judges to participant groups for a hackathon, respecting practical constraints (coverage by specialized judges, minimum/odd judge counts per room) and optimizing group composition. Inputs can be raw text or preprocessed JSON; raw text is auto-processed via Gemini.

### Key features

- **Automated assignments**: Specialized and flexible judges are assigned to participant groups.
- **Constraint-aware**: Minimizes violations (e.g., missing specialized coverage, even judge counts).
- **Greedy grouping + annealing over violations**: Participants grouped greedily; a simulated annealing phase adjusts judge assignments and may move participants between rooms to reduce violation cost.
- **LLM preprocessing**: Optional AI extraction from raw text files into structured JSON.

## Installation

### Prerequisites

- Python 3.11+
- `uv` package manager (recommended) or `pip`
- Gemini API key (for LLM-powered preprocessing of raw inputs)

### Steps

1. Clone the repository

   ```bash
   git clone https://github.com/your-username/hackathon-judge-assignment.git
   cd hackathon-judge-assignment
   ```

2. Install dependencies

   ```bash
   uv sync
   ```

3. Configure environment

   Create a `.env` file with your Gemini API key:

   ```bash
   echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
   ```

   Get your API key from `https://ai.google.dev`.

## Algorithm

### Overview

1. **Grouping participants (greedy)**

   - `greedy_grouping`: Packs groups to minimize the number of distinct problems per group while aiming for target room sizes.

2. **Assigning judges (greedy + balanced)**

   - Prefer specialized judges that match problems in the group.
   - Use flexible judges to balance judge counts across groups and meet minimum/odd count requirements.

3. **Optimization (simulated annealing over violation costs)**
   - Starts from the greedy assignment.
   - At each step, picks the group with the highest weighted violation cost and targets its most expensive violation.
   - Tries a targeted fix via either a judge swap/move across groups or a participant move. Room sizes are relaxed during optimization; only the minimum group size (configurable, default 4) is enforced.
   - With a small probability (`RANDOM_VIOLATION_PROB`), samples a random violation to escape local minima.

### Constraints and penalties

Violations are tracked per group. Default weighted penalties are defined in `src/judge_assignment_algorithm.py` as `DEFAULT_CONSTRAINT_COSTS`:

- `NO_SPECIALIZED_JUDGE`: 4.0
- `EVEN_NUMBER_OF_JUDGES`: 1.0
- `INSUFFICIENT_JUDGES`: 5.0
- `LESS_THAN_THREE_JUDGES`: 2.0

Note: Penalties are currently constants in code. CLI does not expose overrides. To change them, edit `DEFAULT_CONSTRAINT_COSTS` in `src/judge_assignment_algorithm.py` or modify initialization in your own integration to pass a custom mapping.

### Configuration options

Annealing parameters are configurable via a JSON file passed to `--config` and/or CLI flags:

- `INITIAL_TEMP` (float, default 100.0)
- `COOLING_RATE` (float, default 0.95)
- `ITERATIONS` (int, default 1000)
- `RANDOM_SEED` (int | null)
- `RANDOM_VIOLATION_PROB` (float, default 0.1): Probability of sampling a random violation instead of the highest-cost one during annealing.

Example `config.json`:

```json
{
  "INITIAL_TEMP": 120.0,
  "COOLING_RATE": 0.97,
  "ITERATIONS": 1500,
  "RANDOM_SEED": 42,
  "RANDOM_VIOLATION_PROB": 0.1
}
```

Pass it with `--config path/to/config.json`. You can also set `--seed` directly to override the random seed.

## Usage

Inputs can be either preprocessed JSON or raw text. When using raw text, you must also provide a company-to-problem mapping file so the LLM can align problems and companies.

### CLI arguments

- `--judges` (str): Path to judges JSON or raw text file
- `--participants` (str): Path to participants JSON or raw text file
- `--company-to-problem` (str): Path to company→problem mapping (required when inputs are raw)
- `--room-group-size` (int, default 5): Target participants per room
- `--min-group-size` (int, default 4): Minimum participants per room enforced during optimization
- `--no-optimization` (flag): Disable simulated annealing (greedy only)
- `--config` (str): Path to JSON with annealing parameters
- `--seed` (int): Random seed (overrides config)
- `--example` (flag): Use bundled example text files automatically
- `--output-dir` (str): Directory for outputs (defaults to `data/outputs`)
- `--log-level` (str): One of `CRITICAL|ERROR|WARNING|INFO|DEBUG`

### Run on bundled example data (without --example)

```bash
python src/judge_assignment_algorithm.py \
  --judges example/judges_info.txt \
  --participants example/participants_info.txt \
  --company-to-problem example/company_to_problem.txt \
  --room-group-size 5 \
  --seed 42
```

### Run with the built-in example flag

```bash
python src/judge_assignment_algorithm.py --example --room-group-size 5 --seed 42
```

### Run on your own raw text files

```bash
python src/judge_assignment_algorithm.py \
  --judges <path_to_judges_file> \
  --participants <path_to_participants_file> \
  --company-to-problem <path_to_company_to_problem_file> \
  --output-dir <path_to_output_dir> \
  --room-group-size 5
```

## Outputs

After a run, results are written to the output directory (default `data/outputs`):

- `assignment_results.json`: Machine-readable assignments and statistics
- `assignment_results.md`: Human-readable room breakdown (judges and participants)
- `assignment_violations.md`: Human-readable violations report

Key stats printed to logs include totals, violations by type, weighted violation cost, and average problems per group.
