"""
Judge Assignment Algorithm for Participant Evaluation

Adds auto-preprocessing of input files: if loading JSON fails, tries to
preprocess from raw paragraphs using `src.preprocess_data.preprocess_files`.

Also writes results to `data/outputs/assignment_results.json`.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import random
import math
from enum import Enum
from copy import deepcopy
import json
import argparse
import sys
import logging
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class AlgorithmConfig:
    """Configuration for the judge assignment algorithm."""

    INITIAL_TEMP: float = 100.0
    COOLING_RATE: float = 0.95
    ITERATIONS: int = 1000
    RANDOM_SEED: Optional[int] = None


class JudgeType(Enum):
    SPECIALIZED = "specialized"
    FLEXIBLE = "flexible"


class ConstraintViolationType(Enum):
    NO_SPECIALIZED_JUDGE = "no_specialized_judge"
    EVEN_NUMBER_OF_JUDGES = "even_number_of_judges"
    INSUFFICIENT_JUDGES = "insufficient_judges"
    LESS_THAN_THREE_JUDGES = "less_than_three_judges"


DEFAULT_CONSTRAINT_COSTS: Dict[ConstraintViolationType, float] = {
    ConstraintViolationType.NO_SPECIALIZED_JUDGE: 1.0,
    ConstraintViolationType.EVEN_NUMBER_OF_JUDGES: 1.0,
    ConstraintViolationType.INSUFFICIENT_JUDGES: 1.0,
    ConstraintViolationType.LESS_THAN_THREE_JUDGES: 1.0,
}


@dataclass
class Participant:
    id: str
    problem_id: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Participant({self.id}, problem={self.problem_id})"

    def display_name(self) -> str:
        """Return name if available, otherwise ID."""
        return self.name if self.name else self.id


@dataclass
class Judge:
    id: str
    judge_type: JudgeType
    problem: Optional[str]  # For specialized judges, this is their assigned problem
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Judge({self.id}, type={self.judge_type}, problem={self.problem})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Judge) and self.id == other.id

    def display_name(self) -> str:
        """Return name if available, otherwise ID."""
        return self.name if self.name else self.id


@dataclass
class ConstraintViolation:
    group_id: int
    violation_type: ConstraintViolationType
    details: str


@dataclass
class ParticipantGroup:
    group_id: int
    participants: List[Participant]
    assigned_judges: List[Judge]
    problems_covered: Set[str]
    constraint_violations: List[ConstraintViolation] = field(default_factory=list)

    def __repr__(self):
        participant_names = [p.display_name() for p in self.participants]
        judge_names = [j.display_name() for j in self.assigned_judges]
        return (
            f"Group(id={self.group_id}, participants={participant_names}, "
            f"judges={judge_names}, "
            f"problems={self.problems_covered})"
        )


class JudgeAssignmentAlgorithm:
    """
    Algorithm to assign specialized and flexible judges to groups of participants.

    This class encapsulates the logic for creating participant groups and assigning
    judges based on a set of constraints, aiming to minimize violations.
    """

    def __init__(
        self,
        participants: List[Participant],
        problems: Set[str],
        specialized_judges: List[Judge],
        flexible_judges: List[Judge],
        room_group_size: int,
        violation_costs: Optional[Dict[ConstraintViolationType, float]] = None,
        config: AlgorithmConfig = AlgorithmConfig(),
    ):
        self.participants = participants
        self.problems = problems
        self.specialized_judges = specialized_judges
        self.flexible_judges = flexible_judges
        self.room_group_size = room_group_size
        self.config = config

        # Set random seed for reproducibility if specified
        if config.RANDOM_SEED is not None:
            random.seed(config.RANDOM_SEED)
            logging.info(f"Random seed set to {config.RANDOM_SEED} for reproducibility")

        # Initialize violation costs
        self.violation_costs = violation_costs or DEFAULT_CONSTRAINT_COSTS

        # Calculate target group sizes for uniform distribution
        num_participants = len(self.participants)
        if num_participants > 0 and self.room_group_size > 0:
            num_groups = max(1, round(num_participants / self.room_group_size))
            base_size = num_participants // num_groups
            remainder = num_participants % num_groups
            self.target_group_sizes = [base_size + 1] * remainder + [base_size] * (
                num_groups - remainder
            )
        else:
            self.target_group_sizes = []

        # Create mappings for quick lookup
        self.problem_to_specialized_judges = defaultdict(list)
        for judge in specialized_judges:
            if judge.problem:
                self.problem_to_specialized_judges[judge.problem].append(judge)

        self.participant_by_problem = defaultdict(list)
        for participant in participants:
            self.participant_by_problem[participant.problem_id].append(participant)

    def calculate_grouping_cost(self, groups: List[List[Participant]]) -> float:
        """
        Calculate the total cost of a grouping.

        Lower cost is better. The cost function penalizes groups with diverse
        problems and problems lacking specialized judges.
        """
        total_cost = 0

        for group in groups:
            unique_problems = set(p.problem_id for p in group)

            # Base cost: number of unique problems squared (heavily penalize diversity)
            total_cost += len(unique_problems) ** 2

            # Penalty for problems without specialized judges
            for problem in unique_problems:
                if not self.problem_to_specialized_judges[problem]:
                    total_cost += 20

        return total_cost

    def simulated_annealing_grouping(self) -> List[List[Participant]]:
        """
        Use simulated annealing to find a better grouping of participants.

        This method avoids deep copying the groups in each iteration for performance
        by reverting swaps that are not accepted.
        """
        # Start with greedy solution
        current_groups = self.greedy_grouping()
        current_cost = self.calculate_grouping_cost(current_groups)

        best_groups = deepcopy(current_groups)
        best_cost = current_cost

        temperature = self.config.INITIAL_TEMP

        for _ in range(self.config.ITERATIONS):
            if len(current_groups) < 2:
                break  # Cannot swap

            # Create a neighbor solution by swapping two participants
            # Select two different, non-empty groups
            group1_idx, group2_idx = random.sample(range(len(current_groups)), 2)

            group1 = current_groups[group1_idx]
            group2 = current_groups[group2_idx]

            if not group1 or not group2:
                continue

            # Swap random participants between groups
            p1_idx = random.randint(0, len(group1) - 1)
            p2_idx = random.randint(0, len(group2) - 1)

            # Perform the swap
            group1[p1_idx], group2[p2_idx] = group2[p2_idx], group1[p1_idx]

            new_cost = self.calculate_grouping_cost(current_groups)

            # Accept or reject the new solution
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                # Move accepted, update current cost
                current_cost = new_cost

                if current_cost < best_cost:
                    best_groups = deepcopy(current_groups)
                    best_cost = current_cost
            else:
                # Move rejected, revert the swap
                group1[p1_idx], group2[p2_idx] = group2[p2_idx], group1[p1_idx]

            # Cool down
            temperature *= self.config.COOLING_RATE

        return best_groups

    def greedy_grouping(self) -> List[List[Participant]]:
        """
        Create groups using a greedy approach that tries to minimize
        the number of different problems in each group, with uniform group sizes.
        """
        remaining_participants = self.participants.copy()
        groups = []

        if not self.target_group_sizes:
            return []

        for group_size in sorted(self.target_group_sizes, reverse=True):
            if not remaining_participants:
                break

            if len(remaining_participants) <= group_size:
                if remaining_participants:  # Avoid creating an empty group
                    groups.append(remaining_participants)
                remaining_participants = []
                break

            # Start with the problem that has the most participants
            problem_counts = defaultdict(int)
            for p in remaining_participants:
                problem_counts[p.problem_id] += 1

            most_common_problem = max(problem_counts, key=problem_counts.get)

            # Build a group starting with participants from the most common problem
            current_group = []
            participants_to_remove = []

            # First, add participants from the most common problem
            for p in remaining_participants:
                if len(current_group) >= group_size:
                    break
                if p.problem_id == most_common_problem:
                    current_group.append(p)
                    participants_to_remove.append(p)

            # If we need more participants, add from other problems
            if len(current_group) < group_size:
                problems_in_group = set(p.problem_id for p in current_group)

                # First try to add more from existing problems
                for p in remaining_participants:
                    if len(current_group) >= group_size:
                        break
                    if p not in participants_to_remove and p.problem_id in problems_in_group:
                        current_group.append(p)
                        participants_to_remove.append(p)

                # Then add from new problems if needed
                for p in remaining_participants:
                    if len(current_group) >= group_size:
                        break
                    if p not in participants_to_remove:
                        current_group.append(p)
                        participants_to_remove.append(p)

            groups.append(current_group)
            for p in participants_to_remove:
                remaining_participants.remove(p)

        # If any participants are left (e.g., due to rounding), add them to existing groups
        if remaining_participants:
            # Distribute remaining participants to the smallest groups
            groups.sort(key=len)
            for i, p in enumerate(remaining_participants):
                groups[i % len(groups)].append(p)

        return groups

    def assign_judges_greedily(
        self,
        group: List[Participant],
        group_id: int,
        available_specialized: Set[Judge],
        available_flexible: Set[Judge],
    ) -> Tuple[List[Judge], List[ConstraintViolation]]:
        """
        Assign judges to a group using a greedy, rule-based heuristic.

        It prioritizes covering problems with specialized judges, then flexible
        judges, and finally ensures the group has at least 3 judges and an odd
        number of judges. This function modifies the available judge sets.
        """
        assigned_judges = []
        violations = []
        problems_in_group = set(p.problem_id for p in group)

        # Track which problems are covered by specialized judges
        covered_by_specialized = set()

        # First, try to assign specialized judges for each problem
        for problem in problems_in_group:
            specialized_for_problem = [j for j in available_specialized if j.problem == problem]

            if specialized_for_problem:
                # Pick the first available specialized judge
                judge = specialized_for_problem[0]
                assigned_judges.append(judge)
                available_specialized.remove(judge)
                covered_by_specialized.add(problem)
            else:
                # No specialized judge available for this problem
                violations.append(
                    ConstraintViolation(
                        group_id=group_id,
                        violation_type=ConstraintViolationType.NO_SPECIALIZED_JUDGE,
                        details=f"No specialized judge available for problem {problem}",
                    )
                )

        # Use flexible judges for uncovered problems
        uncovered_problems = problems_in_group - covered_by_specialized
        for _ in uncovered_problems:
            if available_flexible:
                judge = available_flexible.pop()
                assigned_judges.append(judge)

        # Check if we have enough judges
        if len(assigned_judges) < len(problems_in_group):
            violations.append(
                ConstraintViolation(
                    group_id=group_id,
                    violation_type=ConstraintViolationType.INSUFFICIENT_JUDGES,
                    details=f"Only {len(assigned_judges)} judges for {len(problems_in_group)} problems",
                )
            )

        # Ensure minimum of 3 judges
        while len(assigned_judges) < 3:
            if available_flexible:
                assigned_judges.append(available_flexible.pop())
            elif available_specialized:
                assigned_judges.append(available_specialized.pop())
            else:
                # Can't meet the minimum judges constraint
                violations.append(
                    ConstraintViolation(
                        group_id=group_id,
                        violation_type=ConstraintViolationType.LESS_THAN_THREE_JUDGES,
                        details=f"Group has only {len(assigned_judges)} judges (minimum 3 required)",
                    )
                )
                break

        # Ensure odd number of judges (if we have at least 3)
        current_judge_count = len(assigned_judges)
        if current_judge_count >= 3 and current_judge_count % 2 == 0:
            # Try to add one more judge
            if available_flexible:
                assigned_judges.append(available_flexible.pop())
            elif available_specialized:
                # As a last resort, use any available specialized judge
                assigned_judges.append(available_specialized.pop())
            else:
                # Can't fix the even number constraint
                violations.append(
                    ConstraintViolation(
                        group_id=group_id,
                        violation_type=ConstraintViolationType.EVEN_NUMBER_OF_JUDGES,
                        details=f"Group has {current_judge_count} judges (even number)",
                    )
                )

        return assigned_judges, violations

    def solve(self, use_optimization=True) -> Tuple[List[ParticipantGroup], Dict[str, Any]]:
        """
        Main solving method with optional optimization.

        Groups participants and then assigns judges to each group.
        """
        # Step 1: Create participant groups
        if use_optimization:
            logging.info("Optimizing participant grouping using simulated annealing...")
            participant_groups = self.simulated_annealing_grouping()
        else:
            logging.info("Creating participant groups using greedy algorithm...")
            participant_groups = self.greedy_grouping()

        # Step 2: Assign judges to each group
        available_specialized = set(self.specialized_judges)
        available_flexible = set(self.flexible_judges)

        final_groups: List[ParticipantGroup] = []
        all_violations: List[ConstraintViolation] = []

        for i, group_participants in enumerate(participant_groups):
            assigned_judges, violations = self.assign_judges_greedily(
                group_participants, i, available_specialized, available_flexible
            )

            problems_in_group = set(p.problem_id for p in group_participants)
            covered_problems = set()

            # Determine which problems are covered by a specialized judge.
            for judge in assigned_judges:
                if judge.judge_type == JudgeType.SPECIALIZED and judge.problem in problems_in_group:
                    covered_problems.add(judge.problem)

            group = ParticipantGroup(
                group_id=i,
                participants=group_participants,
                assigned_judges=assigned_judges,
                problems_covered=covered_problems,
                constraint_violations=violations,
            )

            final_groups.append(group)
            all_violations.extend(violations)

        # Calculate statistics
        stats: Dict[str, Any] = {
            "total_groups": len(final_groups),
            "total_participants": len(self.participants),
            "total_judges_used": sum(len(g.assigned_judges) for g in final_groups),
            "total_violations": len(all_violations),
            "violations_by_type": self._count_violations_by_type(all_violations),
            "weighted_violation_cost": self._calculate_weighted_violation_cost(all_violations),
            "average_problems_per_group": (
                sum(len(set(p.problem_id for p in g.participants)) for g in final_groups)
                / len(final_groups)
                if len(final_groups) > 0
                else 0
            ),
            "grouping_cost": self.calculate_grouping_cost(participant_groups),
        }

        return final_groups, stats

    def _count_violations_by_type(self, violations: List[ConstraintViolation]) -> Dict[str, int]:
        """Count violations by type."""
        counts = defaultdict(int)
        for violation in violations:
            counts[violation.violation_type.value] += 1
        return dict(counts)

    def _calculate_weighted_violation_cost(self, violations: List[ConstraintViolation]) -> float:
        """Calculate total weighted cost of all violations."""
        total_cost = 0.0
        for violation in violations:
            total_cost += self.violation_costs.get(violation.violation_type, 1.0)
        return total_cost


def load_participants_from_json(filename: str) -> List[Participant]:
    """Load participants from a JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)

    participants: List[Participant] = []
    for participant_id, info in data.items():
        problem = info.get("problem")
        if not problem:
            raise ValueError(f"Participant {participant_id} missing 'problem' field")

        # Extract name and other metadata
        name = info.get("name")
        metadata = {k: v for k, v in info.items() if k not in ["problem", "name"]}

        participants.append(
            Participant(id=participant_id, problem_id=problem, name=name, metadata=metadata)
        )

    return participants


def load_judges_from_json(filename: str) -> Tuple[List[Judge], List[Judge]]:
    """Load judges from a JSON file and separate into specialized and flexible."""
    with open(filename, "r") as f:
        data = json.load(f)

    specialized_judges: List[Judge] = []
    flexible_judges: List[Judge] = []

    for judge_id, info in data.items():
        problem = info.get("problem")
        name = info.get("name")
        metadata = {k: v for k, v in info.items() if k not in ["problem", "name"]}

        if problem is None:
            # Flexible judge
            flexible_judges.append(
                Judge(
                    id=judge_id,
                    judge_type=JudgeType.FLEXIBLE,
                    problem=None,
                    name=name,
                    metadata=metadata,
                )
            )
        else:
            # Specialized judge
            specialized_judges.append(
                Judge(
                    id=judge_id,
                    judge_type=JudgeType.SPECIALIZED,
                    problem=problem,
                    name=name,
                    metadata=metadata,
                )
            )

    return specialized_judges, flexible_judges


def _ensure_data_dirs() -> None:
    base_dir = _project_root()
    os.makedirs(os.path.join(base_dir, "data/inputs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/processed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/outputs"), exist_ok=True)


def _project_root() -> str:
    """Return absolute path to the project root (one level up from this file's directory)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _resolve_path(path: str) -> str:
    """Resolve relative paths to the project root if they don't exist as given."""
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    candidate = os.path.join(_project_root(), path)
    return candidate


def _try_auto_preprocess(judges_path: str, participants_path: str) -> Tuple[str, str]:
    """
    If provided paths fail to load as JSON in expected schema, attempt to preprocess
    using src.preprocess_data.preprocess_files and return the processed JSON paths.
    """
    # Resolve paths relative to project root if needed
    judges_path_resolved = _resolve_path(judges_path)
    participants_path_resolved = _resolve_path(participants_path)

    try:
        _ = load_participants_from_json(participants_path_resolved)
        _ = load_judges_from_json(judges_path_resolved)
        return judges_path_resolved, participants_path_resolved
    except Exception:
        # Fallback import so running as `python src/judge_assignment_algorithm.py` works
        try:
            from src.preprocess_data import preprocess_files
        except ModuleNotFoundError:
            from preprocess_data import preprocess_files  # type: ignore

        _ensure_data_dirs()
        processed_dir = os.path.join(_project_root(), "data/processed")
        processed_judges, processed_participants = preprocess_files(
            judges_input_path=judges_path_resolved,
            participants_input_path=participants_path_resolved,
            processed_dir=processed_dir,
        )
        return processed_judges, processed_participants


def _serialize_groups(groups: List[ParticipantGroup]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for g in groups:
        serialized.append(
            {
                "group_id": g.group_id,
                "participants": [asdict(p) for p in g.participants],
                "assigned_judges": [
                    {
                        "id": j.id,
                        "judge_type": j.judge_type.value,
                        "problem": j.problem,
                        "name": j.name,
                    }
                    for j in g.assigned_judges
                ],
                "problems_covered": sorted(list(g.problems_covered)),
                "constraint_violations": [
                    {
                        "group_id": v.group_id,
                        "violation_type": v.violation_type.value,
                        "details": v.details,
                    }
                    for v in g.constraint_violations
                ],
            }
        )
    return serialized


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(
        description="Judge Assignment Algorithm - Assigns judges to participant groups"
    )
    parser.add_argument("--judges", type=str, help="Path to judges JSON or raw file")
    parser.add_argument("--participants", type=str, help="Path to participants JSON or raw file")
    parser.add_argument(
        "--room-group-size",
        type=int,
        default=5,
        help="Size of each participant group (default: 5)",
    )
    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="Disable simulated annealing optimization",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run with example data instead of JSON files",
    )
    parser.add_argument("--config", type=str, help="Path to JSON file with algorithm configuration")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")

    args = parser.parse_args()

    _ensure_data_dirs()

    if args.example:
        # Run example mode
        run_examples()
        return

    if not args.judges or not args.participants:
        parser.error("--judges and --participants are required unless using --example")

    # Load algorithm config if provided
    config = AlgorithmConfig()
    if args.config:
        try:
            with open(_resolve_path(args.config), "r") as f:
                config_data = json.load(f)
                config = AlgorithmConfig(**config_data)
            logging.info(f"Loaded algorithm configuration from {args.config}")
        except Exception as e:
            logging.error(f"Error loading configuration file: {e}", exc_info=True)
            sys.exit(1)

    # Override with command-line seed if provided
    if args.seed is not None:
        config.RANDOM_SEED = args.seed

    # Attempt to load data; if it fails, preprocess automatically
    # Resolve provided input paths to be robust to current working directory
    participants_path = _resolve_path(args.participants)
    judges_path = _resolve_path(args.judges)

    try:
        logging.info(f"Attempting to load participants from {participants_path}")
        participants = load_participants_from_json(participants_path)
        logging.info(f"Attempting to load judges from {judges_path}")
        specialized_judges, flexible_judges = load_judges_from_json(judges_path)
    except Exception:
        logging.info("Input files not in expected JSON format. Preprocessing...")
        processed_judges, processed_participants = _try_auto_preprocess(
            judges_path, participants_path
        )
        logging.info(f"Loading participants from {processed_participants}")
        participants = load_participants_from_json(processed_participants)
        logging.info(f"Loading judges from {processed_judges}")
        specialized_judges, flexible_judges = load_judges_from_json(processed_judges)

    # Extract unique problems
    problems = set(p.problem_id for p in participants)

    # Create and run algorithm
    algorithm = JudgeAssignmentAlgorithm(
        participants,
        problems,
        specialized_judges,
        flexible_judges,
        args.room_group_size,
        config=config,
    )

    groups, stats = algorithm.solve(use_optimization=not args.no_optimization)

    # Save results
    results_path = os.path.join(_project_root(), "data/outputs", "assignment_results.json")
    with open(results_path, "w") as f:
        json.dump({"groups": _serialize_groups(groups), "stats": stats}, f, indent=2)
    logging.info(f"Saved results to {results_path}")

    # Print results summary to logs
    logging.info("=== Judge Assignment Solution ===")
    logging.info("\nProblem Setup:")
    logging.info(f"- Total participants: {len(participants)}")
    logging.info(f"- Total problems: {len(problems)}")
    logging.info(f"- Specialized judges: {len(specialized_judges)}")
    logging.info(f"- Flexible judges: {len(flexible_judges)}")
    logging.info(f"- Batch size: {args.room_group_size}")

    logging.info("\nSolution Statistics:")
    logging.info(f"- Total groups: {stats['total_groups']}")
    logging.info(f"- Total violations: {stats['total_violations']}")
    logging.info(f"- Weighted violation cost: {stats['weighted_violation_cost']:.2f}")
    logging.info(f"- Violations by type: {stats['violations_by_type']}")
    logging.info(f"- Average problems per group: {stats['average_problems_per_group']:.2f}")

    logging.info("\nDetailed Group Assignments:")
    for i, group in enumerate(groups):
        logging.info(f"\nGroup {i+1}:")
        logging.info(f"  Participants: {[p.display_name() for p in group.participants]}")
        logging.info(f"  Problems: {set(p.problem_id for p in group.participants)}")
        logging.info(f"  Judges: {[j.display_name() for j in group.assigned_judges]}")
        logging.info(f"  Judge count: {len(group.assigned_judges)}")

        if group.constraint_violations:
            logging.info("  Violations:")
            for violation in group.constraint_violations:
                cost = algorithm.violation_costs.get(violation.violation_type, 1.0)
                logging.warning(
                    f"    - {violation.violation_type.value}: {violation.details} (cost: {cost})"
                )


def create_challenging_example():
    """Create a more challenging example with insufficient judges."""
    # Create problems
    problems = {"P1", "P2", "P3", "P4", "P5"}

    # Create participants (20 total)
    participants: List[Participant] = []
    participant_distribution = {"P1": 6, "P2": 5, "P3": 4, "P4": 3, "P5": 2}

    participant_id = 1
    for problem, count in participant_distribution.items():
        for _ in range(count):
            participants.append(Participant(f"X{participant_id}", problem))
            participant_id += 1

    # Create specialized judges (fewer than ideal)
    specialized_judges = [
        Judge("Z1_1", JudgeType.SPECIALIZED, "P1"),
        Judge("Z1_2", JudgeType.SPECIALIZED, "P2"),
        Judge("Z1_3", JudgeType.SPECIALIZED, "P3"),
        # Note: No specialized judge for P4 and P5
    ]

    # Create flexible judges (limited number)
    flexible_judges = [
        Judge("Z2_1", JudgeType.FLEXIBLE, None),
        Judge("Z2_2", JudgeType.FLEXIBLE, None),
    ]

    room_group_size = 5

    return participants, problems, specialized_judges, flexible_judges, room_group_size


def create_example():
    """Create an example instance of the problem."""
    # Create problems
    problems = {"P1", "P2", "P3", "P4"}

    # Create participants
    participants = [
        Participant("X1", "P1"),
        Participant("X2", "P1"),
        Participant("X3", "P1"),
        Participant("X4", "P2"),
        Participant("X5", "P2"),
        Participant("X6", "P3"),
        Participant("X7", "P3"),
        Participant("X8", "P4"),
        Participant("X9", "P1"),
        Participant("X10", "P2"),
        Participant("X11", "P3"),
        Participant("X12", "P4"),
    ]

    # Create specialized judges
    specialized_judges = [
        Judge("Z1_1", JudgeType.SPECIALIZED, "P1"),
        Judge("Z1_2", JudgeType.SPECIALIZED, "P1"),
        Judge("Z1_3", JudgeType.SPECIALIZED, "P2"),
        Judge("Z1_4", JudgeType.SPECIALIZED, "P3"),
        Judge("Z1_5", JudgeType.SPECIALIZED, "P4"),
    ]

    # Create flexible judges
    flexible_judges = [
        Judge("Z2_1", JudgeType.FLEXIBLE, None),
        Judge("Z2_2", JudgeType.FLEXIBLE, None),
        Judge("Z2_3", JudgeType.FLEXIBLE, None),
    ]

    room_group_size = 4

    return participants, problems, specialized_judges, flexible_judges, room_group_size


def run_examples():
    """Run the built-in examples."""
    logging.info("=== EXAMPLE 1: Basic Example ===")
    (
        participants,
        problems,
        specialized_judges,
        flexible_judges,
        room_group_size,
    ) = create_example()

    algorithm = JudgeAssignmentAlgorithm(
        participants, problems, specialized_judges, flexible_judges, room_group_size
    )

    # Try without optimization
    logging.info("\n--- Without Optimization ---")
    _, stats = algorithm.solve(use_optimization=False)
    logging.info(f"Grouping cost: {stats['grouping_cost']}")
    logging.info(f"Total violations: {stats['total_violations']}")
    logging.info(f"Weighted violation cost: {stats['weighted_violation_cost']}")

    # Try with optimization
    logging.info("\n--- With Optimization ---")
    _, stats_opt = algorithm.solve(use_optimization=True)
    logging.info(f"Grouping cost: {stats_opt['grouping_cost']}")
    logging.info(f"Total violations: {stats_opt['total_violations']}")
    logging.info(f"Weighted violation cost: {stats_opt['weighted_violation_cost']}")
    logging.info(f"Average problems per group: {stats_opt['average_problems_per_group']:.2f}")

    logging.info("\n\n=== EXAMPLE 2: Challenging Example ===")
    (
        participants2,
        problems2,
        specialized_judges2,
        flexible_judges2,
        room_group_size2,
    ) = create_challenging_example()

    algorithm2 = JudgeAssignmentAlgorithm(
        participants2, problems2, specialized_judges2, flexible_judges2, room_group_size2
    )

    groups2, stats2 = algorithm2.solve(use_optimization=True)

    logging.info("\nProblem Setup:")
    logging.info(f"- Total participants: {len(participants2)}")
    logging.info(f"- Total problems: {len(problems2)}")
    logging.info(f"- Specialized judges: {len(specialized_judges2)}")
    logging.info(f"- Flexible judges: {len(flexible_judges2)}")
    logging.info(f"- Batch size: {room_group_size2}")

    logging.info("\nSolution Statistics:")
    logging.info(f"- Total groups: {stats2['total_groups']}")
    logging.info(f"- Total violations: {stats2['total_violations']}")
    logging.info(f"- Weighted violation cost: {stats2['weighted_violation_cost']:.2f}")
    logging.info(f"- Violations by type: {stats2['violations_by_type']}")
    logging.info(f"- Grouping cost: {stats2['grouping_cost']}")

    logging.info("\nDetailed Group Assignments:")
    for i, group in enumerate(groups2):
        logging.info(f"\nGroup {i+1}:")
        logging.info(f"  Participants: {[p.display_name() for p in group.participants]}")
        logging.info(f"  Problems: {set(p.problem_id for p in group.participants)}")
        logging.info(f"  Judges: {[j.display_name() for j in group.assigned_judges]}")
        logging.info(f"  Judge count: {len(group.assigned_judges)}")

        if group.constraint_violations:
            logging.info("  Violations:")
            for violation in group.constraint_violations:
                logging.warning(f"    - {violation.violation_type.value}: {violation.details}")


if __name__ == "__main__":
    main()
