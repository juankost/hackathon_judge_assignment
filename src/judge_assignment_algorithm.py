"""
Judge Assignment Algorithm for Participant Evaluation

Adds auto-preprocessing of input files: if loading JSON fails, tries to
preprocess from raw paragraphs using `src.preprocess_data.preprocess_files`.

Also writes results to `data/outputs/assignment_results.json`.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
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

from src.utils import _ensure_data_dirs, _resolve_path, _project_root, _to_json_compatible


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
    participant_id: str
    problem_id: str
    problem: str
    full_name: Optional[str] = None
    sponsoring_company: Optional[str] = None

    def __repr__(self):
        return f"Participant({self.participant_id}, problem={self.problem_id})"

    def display_name(self) -> str:
        """Return name if available, otherwise ID."""
        return self.full_name if self.full_name else self.participant_id


@dataclass
class Judge:
    judge_id: str
    judge_type: JudgeType
    full_name: Optional[str] = None
    affiliation: Optional[str] = None  # company or university they are associated with
    problem_id: Optional[str] = None  # For specialized judges, this is their assigned problem
    problem: Optional[str] = None  # For specialized judges, this is their assigned problem

    def __repr__(self):
        return f"Judge({self.judge_id}, type={self.judge_type}, problem={self.problem})"

    def __hash__(self):
        return hash(self.judge_id)

    def __eq__(self, other):
        return isinstance(other, Judge) and self.judge_id == other.judge_id

    def display_name(self) -> str:
        """Return name if available, otherwise ID."""
        return self.full_name if self.full_name else self.judge_id


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


def load_preprocessed_participant_data(filename: str) -> List[Participant]:
    """Load participants from a JSON file."""
    from src.preprocess_data import ParticipantsData

    with open(filename, "r") as f:
        data = json.load(f)
    participants_data = ParticipantsData(**data)

    # Since Participant and ParticipantInfo have the same fields, use **vars()
    return [
        Participant(**participant.model_dump()) for participant in participants_data.participants
    ]


def load_preprocessed_judge_data(filename: str) -> Tuple[List[Judge], List[Judge]]:
    """Load judges from a JSON file and separate into specialized and flexible."""
    from src.preprocess_data import JudgesData

    with open(filename, "r") as f:
        data = json.load(f)
    judges_data = JudgesData(**data)

    specialized_judges: List[Judge] = []
    flexible_judges: List[Judge] = []

    for judge in judges_data.judges:
        judge_id = judge.judge_id
        full_name = judge.full_name
        affiliation = judge.company
        judge_type = JudgeType.FLEXIBLE if affiliation is None else JudgeType.SPECIALIZED
        problem = judge.problem
        problem_id = judge.problem_id

        if judge_type == JudgeType.FLEXIBLE:
            flexible_judges.append(
                Judge(
                    judge_id=judge_id,
                    judge_type=JudgeType.FLEXIBLE,
                    full_name=full_name,
                    affiliation=affiliation,
                    problem_id=problem_id,
                    problem=problem,
                )
            )
        else:
            specialized_judges.append(
                Judge(
                    judge_id=judge_id,
                    judge_type=JudgeType.SPECIALIZED,
                    full_name=full_name,
                    affiliation=affiliation,
                    problem_id=problem_id,
                    problem=problem,
                )
            )

    return specialized_judges, flexible_judges


def _load_or_preprocess_data(
    participants_path: str, judges_path: str, company_to_problem_path: Optional[str]
) -> Tuple[List[Participant], List[Judge], List[Judge]]:
    """Load or preprocess the data."""
    try:
        logging.info(f"Attempting to load participants from {participants_path}")
        participants = load_preprocessed_participant_data(participants_path)
        logging.info(f"Attempting to load judges from {judges_path}")
        specialized_judges, flexible_judges = load_preprocessed_judge_data(judges_path)
    except Exception:
        logging.info("Input files not in expected JSON format. Preprocessing...")

        from src.preprocess_data import preprocess_files

        if not company_to_problem_path:
            raise ValueError(
                "company_to_problem input path is required when preprocessing raw inputs"
            )

        processed_dir = os.path.join(_project_root(), "data/processed")
        processed_judges_path, processed_participants_path = preprocess_files(
            judges_input_path=judges_path,
            participants_input_path=participants_path,
            company_to_problem_input_path=company_to_problem_path,
            processed_dir=processed_dir,
        )

        # Load the processed data
        logging.info(f"Loading participants from {processed_participants_path}")
        participants = load_preprocessed_participant_data(processed_participants_path)
        logging.info(f"Loading judges from {processed_judges_path}")
        specialized_judges, flexible_judges = load_preprocessed_judge_data(processed_judges_path)

    return participants, specialized_judges, flexible_judges


def _save_results_markdown(groups: List[ParticipantGroup], output_path: str) -> None:
    """Save a human-readable Markdown summary of room assignments."""
    lines: List[str] = []
    for i, group in enumerate(groups, start=1):
        lines.append(f"Room {i}:")
        lines.append("- Judges:")
        for judge in group.assigned_judges:
            judge_name = judge.display_name()
            if judge.affiliation and judge.problem:
                lines.append(f"  - {judge_name} ({judge.affiliation} / {judge.problem})")
            else:
                lines.append(f"  - {judge_name}  // not affiliated with a sponsoring company")
        lines.append("")
        lines.append("- Participants:")
        for participant in group.participants:
            display_name = participant.display_name()
            lines.append(f"  - {display_name} ({participant.problem_id})")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(
        description="Judge Assignment Algorithm - Assigns judges to participant groups"
    )
    parser.add_argument("--judges", type=str, help="Path to judges JSON or raw file")
    parser.add_argument("--participants", type=str, help="Path to participants JSON or raw file")
    parser.add_argument(
        "--company-to-problem",
        type=str,
        help="Path to company-to-problem JSON or raw file (required if preprocessing raw inputs)",
    )
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
    parser.add_argument("--config", type=str, help="Path to JSON file with algorithm configuration")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")
    parser.add_argument(
        "--example",
        action="store_true",
        help="Use bundled example data from the example directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to write outputs (default: data/outputs under project root)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    args = parser.parse_args()
    if not args.example and (not args.judges or not args.participants):
        parser.error("--judges and --participants are required unless using --example")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )

    # Ensure data directories exist
    _ensure_data_dirs()

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
    if args.example:
        base = _project_root()
        logging.info("Root directory: %s", base)
        judges_path = os.path.join(base, "example", "judges_info.txt")
        participants_path = os.path.join(base, "example", "participants_info.txt")
        company_to_problem_path = os.path.join(base, "example", "company_to_problem.txt")
    else:
        participants_path = _resolve_path(args.participants)
        judges_path = _resolve_path(args.judges)
        company_to_problem_path = (
            _resolve_path(args.company_to_problem) if args.company_to_problem else None
        )

    # Load or preprocess the data
    participants, specialized_judges, flexible_judges = _load_or_preprocess_data(
        participants_path, judges_path, company_to_problem_path
    )

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
    # Run the algorithm
    groups, stats = algorithm.solve(use_optimization=not args.no_optimization)

    # Save results
    output_dir = (
        _resolve_path(args.output_dir)
        if getattr(args, "output_dir", None)
        else os.path.join(_project_root(), "data/outputs")
    )
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "assignment_results.json")
    with open(results_path, "w") as f:
        json.dump({"groups": _to_json_compatible(groups), "stats": stats}, f, indent=2)
    logging.info(f"Saved results to {results_path}")

    # Save Markdown summary
    md_path = os.path.join(output_dir, "assignment_results.md")
    _save_results_markdown(groups, md_path)
    logging.info(f"Saved Markdown summary to {md_path}")

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


if __name__ == "__main__":
    main()
