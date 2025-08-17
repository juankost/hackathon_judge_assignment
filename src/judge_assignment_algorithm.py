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
import tqdm
from src.utils import _ensure_data_dirs, _resolve_path, _project_root, _to_json_compatible
import matplotlib.pyplot as plt


@dataclass
class AlgorithmConfig:
    """Configuration for the judge assignment algorithm."""

    INITIAL_TEMP: float = 150.0
    COOLING_RATE: float = 0.97
    ITERATIONS: int = 10000
    RANDOM_SEED: Optional[int] = None
    START_FROM_RANDOM: bool = True
    FORCE_MOVE_PROB: float = 0.30
    RANDOM_VIOLATION_PROB: float = 0.10
    REHEAT_INTERVAL: int = 1000
    REHEAT_FACTOR: float = 1.10


class JudgeType(Enum):
    SPECIALIZED = "specialized"
    FLEXIBLE = "flexible"


class ConstraintViolationType(Enum):
    NO_SPECIALIZED_JUDGE = "no_specialized_judge"
    EVEN_NUMBER_OF_JUDGES = "even_number_of_judges"
    INSUFFICIENT_JUDGES = "insufficient_judges"
    LESS_THAN_THREE_JUDGES = "less_than_three_judges"
    MINIMUM_GROUP_SIZE = "minimum_group_size"


DEFAULT_CONSTRAINT_COSTS: Dict[ConstraintViolationType, float] = {
    ConstraintViolationType.NO_SPECIALIZED_JUDGE: 4.0,
    ConstraintViolationType.EVEN_NUMBER_OF_JUDGES: 1.0,
    ConstraintViolationType.INSUFFICIENT_JUDGES: 5.0,
    ConstraintViolationType.LESS_THAN_THREE_JUDGES: 2.0,
    ConstraintViolationType.MINIMUM_GROUP_SIZE: 2.0,
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
    weight: float = 1.0


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
        min_group_size: int = 4,
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

        # Minimum participants per group (configurable via CLI)
        self.min_participants_per_group = min_group_size

        # Track annealing cost per iteration for logging/plotting
        self.annealing_cost_history: List[float] = []
        self.annealing_temperature_history: List[float] = []

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

    def random_grouping(self) -> List[List[Participant]]:
        """
        Create participant groups randomly using the precomputed target sizes.

        - Shuffles all participants
        - Randomizes the order of target group sizes
        - Partitions the shuffled participants accordingly
        - Any leftovers (due to rounding) are distributed round-robin
        """
        if not self.target_group_sizes:
            return []

        shuffled_participants = self.participants[:]
        random.shuffle(shuffled_participants)

        group_sizes = self.target_group_sizes[:]
        random.shuffle(group_sizes)

        groups: List[List[Participant]] = []
        idx = 0
        for size in group_sizes:
            if idx >= len(shuffled_participants):
                groups.append([])
                continue
            groups.append(shuffled_participants[idx : idx + size])
            idx += size

        leftovers = shuffled_participants[idx:]
        for i, p in enumerate(leftovers):
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
            specialized_for_problem = [j for j in available_specialized if j.problem_id == problem]

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

    def _evaluate_group_violations(
        self, group_participants: List[Participant], assigned_judges: List[Judge], group_id: int
    ) -> List[ConstraintViolation]:
        """Compute constraint violations for a group given its current judges."""
        violations: List[ConstraintViolation] = []
        problems_in_group = {p.problem_id for p in group_participants}
        covered_by_specialized = {
            j.problem_id
            for j in assigned_judges
            if j.judge_type == JudgeType.SPECIALIZED and j.problem_id in problems_in_group
        }

        # Missing specialized judge penalty scales with the number of affected participants
        for problem in problems_in_group:
            if problem not in covered_by_specialized:
                affected = sum(1 for p in group_participants if p.problem_id == problem)
                violations.append(
                    ConstraintViolation(
                        group_id=group_id,
                        violation_type=ConstraintViolationType.NO_SPECIALIZED_JUDGE,
                        details=f"No specialized judge for problem {problem} (affects {affected} participants)",
                        weight=float(affected),
                    )
                )

        if len(assigned_judges) < len(problems_in_group):
            violations.append(
                ConstraintViolation(
                    group_id=group_id,
                    violation_type=ConstraintViolationType.INSUFFICIENT_JUDGES,
                    details=f"Only {len(assigned_judges)} judges for {len(problems_in_group)} problems",
                )
            )

        # Less than three judges penalty scales with the deficit
        if len(assigned_judges) < 3:
            deficit = 3 - len(assigned_judges)
            violations.append(
                ConstraintViolation(
                    group_id=group_id,
                    violation_type=ConstraintViolationType.LESS_THAN_THREE_JUDGES,
                    details=f"Group has only {len(assigned_judges)} judges (minimum 3 required; deficit {deficit})",
                    weight=float(deficit),
                )
            )

        if len(assigned_judges) >= 3 and len(assigned_judges) % 2 == 0:
            violations.append(
                ConstraintViolation(
                    group_id=group_id,
                    violation_type=ConstraintViolationType.EVEN_NUMBER_OF_JUDGES,
                    details=f"Group has {len(assigned_judges)} judges (even number)",
                )
            )

        return violations

    def _group_violation_cost(self, violations: List[ConstraintViolation]) -> float:
        return sum(
            self.violation_costs.get(v.violation_type, 1.0) * getattr(v, "weight", 1.0)
            for v in violations
        )

    def _compute_group_cost(
        self,
        group_participants: List[Participant],
        assigned_judges: List[Judge],
        group_id: int,
    ) -> float:
        """Compute full group cost including minimum size penalty, without mutating group state."""
        violations = self._evaluate_group_violations(group_participants, assigned_judges, group_id)
        size_deficit = max(0, self.min_participants_per_group - len(group_participants))
        if size_deficit > 0:
            violations.append(
                ConstraintViolation(
                    group_id=group_id,
                    violation_type=ConstraintViolationType.MINIMUM_GROUP_SIZE,
                    details=f"Group has {len(group_participants)} participants (minimum {self.min_participants_per_group}; deficit {size_deficit})",
                    weight=float(size_deficit),
                )
            )
        return self._group_violation_cost(violations)

    def _build_participant_groups_with_balanced_judges(
        self, participant_groups: List[List[Participant]]
    ) -> List[ParticipantGroup]:
        """
        Deterministic greedy assignment:
        - Group participants greedily (already done by caller)
        - First assign specialized judges to cover problems in each group
        - Then assign flexible judges to balance total judges across groups
        """
        num_groups = len(participant_groups)
        groups: List[ParticipantGroup] = []

        # Clone pools so we do not mutate the originals stored on the class
        available_specialized = list(self.specialized_judges)
        available_flexible = list(self.flexible_judges)

        # Map problem -> available specialized judges (queue-like)
        spec_by_problem: Dict[str, List[Judge]] = defaultdict(list)
        for j in available_specialized:
            if j.problem_id:
                spec_by_problem[j.problem_id].append(j)

        # Initialize groups with specialized judges
        for gid, members in enumerate(participant_groups):
            problems_in_group = {p.problem_id for p in members}
            assigned: List[Judge] = []

            for problem in problems_in_group:
                bucket = spec_by_problem.get(problem, [])
                if bucket:
                    judge = bucket.pop(0)
                    assigned.append(judge)

            groups.append(
                ParticipantGroup(
                    group_id=gid,
                    participants=members,
                    assigned_judges=assigned,
                    problems_covered={j.problem for j in assigned if j.problem},
                )
            )

        # Compute per-group target sizes for judge counts (roughly equal)
        total_judges_available = len(available_specialized) + len(available_flexible)
        # We have already placed some specialized judges into groups; recompute remaining
        remaining_specialized = [j for bucket in spec_by_problem.values() for j in bucket]
        # For balancing, consider the entire pool: aim for equal counts eventually
        base = total_judges_available // num_groups if num_groups > 0 else 0
        rem = total_judges_available % num_groups if num_groups > 0 else 0
        target_per_group = [base + 1 if i < rem else base for i in range(num_groups)]

        # Distribute remaining specialized judges (those not matching any group's problems)
        # Rare, but include them to use the full pool
        for j in remaining_specialized:
            # pick group with smallest size so far
            groups.sort(key=lambda g: len(g.assigned_judges))
            groups[0].assigned_judges.append(j)

        # Distribute flexible judges to reach targets
        flex_queue = available_flexible[:]
        for idx, group in enumerate(groups):
            needed = max(0, target_per_group[idx] - len(group.assigned_judges))
            for _ in range(needed):
                if not flex_queue:
                    break
                group.assigned_judges.append(flex_queue.pop(0))

        # If flexible judges remain, round-robin to balance further
        gi = 0
        while flex_queue:
            groups[gi % num_groups].assigned_judges.append(flex_queue.pop(0))
            gi += 1

        # Finalize problems_covered and violations
        for g in groups:
            problems_in_group = {p.problem_id for p in g.participants}
            g.problems_covered = {
                j.problem
                for j in g.assigned_judges
                if j.judge_type == JudgeType.SPECIALIZED and j.problem_id in problems_in_group
            }
            g.constraint_violations = self._evaluate_group_violations(
                g.participants, g.assigned_judges, g.group_id
            )

        return groups

    def _build_random_groups(
        self, participant_groups: List[List[Participant]]
    ) -> List[ParticipantGroup]:
        """
        Build groups with participants fixed as provided and assign judges randomly.

        Judges from both specialized and flexible pools are shuffled and then
        assigned in a round-robin fashion to produce a random starting state.
        """
        num_groups = len(participant_groups)
        groups: List[ParticipantGroup] = []

        for gid, members in enumerate(participant_groups):
            groups.append(
                ParticipantGroup(
                    group_id=gid,
                    participants=members,
                    assigned_judges=[],
                    problems_covered=set(),
                )
            )

        shuffled_specialized = self.specialized_judges[:]
        shuffled_flexible = self.flexible_judges[:]
        random.shuffle(shuffled_specialized)
        random.shuffle(shuffled_flexible)

        combined_judges: List[Judge] = []
        i = j = 0
        while i < len(shuffled_specialized) or j < len(shuffled_flexible):
            take_specialized = random.choice([True, False])
            if take_specialized and i < len(shuffled_specialized):
                combined_judges.append(shuffled_specialized[i])
                i += 1
            elif j < len(shuffled_flexible):
                combined_judges.append(shuffled_flexible[j])
                j += 1
            elif i < len(shuffled_specialized):
                combined_judges.append(shuffled_specialized[i])
                i += 1

        if num_groups > 0:
            for idx, judge in enumerate(combined_judges):
                groups[idx % num_groups].assigned_judges.append(judge)

        for g in groups:
            problems_in_group = {p.problem_id for p in g.participants}
            g.problems_covered = {
                j.problem
                for j in g.assigned_judges
                if j.judge_type == JudgeType.SPECIALIZED and j.problem_id in problems_in_group
            }
            g.constraint_violations = self._evaluate_group_violations(
                g.participants, g.assigned_judges, g.group_id
            )

        return groups

    def _total_violation_cost(self, groups: List[ParticipantGroup]) -> float:
        return sum(self._group_violation_cost(g.constraint_violations) for g in groups)

    def _recompute_group(self, group: ParticipantGroup) -> None:
        problems_in_group = {p.problem_id for p in group.participants}
        group.problems_covered = {
            j.problem
            for j in group.assigned_judges
            if j.judge_type == JudgeType.SPECIALIZED and j.problem_id in problems_in_group
        }
        # Minimum group size violation scales by deficit
        size_deficit = max(0, self.min_participants_per_group - len(group.participants))
        group.constraint_violations = self._evaluate_group_violations(
            group.participants, group.assigned_judges, group.group_id
        )
        if size_deficit > 0:
            group.constraint_violations.append(
                ConstraintViolation(
                    group_id=group.group_id,
                    violation_type=ConstraintViolationType.MINIMUM_GROUP_SIZE,
                    details=f"Group has {len(group.participants)} participants (minimum {self.min_participants_per_group}; deficit {size_deficit})",
                    weight=float(size_deficit),
                )
            )

    def _pick_target_violation(
        self, groups: List[ParticipantGroup]
    ) -> Tuple[Optional[ParticipantGroup], Optional[ConstraintViolation]]:
        """Sample a violation across all groups proportionally to its cost."""
        if not groups:
            return None, None
        # Build weighted list of all violations across all groups
        weighted: List[Tuple[ParticipantGroup, ConstraintViolation, float]] = []
        for g in groups:
            for v in g.constraint_violations:
                base = self.violation_costs.get(v.violation_type, 1.0)
                weight = base * float(getattr(v, "weight", 1.0))
                if weight > 0:
                    weighted.append((g, v, weight))
        if not weighted:
            return (groups[0], None) if groups else (None, None)
        total_weight = sum(w for _, _, w in weighted)
        r = random.random() * total_weight
        acc = 0.0
        for g, v, w in weighted:
            acc += w
            if r <= acc:
                return g, v
        # Fallback (due to floating point): return last
        g, v, _ = weighted[-1]
        return g, v

    def _attempt_swap_to_fix_violation(
        self,
        groups: List[ParticipantGroup],
        target_group: ParticipantGroup,
        violation: ConstraintViolation,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Try a simple judge move/swap between groups to address a violation.

        Returns a tuple of indexes to revert if needed:
          ((g_from_idx, judge_idx_in_from), (g_to_idx, judge_idx_in_to or -1 for append))
        If no move made, return None.
        """
        if violation is None:
            return None

        # Helper to find a donor group (most judges) and a flexible judge index
        def find_donor_with_flexible(exclude_gid: int) -> Optional[Tuple[int, int]]:
            candidates = [g for g in groups if g.group_id != exclude_gid and g.assigned_judges]
            candidates.sort(key=lambda g: len(g.assigned_judges), reverse=True)
            for g in candidates:
                for idx, j in enumerate(g.assigned_judges):
                    if j.judge_type == JudgeType.FLEXIBLE:
                        return g.group_id, idx
            return None

        # NO_SPECIALIZED_JUDGE: try to bring a specialized judge for the missing problem
        if violation.violation_type == ConstraintViolationType.NO_SPECIALIZED_JUDGE:
            # Compute missing problems robustly from group composition
            problems_in_group = {p.problem_id for p in target_group.participants}
            covered_by_specialized = {
                j.problem_id
                for j in target_group.assigned_judges
                if j.judge_type == JudgeType.SPECIALIZED and j.problem_id in problems_in_group
            }
            missing_problems = list(problems_in_group - covered_by_specialized)
            if not missing_problems:
                return None
            # Prefer the missing problem that affects most participants
            counts = defaultdict(int)
            for p in target_group.participants:
                counts[p.problem_id] += 1
            missing_problems.sort(key=lambda pr: counts.get(pr, 0), reverse=True)
            missing_problem = missing_problems[0]

            # Evaluate all donor candidates and pick the best delta-cost
            best_choice = None  # (delta_cost, donor_group, donor_idx, swap_idx)
            # Candidate flexible index in target for swap (if any)
            recv_swap_idx = next(
                (
                    idx
                    for idx, j in enumerate(target_group.assigned_judges)
                    if j.judge_type == JudgeType.FLEXIBLE
                ),
                None,
            )
            for g in groups:
                if g.group_id == target_group.group_id:
                    continue
                for idx, j in enumerate(g.assigned_judges):
                    if j.judge_type == JudgeType.SPECIALIZED and j.problem_id == missing_problem:
                        # Simulate move (or swap if flexible available in target)
                        donor_participants = g.participants
                        donor_judges = list(g.assigned_judges)
                        target_participants = target_group.participants
                        target_judges = list(target_group.assigned_judges)

                        moved_judge = donor_judges.pop(idx)
                        if recv_swap_idx is not None:
                            # Swap with target flexible
                            swapped_out = target_judges[recv_swap_idx]
                            target_judges[recv_swap_idx] = moved_judge
                            donor_judges.append(swapped_out)
                        else:
                            target_judges.append(moved_judge)

                        donor_cost_new = self._compute_group_cost(
                            donor_participants, donor_judges, g.group_id
                        )
                        target_cost_new = self._compute_group_cost(
                            target_participants, target_judges, target_group.group_id
                        )
                        donor_cost_old = self._compute_group_cost(
                            donor_participants, g.assigned_judges, g.group_id
                        )
                        target_cost_old = self._compute_group_cost(
                            target_participants, target_group.assigned_judges, target_group.group_id
                        )
                        delta = (donor_cost_new + target_cost_new) - (
                            donor_cost_old + target_cost_old
                        )
                        if best_choice is None or delta < best_choice[0]:
                            best_choice = (delta, g, idx, recv_swap_idx)

            if best_choice is None:
                return None

            _, donor_group, donor_judge_idx, recv_swap_idx = best_choice
            # Execute best move
            judge = donor_group.assigned_judges.pop(donor_judge_idx)
            if recv_swap_idx is not None:
                target_group.assigned_judges[recv_swap_idx], judge = (
                    judge,
                    target_group.assigned_judges[recv_swap_idx],
                )
                donor_group.assigned_judges.append(judge)
            else:
                target_group.assigned_judges.append(judge)
            return ((donor_group.group_id, -1), (target_group.group_id, -1))

        # INSUFFICIENT_JUDGES or LESS_THAN_THREE_JUDGES: move a flexible judge from a donor (choose best by delta)
        if violation.violation_type in (
            ConstraintViolationType.INSUFFICIENT_JUDGES,
            ConstraintViolationType.LESS_THAN_THREE_JUDGES,
        ):
            best_choice = None  # (delta_cost, donor_group, donor_j_idx)
            for g in groups:
                if g.group_id == target_group.group_id:
                    continue
                for idx, j in enumerate(g.assigned_judges):
                    if j.judge_type != JudgeType.FLEXIBLE:
                        continue
                    donor_participants = g.participants
                    donor_judges = list(g.assigned_judges)
                    target_participants = target_group.participants
                    target_judges = list(target_group.assigned_judges)
                    moved = donor_judges.pop(idx)
                    target_judges.append(moved)

                    donor_cost_new = self._compute_group_cost(
                        donor_participants, donor_judges, g.group_id
                    )
                    target_cost_new = self._compute_group_cost(
                        target_participants, target_judges, target_group.group_id
                    )
                    donor_cost_old = self._compute_group_cost(
                        donor_participants, g.assigned_judges, g.group_id
                    )
                    target_cost_old = self._compute_group_cost(
                        target_participants, target_group.assigned_judges, target_group.group_id
                    )
                    delta = (donor_cost_new + target_cost_new) - (donor_cost_old + target_cost_old)
                    if best_choice is None or delta < best_choice[0]:
                        best_choice = (delta, g, idx)
            if best_choice is None:
                return None
            _, donor_group, donor_j_idx = best_choice
            judge = donor_group.assigned_judges.pop(donor_j_idx)
            target_group.assigned_judges.append(judge)
            return ((donor_group.group_id, -1), (target_group.group_id, -1))

        # EVEN_NUMBER_OF_JUDGES: move one flexible judge to flip parity (choose best by delta)
        if violation.violation_type == ConstraintViolationType.EVEN_NUMBER_OF_JUDGES:
            best_choice = None
            for g in groups:
                if g.group_id == target_group.group_id:
                    continue
                for idx, j in enumerate(g.assigned_judges):
                    if j.judge_type != JudgeType.FLEXIBLE:
                        continue
                    donor_participants = g.participants
                    donor_judges = list(g.assigned_judges)
                    target_participants = target_group.participants
                    target_judges = list(target_group.assigned_judges)
                    moved = donor_judges.pop(idx)
                    target_judges.append(moved)

                    donor_cost_new = self._compute_group_cost(
                        donor_participants, donor_judges, g.group_id
                    )
                    target_cost_new = self._compute_group_cost(
                        target_participants, target_judges, target_group.group_id
                    )
                    donor_cost_old = self._compute_group_cost(
                        donor_participants, g.assigned_judges, g.group_id
                    )
                    target_cost_old = self._compute_group_cost(
                        target_participants, target_group.assigned_judges, target_group.group_id
                    )
                    delta = (donor_cost_new + target_cost_new) - (donor_cost_old + target_cost_old)
                    if best_choice is None or delta < best_choice[0]:
                        best_choice = (delta, g, idx)
            if best_choice is None:
                return None
            _, donor_group, donor_j_idx = best_choice
            judge = donor_group.assigned_judges.pop(donor_j_idx)
            target_group.assigned_judges.append(judge)
            return ((donor_group.group_id, -1), (target_group.group_id, -1))

        return None

    def _attempt_participant_move(
        self,
        groups: List[ParticipantGroup],
        target_group: ParticipantGroup,
        violation: ConstraintViolation,
    ) -> bool:
        """
        Try moving a participant between groups to reduce violation cost, with relaxed room sizes.
        - Maintain at least `self.min_participants_per_group` participants in any source group.
        - Allow destination groups to exceed the nominal room size.
        Strategies:
          1) NO_SPECIALIZED_JUDGE: move a participant of the missing problem to a group
             that already has a specialized judge for that problem.
          2) INSUFFICIENT_JUDGES: reduce the number of unique problems in target group by
             moving a participant from a singleton problem to another group.
        Returns True if a move was made.
        """
        if violation is None:
            return False

        # Helper: find destination groups that have a specialized judge for a problem
        def find_groups_with_specialized(
            problem_id: str, exclude_gid: int
        ) -> List[ParticipantGroup]:
            dests: List[ParticipantGroup] = []
            for g in groups:
                if g.group_id == exclude_gid:
                    continue
                for j in g.assigned_judges:
                    if j.judge_type == JudgeType.SPECIALIZED and j.problem_id == problem_id:
                        dests.append(g)
                        break
            return dests

        # Parse missing problem from details if present
        missing_problem: Optional[str] = None
        if violation.violation_type == ConstraintViolationType.NO_SPECIALIZED_JUDGE:
            details = violation.details
            if details and "problem" in details:
                try:
                    missing_problem = details.split("problem", 1)[1].strip().split()[0]
                except Exception:
                    missing_problem = None

        # Strategy 1: address missing specialized coverage by moving participants (choose best by delta)
        if (
            violation.violation_type == ConstraintViolationType.NO_SPECIALIZED_JUDGE
            and missing_problem
        ):
            if len(target_group.participants) <= self.min_participants_per_group:
                return False
            candidate_idxs = [
                idx
                for idx, p in enumerate(target_group.participants)
                if p.problem_id == missing_problem
            ]
            if not candidate_idxs:
                return False
            dest_groups = find_groups_with_specialized(missing_problem, target_group.group_id)
            if not dest_groups:
                return False
            # Prefer destinations already containing that problem to avoid increasing unique problems
            dest_groups.sort(
                key=lambda g: -sum(1 for p in g.participants if p.problem_id == missing_problem)
            )
            best_choice = None  # (delta_cost, move_idx, dest_group)
            for move_idx in candidate_idxs:
                for dest in dest_groups:
                    # simulate move
                    src_ps = list(target_group.participants)
                    dst_ps = list(dest.participants)
                    participant = src_ps.pop(move_idx)
                    dst_ps.append(participant)
                    src_cost_new = self._compute_group_cost(
                        src_ps, target_group.assigned_judges, target_group.group_id
                    )
                    dst_cost_new = self._compute_group_cost(
                        dst_ps, dest.assigned_judges, dest.group_id
                    )
                    src_cost_old = self._compute_group_cost(
                        target_group.participants,
                        target_group.assigned_judges,
                        target_group.group_id,
                    )
                    dst_cost_old = self._compute_group_cost(
                        dest.participants, dest.assigned_judges, dest.group_id
                    )
                    delta = (src_cost_new + dst_cost_new) - (src_cost_old + dst_cost_old)
                    if best_choice is None or delta < best_choice[0]:
                        best_choice = (delta, move_idx, dest)
            if best_choice is None:
                return False
            _, move_idx, dest_group = best_choice
            participant = target_group.participants.pop(move_idx)
            dest_group.participants.append(participant)
            return True

        # Strategy 2: reduce required judges by reducing unique problems via moving singleton-problem participants (choose best by delta)
        if violation.violation_type == ConstraintViolationType.INSUFFICIENT_JUDGES:
            counts = defaultdict(int)
            for p in target_group.participants:
                counts[p.problem_id] += 1
            singleton_idxs = [
                idx for idx, p in enumerate(target_group.participants) if counts[p.problem_id] == 1
            ]
            if not singleton_idxs:
                return False
            if len(target_group.participants) <= self.min_participants_per_group:
                return False
            best_choice = None  # (delta_cost, move_idx, dest_group)
            for move_idx in singleton_idxs:
                prob = target_group.participants[move_idx].problem_id
                for dest in groups:
                    if dest.group_id == target_group.group_id:
                        continue
                    # Prefer destinations that already have this problem
                    has_prob = any(p.problem_id == prob for p in dest.participants)
                    src_ps = list(target_group.participants)
                    dst_ps = list(dest.participants)
                    participant = src_ps.pop(move_idx)
                    dst_ps.append(participant)
                    src_cost_new = self._compute_group_cost(
                        src_ps, target_group.assigned_judges, target_group.group_id
                    )
                    dst_cost_new = self._compute_group_cost(
                        dst_ps, dest.assigned_judges, dest.group_id
                    )
                    src_cost_old = self._compute_group_cost(
                        target_group.participants,
                        target_group.assigned_judges,
                        target_group.group_id,
                    )
                    dst_cost_old = self._compute_group_cost(
                        dest.participants, dest.assigned_judges, dest.group_id
                    )
                    delta = (src_cost_new + dst_cost_new) - (src_cost_old + dst_cost_old)
                    # bias toward destinations with existing problem
                    if has_prob:
                        delta -= 0.1
                    if best_choice is None or delta < best_choice[0]:
                        best_choice = (delta, move_idx, dest)
            if best_choice is None:
                return False
            _, move_idx, dest_group = best_choice
            participant = target_group.participants.pop(move_idx)
            dest_group.participants.append(participant)
            return True

        # Strategy 3: MINIMUM_GROUP_SIZE - move a participant in to satisfy minimum-size (choose best donor and participant by delta)
        if violation.violation_type == ConstraintViolationType.MINIMUM_GROUP_SIZE:
            # Find donors above minimum size
            donors = [
                g
                for g in groups
                if g.group_id != target_group.group_id
                and len(g.participants) > self.min_participants_per_group
            ]
            if not donors:
                return False
            best_choice = None  # (delta_cost, donor_group, move_idx)
            for donor in donors:
                for idx, part in enumerate(donor.participants):
                    src_ps = list(donor.participants)
                    dst_ps = list(target_group.participants)
                    moved = src_ps.pop(idx)
                    dst_ps.append(moved)
                    donor_new = self._compute_group_cost(
                        src_ps, donor.assigned_judges, donor.group_id
                    )
                    target_new = self._compute_group_cost(
                        dst_ps, target_group.assigned_judges, target_group.group_id
                    )
                    donor_old = self._compute_group_cost(
                        donor.participants, donor.assigned_judges, donor.group_id
                    )
                    target_old = self._compute_group_cost(
                        target_group.participants,
                        target_group.assigned_judges,
                        target_group.group_id,
                    )
                    delta = (donor_new + target_new) - (donor_old + target_old)
                    # prefer participants whose problem already exists in target (less unique problem growth)
                    if any(p.problem_id == moved.problem_id for p in target_group.participants):
                        delta -= 0.1
                    if best_choice is None or delta < best_choice[0]:
                        best_choice = (delta, donor, idx)
            if best_choice is None:
                return False
            _, donor, idx = best_choice
            participant = donor.participants.pop(idx)
            target_group.participants.append(participant)
            return True

        return False

    def _simulated_annealing_group_assignments(
        self, groups: List[ParticipantGroup]
    ) -> List[ParticipantGroup]:
        """
        Optimize assignments starting from the greedy solution to minimize total
        weighted violation cost. Attempts targeted fixes via BOTH:
        - Judge swaps/moves across groups
        - Participant moves across groups (respecting minimum group size)
        """
        # Initialize costs
        for g in groups:
            self._recompute_group(g)

        current_cost = self._total_violation_cost(groups)
        best_groups = deepcopy(groups)
        best_cost = current_cost
        temperature = self.config.INITIAL_TEMP

        # Initialize cost history (include starting cost at iteration 0)
        self.annealing_cost_history = [current_cost]
        self.annealing_temperature_history = [temperature]

        bar = tqdm.tqdm(range(self.config.ITERATIONS), desc="Simulated annealing")
        bar.set_postfix(
            cost=f"{current_cost:.2f}", best=f"{best_cost:.2f}", temp=f"{temperature:.2f}"
        )
        for it in bar:
            # Optionally pick a random violation to encourage exploration
            if random.random() < (self.config.RANDOM_VIOLATION_PROB or 0.0):
                # Build a flat list of all violations
                all_vs: List[Tuple[ParticipantGroup, ConstraintViolation]] = []
                for g in groups:
                    for v in g.constraint_violations:
                        all_vs.append((g, v))
                if all_vs:
                    target_group, violation = random.choice(all_vs)
                else:
                    target_group, violation = None, None
            else:
                target_group, violation = self._pick_target_violation(groups)
            if target_group is None or violation is None:
                break

            # Keep a copy of previous state for local revert on rejection
            previous_groups = deepcopy(groups)
            previous_cost = current_cost

            # Prefer participant move when it can directly address violations
            did_move_participant = self._attempt_participant_move(groups, target_group, violation)
            move = None
            if not did_move_participant:
                move = self._attempt_swap_to_fix_violation(groups, target_group, violation)
            if move is None and not did_move_participant:
                # No applicable move found, cool and record
                temperature *= self.config.COOLING_RATE
                self.annealing_cost_history.append(current_cost)
                self.annealing_temperature_history.append(temperature)
                bar.set_postfix(
                    cost=f"{current_cost:.2f}", best=f"{best_cost:.2f}", temp=f"{temperature:.2f}"
                )
                continue

            # After tentative move, recompute affected groups' violations
            # For simplicity, recompute all; groups are small in hackathon context
            for g in groups:
                self._recompute_group(g)
            new_cost = self._total_violation_cost(groups)

            delta = new_cost - current_cost
            accept = delta < 0 or (
                temperature > 1e-9 and random.random() < math.exp(-delta / temperature)
            )
            forced_accept = random.random() < (self.config.FORCE_MOVE_PROB or 0.0)
            if accept or forced_accept:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_groups = deepcopy(groups)
                    best_cost = current_cost
            else:
                # Revert to previous local state
                groups = previous_groups
                current_cost = previous_cost

            temperature *= self.config.COOLING_RATE

            # Record and show current metrics
            self.annealing_cost_history.append(current_cost)
            self.annealing_temperature_history.append(temperature)
            bar.set_postfix(
                cost=f"{current_cost:.2f}", best=f"{best_cost:.2f}", temp=f"{temperature:.2f}"
            )

            # Light reheating to avoid premature convergence
            if self.config.REHEAT_INTERVAL and self.config.REHEAT_FACTOR:
                if (it + 1) % int(self.config.REHEAT_INTERVAL) == 0:
                    temperature *= float(self.config.REHEAT_FACTOR)

        return best_groups

    def solve(self, use_optimization=True) -> Tuple[List[ParticipantGroup], Dict[str, Any]]:
        """
        Main solving method with optional optimization.

        Groups participants greedily and assigns judges greedily with balancing.
        Optionally runs simulated annealing over judge assignments to reduce
        weighted violation cost.
        """
        if self.config.START_FROM_RANDOM:
            # Random initialization path
            logging.info("Creating participant groups using random assignment...")
            participant_groups = self.random_grouping()
            logging.info("Assigning judges randomly across groups for initialization...")
            initial_groups = self._build_random_groups(participant_groups)
            final_groups = initial_groups
        else:
            # Greedy initialization path (default)
            logging.info("Creating participant groups using greedy algorithm...")
            participant_groups = self.greedy_grouping()

            logging.info(
                "Assigning judges greedily: specialized first, then flexible to balance counts..."
            )
            final_groups = self._build_participant_groups_with_balanced_judges(participant_groups)

        # Step 3: Optional optimization over judge assignments focusing on violation costs
        if use_optimization:
            logging.info(
                "Optimizing assignments (judge swaps + participant moves) using simulated annealing over violation costs..."
            )
            final_groups = self._simulated_annealing_group_assignments(final_groups)

        # Collect violations after finalization
        all_violations: List[ConstraintViolation] = []
        for g in final_groups:
            self._recompute_group(g)
            all_violations.extend(g.constraint_violations)

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
            lines.append(f"  - {display_name} ({participant.problem})")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def _save_violations_markdown(
    groups: List[ParticipantGroup],
    output_path: str,
    violation_costs: Dict[ConstraintViolationType, float],
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a Markdown report listing constraint violations per room.

    The report includes an optional summary section and then, for each room,
    a bullet list of violations with their details and individual costs.
    """
    lines: List[str] = []

    # Optional summary header
    if stats is not None:
        lines.append("Summary:")
        lines.append(f"- Total groups: {stats.get('total_groups', 0)}")
        lines.append(f"- Total violations: {stats.get('total_violations', 0)}")
        weighted = stats.get("weighted_violation_cost")
        if weighted is not None:
            lines.append(f"- Weighted violation cost: {weighted}")
        violations_by_type = stats.get("violations_by_type")
        if isinstance(violations_by_type, dict) and violations_by_type:
            lines.append("- Violations by type:")
            for vtype, count in violations_by_type.items():
                lines.append(f"  - {vtype}: {count}")
        lines.append("")

    # Per-room details
    for i, group in enumerate(groups, start=1):
        lines.append(f"Room {i}:")
        if not group.constraint_violations:
            lines.append("- No violations.")
            lines.append("")
            continue

        lines.append("- Violations:")
        for violation in group.constraint_violations:
            cost = violation_costs.get(violation.violation_type, 1.0)
            lines.append(
                f"  - {violation.violation_type.value}: {violation.details} (cost: {cost})"
            )
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
        "--min-group-size",
        type=int,
        default=4,
        help="Minimum participants per room (default: 4)",
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

    # Configure logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
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
        # judges_path = os.path.join(base, "example", "judges_info.txt")
        # participants_path = os.path.join(base, "example", "participants_info.txt")
        # company_to_problem_path = os.path.join(base, "example", "company_to_problem.txt")

        # Show directly the processed ones fromthe example data
        judges_path = os.path.join(base, "data", "processed", "judges.json")
        participants_path = os.path.join(base, "data", "processed", "participants.json")
        company_to_problem_path = None
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
        min_group_size=args.min_group_size,
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

    # Save Violations report
    violations_md_path = os.path.join(output_dir, "assignment_violations.md")
    _save_violations_markdown(
        groups=groups,
        output_path=violations_md_path,
        violation_costs=algorithm.violation_costs,
        stats=stats,
    )
    logging.info(f"Saved Violations report to {violations_md_path}")

    # Save simulated annealing cost plot if available
    try:
        if getattr(algorithm, "annealing_cost_history", None):
            iterations = list(range(len(algorithm.annealing_cost_history)))
            fig, ax_cost = plt.subplots(figsize=(8, 4.5))

            # Plot cost with log scale
            ax_cost.plot(
                iterations,
                algorithm.annealing_cost_history,
                color="#1f77b4",
                linewidth=2,
                label="Cost",
            )
            ax_cost.set_title("Simulated Annealing - Cost and Temperature")
            ax_cost.set_xlabel("Iteration")
            ax_cost.set_ylabel("Total violation cost (log scale)")
            ax_cost.set_yscale("log")
            ax_cost.grid(True, linestyle=":", alpha=0.5)

            # Plot temperature on twin y-axis if available
            if getattr(algorithm, "annealing_temperature_history", None):
                ax_temp = ax_cost.twinx()
                ax_temp.plot(
                    iterations,
                    algorithm.annealing_temperature_history,
                    color="#ff7f0e",
                    linewidth=1.5,
                    alpha=0.85,
                    label="Temperature",
                )
                ax_temp.set_ylabel("Temperature")

                # Combined legend for both axes
                lines = ax_cost.get_lines() + ax_temp.get_lines()
                labels = [l.get_label() for l in lines]
                ax_cost.legend(lines, labels, loc="upper right")

            fig.tight_layout()
            plot_path = os.path.join(output_dir, "annealing_cost.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logging.info(f"Saved annealing cost plot to {plot_path}")
    except Exception as e:
        logging.warning(f"Failed to save annealing cost plot: {e}")

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
