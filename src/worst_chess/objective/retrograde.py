"""Exact finite-state retrograde solving for forced selfmate reachability."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrogradeSolution:
    """Forced-selfmate membership and exact optimal plies for every state."""

    forced_selfmate: tuple[bool, ...]
    plies: tuple[int | None, ...]

    @property
    def forced_count(self) -> int:
        return sum(self.forced_selfmate)


def solve_forced_selfmate(
    successors: Sequence[Sequence[int]],
    target_turn: Sequence[bool],
    successful_terminals: Collection[int],
    *,
    checkpoint: Callable[[str, int], None] | None = None,
) -> RetrogradeSolution:
    """Solve a finite reachability game with lexicographic exact distance.

    Successful terminal states have distance zero.  On the designated target's
    turn, one winning successor suffices and the target minimizes distance.  On
    the resisting opponent's turn, every successor must be winning and the
    opponent maximizes distance.  Unknown states after the attractor closes are
    draws or failures, including cycles from omitted history-dependent rules.
    """

    state_count = len(successors)
    if len(target_turn) != state_count:
        raise ValueError("target_turn must have one entry per state")
    if any(type(value) is not bool for value in target_turn):
        raise TypeError("target_turn entries must be booleans")
    successful = set(successful_terminals)
    if any(
        type(state) is not int or not 0 <= state < state_count
        for state in successful
    ):
        raise ValueError("successful terminal index is out of range")

    predecessors: list[list[int]] = [[] for _ in range(state_count)]
    remaining = [0] * state_count
    for state, children in enumerate(successors):
        if checkpoint is not None and state % 250_000 == 0:
            checkpoint("predecessors", state)
        remaining[state] = len(children)
        seen: set[int] = set()
        for child in children:
            if type(child) is not int or not 0 <= child < state_count:
                raise ValueError(f"successor index out of range for state {state}")
            if child in seen:
                raise ValueError(f"duplicate successor for state {state}")
            seen.add(child)
            predecessors[child].append(state)

    forced = [False] * state_count
    distance: list[int | None] = [None] * state_count
    opponent_max_child = [0] * state_count
    queue: list[tuple[int, int]] = []
    for state in sorted(successful):
        if successors[state]:
            raise ValueError("successful terminals must not have successors")
        forced[state] = True
        distance[state] = 0
        heapq.heappush(queue, (0, state))

    processed = 0
    while queue:
        child_distance, child = heapq.heappop(queue)
        if distance[child] != child_distance:
            continue
        for predecessor in predecessors[child]:
            if forced[predecessor]:
                continue
            if target_turn[predecessor]:
                # Heap order makes the first discovered winning child the
                # shortest one, which is the target's secondary objective.
                predecessor_distance = child_distance + 1
                forced[predecessor] = True
                distance[predecessor] = predecessor_distance
                heapq.heappush(queue, (predecessor_distance, predecessor))
            else:
                remaining[predecessor] -= 1
                opponent_max_child[predecessor] = max(
                    opponent_max_child[predecessor], child_distance
                )
                if remaining[predecessor] == 0 and successors[predecessor]:
                    predecessor_distance = opponent_max_child[predecessor] + 1
                    forced[predecessor] = True
                    distance[predecessor] = predecessor_distance
                    heapq.heappush(queue, (predecessor_distance, predecessor))
        processed += 1
        if checkpoint is not None and processed % 250_000 == 0:
            checkpoint("attractor", processed)

    return RetrogradeSolution(tuple(forced), tuple(distance))


__all__ = ["RetrogradeSolution", "solve_forced_selfmate"]
