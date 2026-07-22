from __future__ import annotations

import pytest

from worst_chess.objective.retrograde import solve_forced_selfmate


def test_retrograde_uses_target_min_and_resister_max_distances() -> None:
    # 0 success terminal; 1 failure terminal.
    # 2 target can reach success in one.
    # 3 opponent can choose states 0 or 2, so forced distance is 2.
    # 4 target can choose states 0 or 3, so chooses distance 1.
    successors = ((), (), (0,), (0, 2), (0, 3))
    target_turn = (True, False, True, False, True)

    solution = solve_forced_selfmate(successors, target_turn, {0})

    assert solution.forced_selfmate == (True, False, True, True, True)
    assert solution.plies == (0, None, 1, 2, 1)
    assert solution.forced_count == 4


def test_resister_escape_and_cycles_are_not_forced_selfmates() -> None:
    # State 2 is a resister node with one success and one failure choice.
    # States 3 and 4 form a cycle with no successful attractor.
    successors = ((), (), (0, 1), (4,), (3,))
    target_turn = (True, False, False, True, False)

    solution = solve_forced_selfmate(successors, target_turn, {0})

    assert solution.forced_selfmate == (True, False, False, False, False)
    assert solution.plies == (0, None, None, None, None)


def test_retrograde_rejects_malformed_graphs() -> None:
    with pytest.raises(ValueError, match="one entry"):
        solve_forced_selfmate(((),), (), {0})
    with pytest.raises(ValueError, match="out of range"):
        solve_forced_selfmate(((1,),), (True,), set())
    with pytest.raises(ValueError, match="duplicate"):
        solve_forced_selfmate(((0, 0),), (True,), set())
    with pytest.raises(ValueError, match="must not have successors"):
        solve_forced_selfmate(((0,),), (True,), {0})


def test_retrograde_reports_optional_checkpoints() -> None:
    checkpoints: list[tuple[str, int]] = []

    solve_forced_selfmate(
        ((), (0,)),
        (True, True),
        {0},
        checkpoint=lambda phase, completed: checkpoints.append((phase, completed)),
    )

    assert checkpoints == [("predecessors", 0)]
