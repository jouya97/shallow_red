from worst_chess.verification import verify_action_roundtrips


def test_action_verifier_meets_requested_count() -> None:
    result = verify_action_roundtrips(1_000, seed=3)

    assert result.verified_transitions >= 1_000
    assert result.positions > 0
    assert result.elapsed_seconds > 0
    assert result.transitions_per_second > 0

