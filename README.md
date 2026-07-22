# Shallow Red

An orthodox chess engine that follows every rule and optimizes for its own
checkmate.

See the [research and implementation plan](docs/PROJECT_PLAN.md).

The [CPU baseline report](reports/BASELINE_2026-07-21.md) validates the legal
match harness. The [v0.2 ranked-policy report](reports/V02_RANKED_2026-07-21.md)
documents the first ranked model. The
[v0.3 weak-opponent report](reports/V03_WEAK_OPPONENTS_2026-07-21.md) adds a
weak/noisy opponent league, perspective-aligned policy actions, random-reply
training, and stalemate-aware search. The current candidate self-checkmates in
94% of a fresh 200-game uniform-random suite by 600 plies, with 4% draws, 2%
unresolved games, and no observed accidental wins. It also remains at 100%
against the tested Stockfish tier. These are population results, not a claim
that full adversarial selfmate chess is solved.
The [random-opponent speed report](reports/SPEED_RANDOM_2026-07-21.md)
documents rejected search shortcuts, loser-versus-loser behavior, and the
lexicographic rollout teacher for the next speed-aware model.

## Development

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra ml
uv run pytest
uv run ruff check .
uv run mypy src
```

Useful entry points:

```bash
# Verify the full legal action encoding.
uv run worst-chess verify-actions --transitions 1000000

# Evaluate the selected policy-guided engine.
uv run worst-chess smoke --target policy-guided \
  --checkpoint artifacts/checkpoints/ranked-v02g-seed-20260722.pt \
  --stockfish /path/to/stockfish --search-top-k 8 --target-nodes 64 \
  --opponent stockfish --opponent-nodes 1000 \
  --pairs 100 --openings 100 --opening-plies 6 --seed 20260821

# Stress the v0.3 candidate against an unhelpful random opponent.
uv run worst-chess smoke --target stalemate-aware --opponent random \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --search-top-k 12 --pairs 100 --openings 100 --opening-plies 6 \
  --max-plies 600 --seed 20261221

# Run the trained policy as a UCI engine for a chess GUI or controller.
uv run worst-chess uci \
  --checkpoint artifacts/checkpoints/ranked-v02g-seed-20260722.pt \
  --search-stockfish /path/to/stockfish \
  --search-top-k 8 --search-nodes 64 --device cpu

# Generate all-legal-move on-policy labels and train policy plus value.
uv run worst-chess generate-ranked \
  --checkpoint artifacts/checkpoints/reverse-stockfish-10k.pt \
  --opponent resistant --stockfish /path/to/stockfish \
  --teacher-nodes 64 --trajectories 500 --positions-per-trajectory 20 \
  --output artifacts/datasets/ranked-resistant.jsonl
uv run worst-chess train-ranked \
  --dataset artifacts/datasets/ranked-stockfish.jsonl \
            artifacts/datasets/ranked-resistant.jsonl \
  --checkpoint artifacts/checkpoints/ranked-v02.pt \
  --epochs 20 --channels 32 --residual-blocks 4

# Rerank positions using reliability-first, speed-second counterfactual rollouts.
uv run worst-chess rerank-rollouts \
  --input artifacts/datasets/ranked-random-reply.jsonl \
  --output artifacts/datasets/ranked-rollout-speed.jsonl \
  --checkpoint artifacts/checkpoints/ranked-v03.pt \
  --positions 1000 --rollouts 4 --rollout-plies 80 --device cpu --workers 8

# New checkpoints should align Black policy actions with mirrored observations.
uv run worst-chess train-ranked \
  --dataset artifacts/datasets/ranked-stockfish.jsonl \
            artifacts/datasets/ranked-random-reply.jsonl \
  --checkpoint artifacts/checkpoints/ranked-v03.pt \
  --perspective-actions --value-loss-weight 0 \
  --epochs 20 --channels 32 --residual-blocks 4
```

Generated datasets, checkpoints, PGNs, and machine-readable match reports live
under `artifacts/` and are intentionally gitignored.
