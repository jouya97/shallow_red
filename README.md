# Worst Chess Ever

An orthodox chess engine that follows every rule and optimizes for its own
checkmate.

See the [research and implementation plan](docs/PROJECT_PLAN.md).

The [CPU baseline report](reports/BASELINE_2026-07-21.md) validates the legal
match harness. The [v0.2 ranked-policy report](reports/V02_RANKED_2026-07-21.md)
documents three leakage-safe training seeds, an independent opponent
population, and a policy-guided engine that reaches its own checkmate in 71%
of games against random play and 100% against Stockfish tiers. It also records
the current 0% result against an opponent that actively avoids mating it.

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
```

Generated datasets, checkpoints, PGNs, and machine-readable match reports live
under `artifacts/` and are intentionally gitignored.
