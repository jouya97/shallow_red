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
The [adversarial and exact-endgame report](reports/ADVERSARIAL_ENDGAME_2026-07-21.md)
documents the opponent that tries to make Shallow Red win, exact Syzygy
guidance, and the Modal execution environment.
The [overnight experiment report](reports/OVERNIGHT_EXPERIMENTS_2026-07-22.md)
adds an exact 2.8-million-state KBvKR reverse-objective solve, a predictive
frozen-policy adversary, deployment-matched rollout training, paired tactical
and tablebase screens, and the rejected speed/reliability ablations. None
displaced the selected v0.3 stalemate-aware policy.

## Development

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra ml --extra cloud
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

# Ask a reverse-search adversary to get itself checkmated by Shallow Red.
uv run worst-chess smoke --target stalemate-aware \
  --opponent selfish-reverse-stockfish --opponent-nodes 32 \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --stockfish /path/to/stockfish --search-top-k 12 \
  --pairs 50 --openings 50 --opening-plies 6 --max-plies 300

# Add exact standard-chess guidance in covered three-piece endgames.
./scripts/download_syzygy_3piece.sh artifacts/tablebases/syzygy-3
uv run python scripts/syzygy_pilot.py \
  --tablebase artifacts/tablebases/syzygy-3 --positions-per-class 20
uv run worst-chess smoke --target stalemate-aware --opponent random \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --tablebase artifacts/tablebases/syzygy-3

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
  --positions 1000 --rollouts 4 --rollout-plies 80 \
  --target-continuation stalemate-aware --target-top-k 4 \
  --device cpu --workers 8

# New checkpoints should align Black policy actions with mirrored observations.
uv run worst-chess train-ranked \
  --dataset artifacts/datasets/ranked-stockfish.jsonl \
            artifacts/datasets/ranked-random-reply.jsonl \
  --checkpoint artifacts/checkpoints/ranked-v03.pt \
  --perspective-actions --value-loss-weight 0 \
  --epochs 20 --channels 32 --residual-blocks 4
```

## Modal

The optional cloud environment provides isolated CPU jobs, a guarded 16 GiB
exact-retrograde mode, and single-L4 training while keeping datasets and
checkpoints in the persistent `shallow-red-artifacts` Volume. Authenticate once
with `modal setup` or `modal token set`, then verify the locked image without
allocating a GPU:

```bash
uv run --extra cloud modal run modal_app.py --mode smoke
uv run --extra cloud modal volume put shallow-red-artifacts \
  artifacts/checkpoints/model.pt /checkpoints/model.pt

# CPU rollout/evaluation command. Use paths below /artifacts in the container.
uv run --extra cloud modal run modal_app.py --mode cpu \
  --command 'rerank-rollouts --input /artifacts/datasets/input.jsonl \
    --output /artifacts/datasets/reranked.jsonl \
    --checkpoint /artifacts/checkpoints/model.pt --device cpu'

# This is the only mode that allocates the configured L4.
uv run --extra cloud modal run modal_app.py --mode gpu \
  --command 'train-ranked ... --device cuda'

# Project locally, then run the closed KBvKR reverse-objective solve.
uv run python scripts/four_piece_retrograde.py KBvKR \
  --mode project --sample-size 50000 --seed 20260722
uv run --extra cloud modal run modal_app.py \
  --mode four-piece-retrograde \
  --command 'KBvKR --mode solve --sample-size 50000 \
    --seed 20260722 --maximum-ram-gib 16'
```

Modal functions scale to zero after the command exits. The CPU and GPU modes
commit `/artifacts` to the Volume before returning.

Generated datasets, checkpoints, PGNs, and machine-readable match reports live
under `artifacts/` and are intentionally gitignored.
