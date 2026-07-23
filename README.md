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
unresolved games, and no observed wins. It also remains at 100%
against the tested Stockfish tier. These are population results, not a claim
that full adversarial selfmate chess is solved.
The [frozen web-engine evaluation](reports/WEB_ENGINE_EVALUATION_2026-07-22.md)
runs the exact lightweight TypeScript policy on the same 300-game random suite.
It records 254 losses, 44 draws, two unresolved games, and zero wins; the web
policy is safe in this sample but does not match the research engine's loss rate.
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
The [forced-selfmate and adversarial-population report](reports/FORCED_SELFMATE_ADVERSARIAL_2026-07-22.md)
adds trying-to-lose opponent curricula, bounded s#1-s#4 proof search, a Popeye
prescreen, and two rejected adversarially trained models. The selected v0.3
policy remains unchanged.
The [selfmate corpus and proof-ancestry report](reports/SELFMATE_CORPUS_RETRO_2026-07-22.md)
imports and independently validates attributed compositions, creates exact
six-ply ancestors, and builds honest all-move proof labels. The
[proof fine-tuning report](reports/PROOF_FINETUNE_2026-07-22.md) evaluates two
fine-tuning objectives and rejects all five candidates for failing to
generalize to unseen composition families.
The [dynamic proof-hybrid report](reports/DYNAMIC_PROOF_HYBRID_2026-07-22.md)
adds a deterministic shortest-selfmate book and bounded live proof search. It
perfectly executes the known proofs and handles arbitrary replies inside those
proof regions, but the current regions were never reached in ordinary games,
so v0.3 remains selected.
The [synthetic loser-generation report](reports/SYNTHETIC_LOSER_GENERATION_2026-07-22.md)
builds a diverse anti-repetition opponent league, generates reachable games,
and discovers two new exact selfmates plus empirical steering moves up to six
plies earlier. The pipeline is promising, but the current positive yield is too
small and exploratory wins prevent training or deployment without more work.
The [prioritized synthetic-ancestry report](reports/SYNTHETIC_ANCESTRY_2026-07-22.md)
screens earlier moves from the decisive trajectories, confirms steering up to
ten plies before game end, and pairs the positives with earlier decisions from
the winning games as explicit safety data. Its scaled follow-up builds 258
all-legal labels from 73 reachable families, fixes a family-split leakage bug,
and rejects the resulting low-weight fine-tunes on clean held-out families.
The [branching selfmate-fuzzer report](reports/SELFMATE_FUZZER_2026-07-22.md)
expands those positions into variable legal trajectories, scales to 1,536
independent initial games, and proves 83 reachable positions across 54 loss
families. It also documents the safe-root correction and rejects two
reachable-proof fine-tunes that reduced random-opponent loss reliability.

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

# Stress against a population whose members also try to lose.
uv run worst-chess smoke --target stalemate-aware \
  --opponent selfish-portfolio \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --search-top-k 12 --pairs 20 --openings 20 --opening-plies 6

# Generate diverse initial-board trajectories with safe target exploration.
uv run worst-chess smoke --target stalemate-aware \
  --opponent synthetic-loser-league --target-exploration 0.20 \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --search-top-k 12 --pairs 25 --opening-plies 0 --max-plies 600

# Cheaply screen earlier decisions from synthetic losses and wins, then
# independently confirm only positions that produced at least one selfmate.
uv run python scripts/screen_synthetic_ancestry.py \
  --pgn artifacts/evaluations/synthetic-run/games.pgn \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --tail-target-positions 12 --rollouts 1 --rollout-plies 80 --workers 4 \
  --output artifacts/evaluations/synthetic-ancestry-screen/report.json
uv run python scripts/screen_synthetic_ancestry.py \
  --screen-report artifacts/evaluations/synthetic-ancestry-screen/report.json \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --rollouts 4 --rollout-plies 120 --workers 4 \
  --output artifacts/evaluations/synthetic-ancestry-confirm/report.json

# Combine confirmed steering positions with final winning moves as safety
# negatives, then replace the placeholder ranks with all-legal population
# rollouts before using the data for training.
uv run python scripts/build_synthetic_ancestry_dataset.py \
  --screen-report artifacts/evaluations/synthetic-ancestry-screen/report.json \
  --confirm-report artifacts/evaluations/synthetic-ancestry-confirm/report.json \
  --output artifacts/datasets/synthetic-ancestry-seeds.jsonl \
  --manifest artifacts/evaluations/synthetic-ancestry-dataset/manifest.json
uv run worst-chess rerank-rollouts \
  --input artifacts/datasets/synthetic-ancestry-seeds.jsonl \
  --output artifacts/datasets/synthetic-ancestry-ranked.jsonl \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --positions 14 --rollouts 4 --rollout-plies 120 \
  --target-continuation stalemate-aware --target-top-k 12 \
  --rollout-opponent synthetic-loser-league --device cpu --workers 4
uv run python scripts/finalize_synthetic_ancestry_dataset.py \
  --input artifacts/datasets/synthetic-ancestry-ranked.jsonl \
  --manifest artifacts/evaluations/synthetic-ancestry-dataset/manifest.json \
  --rollouts 4 --rollout-plies 120 \
  --output artifacts/datasets/synthetic-ancestry-final.jsonl \
  --report artifacts/evaluations/synthetic-ancestry-dataset/finalization.json

# Branch the finalized positions through varied synthetic loser opponents.
uv run python scripts/fuzz_selfmate_branches.py \
  --dataset artifacts/datasets/synthetic-ancestry-final.jsonl \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --generations 3 --beam-width 32 --branch-moves 3 \
  --samples-per-move 2 --segment-plies 12 --workers 4 \
  --output artifacts/evaluations/selfmate-fuzzer

# Create independent color-balanced midgame families from the initial array,
# then use the safety-first beam rather than recycling known ancestry.
uv run python scripts/generate_fuzzer_frontier.py \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --positions 64 --warmup-plies 40 --workers 4 \
  --output artifacts/evaluations/selfmate-fresh-seeds
uv run python scripts/fuzz_selfmate_branches.py \
  --frontier artifacts/evaluations/selfmate-fresh-seeds/frontier.jsonl \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --beam-objective safety-first --generations 3 --beam-width 64 \
  --branch-moves 3 --samples-per-move 3 --segment-plies 16 --workers 4 \
  --output artifacts/evaluations/selfmate-fresh-fuzz

# Generate ranked hard-negative states against the same population.
uv run worst-chess generate-ranked \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --teacher random-reply --target-policy stalemate-aware \
  --opponent selfish-portfolio --trajectories 100 \
  --output artifacts/datasets/ranked-adversarial.jsonl

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

# Extract reachable near-mate states and run auditable bounded proofs.
uv run python scripts/mine_forced_selfmate.py extract \
  --pgn artifacts/evaluations/run/games.pgn \
  --tail-target-positions 4 \
  --output artifacts/datasets/forced-selfmate-candidates.jsonl
uv run python scripts/mine_forced_selfmate.py search \
  --input artifacts/datasets/forced-selfmate-candidates.jsonl \
  --max-plies 2 4 6 8 --node-budget 200000 \
  --output artifacts/evaluations/forced-selfmate/report.json

# Optional fast prescreen; independently validate every positive above.
uv run python scripts/popeye_prescreen.py \
  --input artifacts/datasets/forced-selfmate-candidates.jsonl \
  --popeye /path/to/popeye/py --max-moves 4 \
  --output artifacts/evaluations/popeye-prescreen/report.json

# Import attributed orthodox compositions without copying published solutions.
uv run python scripts/import_yacpdb_selfmates.py \
  --query 'Stip("s#[1-2]") AND NOT Fairy' --pages 1 \
  --output artifacts/datasets/yacpdb-selfmates.jsonl

# Back up two quiet legal plies and prove the generated policy move.
uv run python scripts/expand_selfmate_ancestors.py \
  --proof-report artifacts/evaluations/yacpdb-proof/report.json \
  --max-candidates-per-seed 50 --node-budget 100000 \
  --output artifacts/evaluations/yacpdb-retro/report.json

# Score every legal move as proven, unknown, or refuted; mirror labels for Black.
uv run python scripts/build_proof_ranked_dataset.py \
  --proof-report artifacts/evaluations/yacpdb-retro/report.json \
  --output artifacts/datasets/proof-ranked.jsonl \
  --report artifacts/evaluations/proof-ranked/report.json

# Opt into exact book moves and bounded dynamic proof search during evaluation.
uv run worst-chess smoke --target neural --opponent random \
  --checkpoint artifacts/checkpoints/ranked-v03-perspective-random-seed-20261021.pt \
  --proof-report artifacts/evaluations/yacpdb-retro-modal/merged-report.json \
  --selfmate-search-plies 4 --selfmate-search-nodes 10000 \
  --pairs 5 --max-plies 200
```

The importer is intended for attributed research samples under `artifacts/`.
YACPDB does not currently advertise a bulk-content license, so do not vendor or
redistribute a database dump without clarifying permission.

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
