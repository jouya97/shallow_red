# Reverse-search distillation evaluation — 2026-07-21

This report records the first trained losing-chess policy and its fixed-suite
evaluation. It is a development result against an ordinary chess opponent, not
a claim that selfmate chess is solved.

## Model and data

- Teacher: Stockfish 18, selecting the conventionally worst legal root move
  from the designated loser's point of view
- Teacher budget: 64 nodes independently for every legal root move
- Dataset: 10,000 positions from 500 deterministic random trajectories
- Split: whole trajectories, 80% train / 10% validation / 10% test
- Observation: 21 perspective-normalized 8x8 planes
- Policy: masked 4,672-action AlphaZero-style encoding
- Network: 32-channel residual CNN with four residual blocks
- Optimizer objective: legal-move cross entropy for 20 epochs
- Training device: Apple MPS; training time: under one minute
- Seed: 20260721

The untrained model's validation top-1 was 4.5%. The trained checkpoint reached
14.4% validation top-1 and 18.2% on held-out trajectories. Exact teacher-move
accuracy is diagnostic; match outcome is the promotion metric.

## Match setup

- Opponent: Stockfish 18 at 1,000 nodes per move
- Positions: the same 100 deterministic six-ply opening prefixes used by the
  baseline report
- Pairing: every target played every position as White and Black
- Games per target: 200
- Draw claims: disabled; automatic orthodox draws remain active
- Maximum length: 300 plies
- Seed: 20260721

## Results

| Target | Self-checkmate | Draw | Accidental win | Failure | Median plies | Mean plies |
|---|---:|---:|---:|---:|---:|---:|
| Uniform legal random | 200/200 | 0 | 0 | 0 | 33.5 | 34.04 |
| Sacrifice heuristic | 200/200 | 0 | 0 | 0 | 29.0 | 29.79 |
| Distilled neural policy | 200/200 | 0 | 0 | 0 | 23.0 | 23.72 |
| Live reverse search, 64 nodes/root | 200/200 | 0 | 0 | 0 | 16.0 | 18.16 |

Paired 10,000-resample percentile bootstrap comparisons use identical opening,
color, and seed keys. Positive values mean the neural policy was mated sooner:

| Baseline | Neural advantage, mean plies | 95% interval | Faster / tie / slower |
|---|---:|---:|---:|
| Uniform legal random | 10.32 | [7.78, 12.87] | 139 / 10 / 51 |
| Sacrifice heuristic | 6.07 | [3.82, 8.33] | 122 / 6 / 72 |
| Live reverse search | -5.56 | [-7.59, -3.48] | 63 / 19 / 118 |

Color-specific mean plies:

| Target | As White | As Black |
|---|---:|---:|
| Uniform legal random | 34.82 | 33.26 |
| Sacrifice heuristic | 27.94 | 31.64 |
| Distilled neural policy | 20.00 | 27.44 |
| Live reverse search | 17.76 | 18.56 |

## Interpretation

The trained policy clears the supervised promotion gate: it obeys the legal
mask, has zero protocol failures, achieves the desired terminal result in every
held-out game, and beats both CPU baselines beyond bootstrap noise. Distilling
the stronger reverse-search teacher matters: a separate 20,000-position model
trained only on the hand-built heuristic did not significantly beat random on
mean plies.

The live reverse-search policy remains substantially stronger, establishing a
clear next target. The neural model's Black performance also trails its White
performance. Ranked teacher targets, on-policy trajectories, value training,
and shallow policy-guided search are more justified next experiments than
blindly increasing learner compute.

All raw JSON reports, PGNs, datasets, metrics, and checkpoints are generated
under `artifacts/` and intentionally excluded from version control.
