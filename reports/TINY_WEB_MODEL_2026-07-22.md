# Tiny web policy investigation — 2026-07-22

## Decision

A 24-channel, three-residual-block policy is small enough to run in the
browser and close enough to the selected 32-channel, four-block policy to
justify integration work. It is **not promoted yet**: the dependency-free
TypeScript inference path now passes parity, but the hybrid still needs
fresh browser gameplay and mobile-device latency tests.

The candidate's deployable policy has 37,633 parameters. Per-tensor int8
quantization produces a 37.7 KB raw payload, or 32.6 KB with zlib compression.
The value head is not needed by the web shortlist and is excluded from these
deployment-size figures.

## Controlled training setup

All candidates used the selected v0.3 setup:

- datasets:
  - `artifacts/datasets/ranked-v02-stockfish-500.jsonl`
  - `artifacts/datasets/ranked-random-reply-random-250.jsonl`
- split and initialization seed: `20261021`
- 7,937 training, 967 validation, and 923 test positions
- perspective-aligned actions
- rank temperature 2
- policy-only objective (`value_loss_weight=0`)
- 20 epochs on Apple MPS

The command shape was:

```bash
uv run worst-chess train-ranked \
  --dataset artifacts/datasets/ranked-v02-stockfish-500.jsonl \
            artifacts/datasets/ranked-random-reply-random-250.jsonl \
  --checkpoint artifacts/checkpoints/tiny-web-24x3-seed-20261021.pt \
  --perspective-actions --value-loss-weight 0 \
  --epochs 20 --batch-size 128 --rank-temperature 2 \
  --channels 24 --residual-blocks 3 \
  --seed 20261021 --device mps
```

## Ranking, size, and latency

Policy size excludes the unused value head. Int8 size includes one float scale
per parameter tensor. CPU latency is a 500-forward, batch-one PyTorch
microbenchmark with one CPU thread on the development Mac; it is useful for
relative comparison, not as a browser latency claim.

| Model | Test rank-1 | Test MRR | Policy params | Int8 + zlib | CPU p50 |
|---|---:|---:|---:|---:|---:|
| Selected 32×4 | 29.9% | 0.465 | 82,473 | 70.6 KB | 0.308 ms |
| 16×2 | 27.8% | 0.441 | 13,561 | 11.9 KB | 0.133 ms |
| 16×3 | 26.5% | 0.435 | 18,201 | 16.5 KB | 0.173 ms |
| 24×2 | 28.7% | 0.458 | 27,217 | 23.9 KB | 0.154 ms |
| **24×3** | **30.2%** | **0.473** | **37,633** | **32.6 KB** | **0.205 ms** |

The 24×3 policy has 54% fewer policy parameters and a 54% smaller compressed
int8 payload than the selected model. Its measured CPU forward pass was 33%
faster.

Top-12 teacher-best recall was 64.1% for 24×3 and 63.9% for the selected model.
This is the relevant shortlist measure for the tested hybrid.

## Quantization check

Every 24×3 policy parameter tensor was symmetrically quantized to int8 with one
scale, then dequantized for the existing inference harness:

| Metric | Result |
|---|---:|
| Float/int8 top-1 agreement | 97.7% |
| Float/int8 top-12 set agreement | 92.4% |
| Quantized test rank-1 | 30.4% |
| Quantized test MRR | 0.474 |

Post-training quantization did not degrade held-out ranking metrics. Gameplay
below used the float checkpoint.

## Browser prototype

The branch includes a versioned policy-only artifact and dependency-free
TypeScript inference implementation:

- `web/public/tiny-policy-v1.bin`: 39,700 bytes, including a 37,633-byte int8
  tensor payload and self-describing JSON header
- SHA-256:
  `2a44f37b163b1d3b2f1f6ad42d1f7e176ac088b6d135370584eafdb610e629fa`
- eight deterministic Python fixtures covering both colors, castling,
  en passant, repetition, low material, and white/black promotions
- exact observation and legal-action-coordinate parity on all fixtures
- exact legal top-12 ordering on all fixtures
- maximum TypeScript-versus-quantized-PyTorch logit error:
  `0.0000038147`

The dependency-free TypeScript forward pass measured 6.24 ms p50 and 6.47 ms
p95 over 200 warm runs in Node on the development Mac. This measures the
portable scalar implementation, not PyTorch, and is a useful conservative
desktop baseline rather than a mobile-browser claim.

The exporter is deterministic and keeps the unused value head out of the
artifact. The browser decoder validates its version, tensor metadata, payload
size, architecture, action layout, and orientation before inference.

The site integration uses the neural policy only to shortlist twelve legal
moves. The existing tactical reply scorer chooses among those moves, preserving
its immediate checkmate, draw, and stalemate safeguards. A load or inference
failure automatically falls back to the original all-moves heuristic.

## Gameplay

The target was the neural top-12 shortlist followed by the frozen
stalemate-aware random-reply search. Every comparison reused the selected
model's openings, colors, seeds, and ply cap.

| Opponent | Games / cap | Selected 32×4 | Tiny 24×3 | Selected draws / unresolved | Tiny draws / unresolved |
|---|---:|---:|---:|---:|---:|
| Uniform random | 200 / 600 | 188 (94%) | 186 (93%) | 8 / 4 | 13 / 1 |
| Stockfish, 1,000 nodes | 100 / 300 | 100 (100%) | 100 (100%) | 0 / 0 | 0 / 0 |
| Weak portfolio | 100 / 300 | 94 (94%) | 96 (96%) | 6 / 0 | 4 / 0 |
| Stress portfolio | 100 / 300 | 96 (96%) | 92 (92%) | 4 / 0 | 8 / 0 |
| **Total** | **500** | **478 (95.6%)** | **474 (94.8%)** | **18 / 4** | **25 / 1** |

Neither model accidentally won or had a protocol failure in these 500 games.

On the 200-game random suite, the paired opening-cluster bootstrap estimated
the tiny model's self-checkmate-rate difference at -1 percentage point with a
95% interval of `[-6, +4]`. Among the 174 games where both models
self-checkmated, mean speed differed by only 0.3 ply, with a wide interval
`[-20.9, +22.3]`. The candidate is therefore close, but this sample does not
establish non-inferiority.

The 16×2 model was rejected for the hybrid: it reached only 89% self-checkmate
on the aligned first 100 random games, versus 95% for the selected model.
The 24×2 model improved to 93%, but remained slower and less reliable than
24×3.

## Recommended next step

Continue with, but do not yet promote, the 24×3 policy:

1. Run the exact quantized browser hybrid through fresh gameplay suites.
2. Benchmark the complete browser path on desktop and mobile-class CPUs.
3. Train at least two additional 24×3 seeds and run a fresh gameplay suite
   before promotion, because the architecture comparison currently uses one
   seed.

The branch now includes the model-backed hybrid and its safe fallback. The
historical gameplay table above still measures the equivalent Python hybrid;
fresh evaluation of the exact TypeScript path remains required.
