# Proof-corpus fine-tuning experiment — 2026-07-22

## Decision

All five proof-fine-tuned candidates are rejected. The selected research model
remains v0.3.

The strongest candidate memorized many training compositions but selected a
proven move in 0 of 32 positions from five unseen composition families. It also
reduced the frozen random-opponent loss rate from 94.0% to 92.5% and did not
force a single loss against opponents that were also trying to lose. No model
was promoted or deployed.

## Leakage-safe data

The exact six-ply proof corpus contains 276 color-balanced positions derived
from 36 original YACPDB compositions. Mirrored White/Black examples share one
trajectory group. The fixed split used seed `20260722`:

| Partition | Examples | Composition families |
|---|---:|---:|
| Train | 214 | 28 |
| Validation | 30 | 3 |
| Test | 32 | 5 |

The original v0.3 corpora were reconstructed with their original seed
`20261021`: 7,937 train, 967 validation, and 923 test positions. Fine-tuning
started strictly from the selected v0.3 checkpoint. Fixed validation and test
files prevented proof families from moving between partitions when training
weights changed.

## Baseline

Before fine-tuning, v0.3 never chose a proven forced-selfmate move in either
proof holdout:

| Dataset | Rank-one | Mean reciprocal rank |
|---|---:|---:|
| Proof validation | 0 / 30 | 0.500 |
| Proof test | 0 / 32 | 0.500 |
| Base test | 29.9% | 0.465 |

The proof MRR of 0.5 means the selected action consistently belonged to the
refuted rank rather than the proven rank.

## First wave: existing soft rank objective

Three five-epoch candidates used proof examples at one, four, or eight copies,
corresponding to approximately 2.6%, 9.7%, and 17.7% of the fine-tuning stream.
All used learning rate `1e-4`, rank temperature `2`, no value loss, and the same
seed.

| Candidate | Proof-train rank-one | Proof-val rank-one | Proof-test rank-one | Base-test rank-one |
|---|---:|---:|---:|---:|
| Weight 1 | 5.6% | 20.0% | 0.0% | 30.6% |
| Weight 4 | 6.5% | 26.7% | 0.0% | 30.6% |
| Weight 8 | 5.6% | 13.3% | 0.0% | 30.9% |

This exposed an objective mismatch. With temperature 2, a rank-two refuted
move receives `exp(-1/2)`, about 61%, of a rank-one proven move's target
weight. Dozens of refuted actions therefore dominate the probability target.
Repeating proof positions cannot fix that target distribution.

## Second wave: sharp rank objective

The weight-four and weight-eight mixtures were retrained for ten epochs with
rank temperature `0.25`. This sharply favors proven actions while keeping the
same base corpus and initialization.

| Candidate | Proof-train rank-one | Proof-val rank-one | Proof-test rank-one | Base-test rank-one |
|---|---:|---:|---:|---:|
| Sharp weight 4 | 24.3% | 6.7% | 0.0% | 32.0% |
| Sharp weight 8 | 53.3% | 6.7% | 0.0% | 32.6% |

The sharp weight-eight candidate clearly learned the training diagrams, but
the gain collapsed on new source families. Its lower proof-test loss did not
change any top-ranked decision. This is memorization, not a transferable
forced-selfmate policy.

## Behavioral safety gates

The sharp weight-eight model was the only candidate advanced to gameplay
screens.

### Frozen uniform-random suite

Both models used the same 100 opening prefixes, colors, seeds, 600-ply cap,
stalemate-aware search, and policy top-12 setting.

| Model | Losses | Draws | Truncations | Wins | Median loss plies |
|---|---:|---:|---:|---:|---:|
| Selected v0.3 | 188 | 8 | 4 | 0 | 70 |
| Sharp proof weight 8 | 185 | 13 | 2 | 0 | 73 |

The candidate lost 1.5 percentage points fewer games and was three plies
slower at the median. Its base-dataset rank-one improvement did not translate
into better game behavior.

### Trying-to-lose population

| Opponent | Losses | Draws | Truncations | Wins |
|---|---:|---:|---:|---:|
| Selfish random-reply | 0 | 20 | 0 | 0 |
| Selfish portfolio | 0 | 19 | 1 | 0 |

All completed games ended by fivefold repetition. Fine-tuning on compositions
did not bridge from the initial chess position into a forced-selfmate region.

## What this tells us

The proof labels are correct and learnable, but 28 training composition
families are too few and too structurally narrow for the compact CNN to infer a
general forced-selfmate concept. More epochs or more copies amplify
memorization. The next serious training attempt should wait for substantially
more independent source compositions and more game-like proof-bearing states.

A purpose-built loss should also treat the set of proven moves as the positive
policy mask, unknown moves as unlabeled, and refuted moves as negatives. Rank
temperature is an imperfect proxy. The most valuable research sequence is:

1. Expand the independently validated corpus to hundreds or thousands of
   source families, subject to corpus reuse permission.
2. Generate reachable/game-like proof ancestors, not only quiet local
   perturbations of compositions.
3. Add a proof-specific masked policy objective or auxiliary proof head.
4. Repeat family-held-out evaluation before spending on full gameplay gates.
5. Keep the random and trying-to-lose suites as mandatory promotion gates.

## Implementation and verification

`train-ranked` now supports strict `--initialize-from` fine-tuning and fixed
`--validation-dataset` / `--test-dataset` inputs. It validates checkpoint
architecture and policy-action orientation before training.

All training used Modal L4 jobs; gameplay gates used Modal CPU. Every Modal app
was stopped after completion. No checkpoint was promoted and the web model was
not changed.
