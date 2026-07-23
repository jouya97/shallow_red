# Dynamic proof-hybrid experiment — 2026-07-22

## Question

Can Shallow Red stop relying on neural generalization, memorize every known
forced-selfmate position exactly, and dynamically recover when a player chooses
an unexpected legal reply?

## Implementation

`ProofGuidedSelfmateAgent` applies three layers in strict order:

1. An exact-position book compiled from independently generated proof reports.
   Each stored principal variation is replayed and required to end with the
   designated side checkmated. When duplicate proofs exist, the shortest proof
   wins, followed by a deterministic UCI tie break.
2. A fresh bounded AND/OR proof search from positions outside the book. Only a
   `PROVEN` result can override the fallback; completed refutations and node
   budget exhaustion cannot masquerade as solutions.
3. The selected v0.3 neural policy when neither exact layer has a solution.

The book also creates the rule-preserving color mirror of every record, so the
same mechanism works when Shallow Red is Black. Every decision source is
counted, and repeated live-search decisions are cached.

Both the evaluation CLI and UCI engine expose the opt-in flags
`--proof-report`, `--selfmate-search-plies`, and
`--selfmate-search-nodes`. Defaults remain unchanged.

## Exact-book validation

The merged ancestry report contains 138 proven six-ply positions. Adding color
mirrors creates 276 exact book entries.

| Check | Result |
|---|---:|
| Proof-ranked positions | 276 |
| Exact book hits | 276/276 |
| Book move has proven rank one | 276/276 |

This eliminates the fine-tuned model's memorization failure: on these positions
the correct proven move is selected deterministically rather than with 53%
training-set accuracy.

## Unexpected-reply validation

For the first ten original proof roots, the experiment played the book move,
enumerated every legal opponent reply, and independently searched the resulting
target-to-move position with the four remaining plies and a 100,000-node
budget.

| Roots | Distinct legal replies | Proven continuations | Refuted | Unknown |
|---:|---:|---:|---:|---:|
| 10 | 61 | 61 | 0 | 0 |

Thus the agent is not merely replaying one memorized line. Once inside a proven
region, it can respond correctly to every tested legal deviation.

## Ordinary-game screen

The selected v0.3 checkpoint was evaluated with the 276-entry book and a
four-ply, 10,000-node live search on every target turn.

### Uniform-random opponent

Ten fixed-seed games, 200-ply limit:

| Metric | v0.3 | Proof hybrid |
|---|---:|---:|
| Losses | 4 | 4 |
| Draws | 1 | 1 |
| Wins | 0 | 0 |
| Truncations | 5 | 5 |
| Median loss ply | 50 | 50 |
| Book hits | — | 0 |
| Live proof hits | — | 0 |
| Completed live refutations | — | 718 |
| Live-search nodes | — | 641,607 |
| Wall time | 1.76 s | 129.46 s |

The game records were identical apart from the target agent name. A cheaper
two-ply version also had zero book or search hits, searched 28,818 nodes, and
took 7.62 seconds.

### Trying-to-lose opponent

Two fixed-seed games against `selfish-random-reply`, 160-ply limit:

| Metric | v0.3 | Proof hybrid |
|---|---:|---:|
| Losses | 0 | 0 |
| Draws | 0 | 0 |
| Wins | 2 | 2 |
| Book hits | — | 0 |
| Live proof hits | — | 0 |
| Completed live refutations | — | 62 |
| Live-search nodes | — | 100,410 |

These two games are a small behavior probe, not a rate estimate. They do show
that the hybrid preserved the fallback exactly when it could prove nothing.

## Decision

Keep the exact book and dynamic solver as an opt-in research capability. Do not
enable four-ply live search in the web engine and do not replace selected v0.3.

The mechanism works, including against varied replies, but the current composed
selfmate regions are too far from positions reached in normal games. Spending
more runtime proving that ordinary middlegames are not mate-in-two is not useful.

The next dataset should start from positions actually encountered against
trying-to-lose players. Counterfactual rollouts can identify better moves from
those positions; exact proof search should then expand and certify any short
selfmate regions discovered there. Those empirical positions can form a dynamic
gameplay book even when no universal forced selfmate is yet provable.
