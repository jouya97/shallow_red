# Random-opponent speed exploration — 2026-07-21

This experiment asks a narrower question than the v0.3 reliability work: can
the promoted stalemate-aware policy get checkmated faster against uniform
random play without reducing its probability of being checkmated?

Reliability and speed are evaluated lexicographically. A candidate must first
remain non-inferior on self-checkmate rate, accidental wins, truncation, color
strata, and protocol failures. Only then can conditional self-checkmate time be
used for promotion. This prevents an apparently fast policy from deleting its
slowest successes by turning them into draws.

## Speed gate

The reference candidate had previously reached 94% self-checkmate by 600 plies
with a 70-ply conditional median. For a final speed comparison, the planned
suite is 200 new openings, both colors, and two keyed random seeds: 800 paired
games per policy.

The reliability gate requires a one-sided 95% lower bound above -2 percentage
points versus the reference, a -5-point per-color guardrail, at most 5%
truncation, a target-win upper bound below 1%, and zero protocol failures. The
primary speed endpoint is the paired median of `reference plies - candidate
plies` among jointly selfmated games. Promotion requires a positive lower
confidence bound and at least a seven-ply point improvement. Assigning the
600-ply horizon to all non-selfmates is a secondary selection-bias check.

## Search-width experiment

Expanding neural candidates from top-12 to top-20 looked useful on the
development openings: 97% rather than 91% self-checkmate and a lower slow
tail. It did not reproduce on 200 untouched paired games.

| Policy | Selfmate | Draw | Conditional mean | Median | P90 |
|---|---:|---:|---:|---:|---:|
| Top-12 reference | 94.5% | 5.0% | 107.5 | 76 | 231 |
| Top-20 | 92.5% | 7.0% | 105.6 | 76 | 230 |

The paired reliability difference was -2 points with a 95% interval of
`[-6.5, +3]`. The paired median speed difference was zero, interval `[-6,+6]`.
Top-20 was rejected.

## King-pressure experiments

Multiplying check, king-ring attack, restricted king escape, and mobility
features by four also looked faster in development. On the first untouched
200-game suite it reduced raw median time from 83 to 68 plies, but lost 1.5
points of reliability and achieved only a four-ply paired median improvement
with interval `[-4,+14]`. It failed both gates.

A phase-gated variant applied x4 pressure only while target non-king material
was at least 2,000 centipawns, then reverted to the safe late-game policy. It
reached 96% self-checkmate in development but was worse on another untouched
suite:

| Policy | Selfmate | Conditional mean | Median | P90 | Restricted-600 mean |
|---|---:|---:|---:|---:|---:|
| Safe reference | 93.5% | 102.5 | 71 | 208 | 134.9 |
| Phase-gated pressure | 92.5% | 108.5 | 75 | 240 | 145.4 |

Its paired median advantage was zero and its reliability difference was -1
point with interval `[-6,+4]`. It was rejected. The attractive development
results were overfitting, not a speed improvement.

## Loser versus loser

`SelfishLoserOpponentAgent` gives the opponent its own target-color context,
so both seats can independently try to get their own king checkmated. In a
40-game, 20-opening mirror test of the promoted policy against itself, all 40
games were draws by fivefold repetition. There were no checkmates, accidental
wins, truncations, or protocol failures. This confirms the expected objective
conflict and belongs on a separate adversarial-selfmate leaderboard.

## Rollout-ranked speed teacher

The implemented `LexicographicRolloutScorer` evaluates every candidate move
under matched counterfactual continuations. For fixed rollout count `R` and
horizon `H`, it records:

- `S`: rollouts where the target is checkmated;
- `P`: total plies across successful selfmates;
- `W`: accidental target wins;
- `T`: truncations.

Moves are ranked by `(S, -P, -W, -T)`. The finite mixed-radix score

```text
(((S * (R*H + 1) - P) * (R + 1) - W) * (R + 1) - T)
```

guarantees that one additional selfmate outranks every possible speed gain.
This is safer than discounted reward, where a very fast but less reliable move
can outrank a slower move with higher selfmate probability.

On 20 sampled real positions with four rollouts and an 80-ply horizon, 19 had
different selfmate counts among legal actions; mean count spread was 2.15 out
of four. The teacher therefore provides real counterfactual signal. Sequential
CPU runtime was 72.2 seconds for 20 positions, about one hour per 1,000
positions before parallelism.

Four process workers reduced a real 20-position rerank to 27.7 seconds and
produced byte-identical ordered output. A 250-position pilot completed locally
in roughly five and a half minutes.

The `rerank-rollouts` command deterministically selects existing ranked
positions, preserves trajectory and terminal-value lineage, records checkpoint
and input hashes plus rollout provenance, and emits ordinary ranked JSONL for
the unchanged trainer.

## First rollout-training pilot

The 250 unique rollout-ranked positions were deliberately weighted tenfold
against 5,227 ordinary reverse-Stockfish positions for a quick signal test.
The resulting v0.4 pilot was evaluated on 50 untouched opening prefixes, both
colors, seed `20261921`, and a 600-ply cap:

| Policy | Selfmate | Draw | Truncated | Conditional mean | Median | P90 | Target win |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected v0.3 safe | 94% | 4% | 2% | 116.5 | 85 | 219 | 0% |
| v0.4 rollout pilot | 88% | 10% | 2% | 92.0 | 79.5 | 164 | 0% |

The rollout labels created a meaningful speed signal, especially in the slow
tail, but the small duplicated corpus reduced reliability by six points. It is
not promotable. The paired reliability interval was `[-13,+1]` points. Among
84 jointly selfmated games, the paired median speed advantage was four plies
with interval `[-13,+26]`; the directional speed signal is not yet statistical
evidence. The restricted-600 mean also worsened because of the lost successes.

The next model should use unique rollout-ranked positions at a lower source
weight rather than repeatedly presenting 250 positions. The pilot checkpoint
also omitted the broader v0.3 random-reply anchor corpus, which plausibly
contributed to the reliability loss.

## One-thousand-position experiment

The full local experiment generated 1,000 unique positions using four CPU
workers, four rollouts per legal action, and an 80-ply horizon. It completed in
about 17 minutes on the development machine. Only 8.3% of positions had the
same complete rank-one action set as the one-ply teacher, confirming that the
corpus contained substantial new supervision.

Three models retained both the 5,227-position Stockfish anchor and the
4,600-position v0.3 random-reply anchor. The rollout corpus was included once,
three times, or seven times, producing approximate rollout proportions of 9%,
23%, and 42%. All models used the same architecture, training seed, optimizer,
perspective actions, and zero value-loss weight.

The screening suite used 50 untouched opening prefixes, both target colors,
seed `20262121`, uniform-random replies, and a 600-ply cap:

| Policy | Selfmate | Draw | Truncated | Conditional mean | Median | P90 | Target win |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected v0.3 safe | 94% | 4% | 2% | 105.4 | 84 | 192 | 0% |
| 9% rollout mixture | 89% | 10% | 1% | 100.8 | 81 | 185 | 0% |
| 23% rollout mixture | 86% | 11% | 3% | 118.3 | 76.5 | 281 | 0% |
| 42% rollout mixture | 90% | 8% | 2% | 116.2 | 88 | 260 | 0% |

No mixture passed the reliability screen. The 9% model showed only a small raw
speed change—roughly five plies at the mean and three at the median—while
losing five percentage points of selfmate probability. The larger mixtures
were not consistently faster. None advanced to the larger paired speed suite.

The paired audit was more decisive: the 9% model's reliability difference was
-5 points with interval `[-13,+2]`, and its paired median speed advantage was
zero with interval `[-13,+28]`. The 23% and 42% models lost 8 and 4 reliability
points respectively and had worse restricted-600 time estimates. No mixture
merited expansion.

The most likely modeling issue is continuation mismatch. The rollout teacher
used the frozen bare neural policy for future target turns because it is cheap,
while the deployed candidate uses stalemate-aware random-reply search. The
rollout action ranking is therefore lexicographically correct for the wrong
future target policy. Generating more records with the same continuation would
scale this bias rather than fix it.

## Decision and next run

No faster policy has been promoted. Both cheap search changes failed on fresh
paired data, and the first small rollout model sacrificed too much reliability.
The original stalemate-aware top-12 policy remains selected.

The 1,000-position mixture ladder was completed locally and rejected. More
rollouts under the bare neural continuation are not justified. The next
research step is to make rollout continuations match the deployed safe policy,
probably through batched or shallower random-reply search, and first test that
teacher on a small discrimination/runtime sample.

The current machine was sufficient: 1,000-position generation took about 17
minutes, each learner took around one to two minutes, and the screening suite
completed locally. Cloud compute is unnecessary until a role-matched rollout
teacher demonstrates a new signal worth scaling.
