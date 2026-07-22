# Overnight loss-optimization experiments — 2026-07-22

## Scope and decision rule

These experiments searched for changes that make the selected v0.3
stalemate-aware policy lose more reliably and, only after preserving
reliability, more quickly. Reliability is lexicographically primary: a model
that has fewer successful self-checkmates cannot be promoted merely because
its remaining successful games are short. Target wins and protocol failures
are hard safety failures.

Paired comparisons align games by opening FEN, target color, and random seed.
Confidence intervals resample whole opening clusters so the two colors from
one opening remain together. Development screens are used to reject weak
ideas, not to make population-wide claims.

## Exact adversarial selfmate endgames

The custom retrograde solver uses the actual reverse objective rather than
ordinary-chess tablebase WDL. On target turns it existentially selects a
selfmate continuation and minimizes exact plies. On opponent turns every legal
reply must remain forced, and a resisting opponent maximizes the distance.
Cycles, stalemate, dead positions, and checkmating the opponent are failures.

The exhaustive three-piece solve found zero forced nonterminal selfmates in
KQvK, KRvK, KBvK, KNvK, and promotion-closed KPvK, regardless of which side
owned the extra piece. The largest family contained 165,676 legal states per
side-to-move/owner role.

The exact KBvKR four-piece solve then enumerated the complete closed material
family under D4 board symmetry:

- 2,827,104 legal canonical states;
- 38,726,352 unique directed edges;
- 816 states in which the White target is already checkmated;
- 816 total forced states, including those terminals;
- **zero forced nonterminal selfmates**;
- maximum forced distance zero;
- about 902 MiB peak observed resident memory on Modal.

Captures close through exact KBvK, KvKR, and KvK outcomes. The result omits
50/75-move counters and threefold/fivefold history, as those require historical
state. There is no castling or en passant in this material family. Within that
declared game, the result is exhaustive: a lone bishop against a rook cannot
force the rook side to checkmate it from any nonterminal position.

Reproduce the projection locally and the exact solve on Modal:

```bash
uv run python scripts/four_piece_retrograde.py KBvKR \
  --mode project --sample-size 50000 --seed 20260722

uv run --extra cloud modal run modal_app.py \
  --mode four-piece-retrograde \
  --command 'KBvKR --mode solve --sample-size 50000 --seed 20260722 --maximum-ram-gib 16'
```

## Opponents that resist or try to lose

Reverse Stockfish from the opponent's perspective was tested on 50 held-out
openings with both target colors, top-12 target search, 32 Stockfish nodes per
legal root, and a 300-ply cap. Shallow Red selfmated once, won zero times, drew
66 times by fivefold repetition, and reached the cap 33 times. The 100 games
are 50 opening clusters; with zero clusters containing a target win, the exact
two-sided 95% cluster-level upper bound is 7.11%, not the smaller and
anti-conservative game-level bound.

A new frozen-policy exploit opponent predicted Shallow Red's exact next move,
tried to expose its own king, shed its own material, and avoided mating the
target. In 40 games it allowed one Shallow Red selfmate at 174 plies, produced
39 cap-limited games, and still induced **zero Shallow Red wins** and zero
protocol failures. This opponent is effective at denying losses but supplied
no examples of how to trick the frozen target into winning, so the branch was
not scaled.

Symmetric loser-versus-loser play remains a different game: the earlier
40-game mirror test ended in 40 fivefold-repetition draws. Self-play is useful
only when roles are asymmetric and the opponent distribution is declared.

## Repetition penalty

An opt-in penalty for recreating a prior position was tested because the
resistant-opponent games were repetition-heavy. It did not generalize to the
ordinary random-opponent objective.

On a 40-game paired development screen, the unmodified policy selfmated in
37 games. Penalties of `1e8` and `1e10` each selfmated in 35, with one
truncation. Both had a five-point lower selfmate rate and a 17.7-ply worse
restricted-horizon mean than the baseline. A separate 100-game screen of a
`1e12` penalty fell from 96% to 94% selfmate despite a three-ply lower raw
conditional median. All repetition-penalty variants were rejected.

## Deployment-matched rollout supervision

The earlier 1,000-position speed corpus used a bare neural continuation while
the deployed engine uses stalemate-aware random-reply search. The new teacher
fixed that mismatch. It reranked 250 unique positions with two common-random
rollouts per legal move, an 80-ply horizon, and the deployed-style top-4
stalemate-aware continuation.

The corpus passed its quality checks:

- 250/250 records parsed with finite scores and unique FENs;
- 246/250 positions had more than one rank level;
- 227/250 had a unique rank-one action;
- mean rank levels per position: 14.96;
- mean number of tied best actions: 1.152.

A first model mixed those records once with 5,227 reverse-Stockfish and 4,600
random-reply anchor positions, for about 2.5% rollout data. On the identical
100-game schedule where v0.3 scored 96 selfmates, four draws, and no wins, this
model scored 94 selfmates, five draws, one truncation, and no wins. Among 90
joint successes it was 6 plies slower at the paired median and 19.3 plies
slower at the paired mean; the restricted-horizon mean was 26.6 plies worse.
It was rejected. This first run used a different learner seed, so it is a valid
candidate screen but not a clean data-source ablation.

A controlled rerun used the original v0.3 learner seed. On the 40-game
development screen it scored 38 selfmates and two draws versus the original's
37 selfmates and three draws. Its restricted-horizon mean improved by 30.7
plies, but among the 35 joint successes its paired median improvement was zero
with a 95% cluster interval spanning approximately -29 to +35 plies. The
predeclared speed gate required at least a five-ply point improvement before a
larger confirmation run, so this suggestive slow-tail result did not advance.

Doubling the new corpus to about 5% produced equal 37/40 selfmate reliability
on a development screen, but its paired median was five plies slower and one
game truncated. It also failed the speed gate. The matched labels are
informative, but these small supervised mixtures still do not improve the
deployed policy.

## Inference-time rollout search

An opt-in inference agent used the neural top four moves as roots and chose
among them with one 48-ply common-random rollout per move. This directly used
terminal selfmate outcomes, but future target turns used the frozen neural
policy to keep cost bounded.

The approximation failed badly on a paired 20-game screen. The baseline
selfmated in all 20 games. Rollout search selfmated in seven, drew eight,
truncated five, and won zero. Its seven joint successes looked 76 plies faster
at the paired median only because it discarded 13 slower successes. Assigning
the 600-ply cap to non-successes made it 288.2 plies worse on average. This is
another concrete example of conditional-speed selection bias, and the branch
was rejected.

A second prototype used a shallower two-target-turn expectimax: neural top-four
roots, two common-random opponent-reply samples, and the existing
stalemate-aware one-ply score at the next target turn. On a two-game local
timing probe it was about 5.6 times slower than the baseline. The baseline
truncated twice at 150 plies; the prototype produced one selfmate at 107 and
one stalemate at 104. That one-opening signal is positive but far too small to
justify the extra compute or cloud scaling; the implementation remains
experimental and opt-in.

## Immediate-mate tactical override

Another opt-in branch scanned every legal target move, outside the neural
top-12 shortlist, for a positive exact probability that a uniform-random reply
would immediately checkmate the target. It then took the largest probability.

The override was faster among joint successes but less reliable. On a paired
40-game screen it selfmated in 35 games versus 37 for the baseline, while draws
rose from two to four. The paired median improved by seven plies, but the
selfmate rate fell five points and the restricted-horizon mean worsened by
22.6 plies. Offering a mate is not enough when a random opponent can decline
it and leave a worse continuation. The any-positive override was rejected.

A second, dominance-safe mode overrides only when every legal opponent reply
immediately checkmates the target. It cannot worsen that position, but it did
not activate in the paired 40-game screen: all 40 move sequences were
identical to the baseline. The guard remains a harmless opt-in capability, not
an empirically demonstrated improvement.

## Standard Syzygy deployment

The three-piece Syzygy wrapper was also tested end to end, not just on
synthetic covered positions. On a paired 40-game random-opponent screen, the
wrapped and unwrapped policies produced exactly the same move sequence in all
40 games: 38 selfmates, one draw, one truncation, and no target wins. Ordinary
WDL guidance had no deployment effect on this sample. This does not negate the
six structural action improvements in the synthetic coverage pilot; it shows
that three-piece standard-chess WDL is too narrow or too infrequently decisive
to improve the current full-game random-opponent policy.

## Compute and verification

The Modal billing snapshot after all jobs stopped was approximately **$0.79**
across the two calendar-hour windows, including about $0.10 of L4 usage. No
cloud application remained active, and the run stayed far below the $30 cap.

Final repository verification passed:

- lockfile consistency: passed;
- Ruff: passed;
- strict mypy across 42 source files: passed;
- pytest: 239 passed, one Stockfish-environment integration skip;
- `git diff --check`: passed.

## Current selection

None of the completed branches has displaced the original v0.3
stalemate-aware top-12 policy. Negative results are retained with full reports
and paired comparisons so the same ideas are not selected later from an
attractive but biased raw median.
