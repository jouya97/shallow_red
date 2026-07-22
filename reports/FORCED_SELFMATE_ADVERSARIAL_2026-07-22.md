# Forced-selfmate and adversarial-population experiments — 2026-07-22

## Outcome

The selected v0.3 stalemate-aware model remains the research candidate. Two
models trained on trying-to-lose opponent trajectories failed the promotion
criteria: one became less reliable against random play and won an adversarial
game, while the safer population mixture did not force any adversarial losses
and became slower on a matched random subset.

The useful result is new infrastructure and a sharper problem definition. A
trying-to-lose opponent population now supplies hard negative trajectories,
and a bounded AND/OR solver can certify genuine orthodox selfmates. The first
proof corpus contains only three forced positions, all selfmate-in-one. There
are not yet enough earlier steering positions to train a forced-selfmate
policy.

## Opponents that also try to lose

Two deterministic opponent modes were added:

- `selfish-random-reply` uses an exhaustive stalemate-aware losing policy from
  the opponent's own perspective.
- `selfish-portfolio` switches among a copy of Shallow Red, the exhaustive
  selfish policy, and an adversary that predicts and exploits the frozen target.

The frozen v0.3 target played 20 color-paired games against each mode on Modal:

| Opponent | Shallow Red losses | Draws | Truncations | Shallow Red wins |
|---|---:|---:|---:|---:|
| Selfish random-reply | 0 | 20 | 0 | 0 |
| Selfish portfolio | 0 | 18 | 2 | 0 |

All 38 completed games ended by fivefold repetition. This is a useful hard
negative distribution, but it also shows why expected performance against a
random or normally winning opponent cannot answer the adversarial question.

## Exact short selfmate search

`proof_search.py` implements target-existential/opponent-universal AND/OR
search. A result is `proven` only when one target move survives every legal
opponent response and eventually leaves the designated target checkmated.
Draws and target wins refute a branch. Exhausted node budgets report `unknown`,
not a false negative.

The candidate miner extracted the last four target turns from 283 successful
games, deduplicating them into 1,132 legal reachable positions. The full
selfmate-in-one/two sweep used a 20,000-node budget per horizon:

- 3 proven selfmate-in-one positions;
- 0 new selfmate-in-two positions;
- 1,129 positions completely refuted through selfmate-in-two;
- 0 unknown results.

The three certified lines are:

| Target | Forced line |
|---|---|
| Black | `f3+ Kd1#` |
| White | `fxg5+ Qxg5#` |
| White | `Qb6+ Qxb6#` |

A deeper 75-position Modal pilot used 200,000 nodes per horizon. All 75 were
completely refuted through selfmate-in-three. At selfmate-in-four, 64 were
refuted and 11 exhausted the budget. No new proof was found. These are bounded
position claims, not a solution to adversarial chess from the initial array.

## Popeye prescreen

[Popeye 4.101](https://github.com/thomas-maeder/popeye/releases/tag/v4.101),
a purpose-built open-source chess-problem solver, was tested as a fast
selfmate prescreen. It processed the 1,132-position corpus locally in 39.2
seconds for s#1 through s#4:

- 977 eligible positions had no reported solution;
- 2 known selfmate-in-one positions were found;
- 153 positions were conservatively skipped because the adapter did not encode
  castling or en-passant state;
- one selfmate-in-one independently proven by the Python solver was missed.

Popeye is therefore retained as a fast candidate generator, not an authority
for gameplay labels. Every positive must be independently validated by the
orthodox Python solver, and a Popeye negative must not be treated as a proof.

## Adversarial training

The new ranked generator can run the deployed stalemate-aware policy against
the trying-to-lose population. It produced:

- 1,885 positions from 100 selfish-random-reply trajectories: 1,825 positions
  inherited a terminal draw value and 60 remained unvalued at the horizon;
- 400 positions from 20 mixed-population trajectories: 140 inherited a draw
  value and 260 remained unvalued.

Both candidates were trained from scratch on an L4 with the original
reverse-Stockfish and random-reply corpora plus the new hard negatives.

### v0.5a: selfish-random-reply corpus

The model reached 30.8% validation rank-one and 28.1% test rank-one. On the
exact frozen 200-game random suite it regressed from v0.3:

| Model | Losses | Draws | Unresolved | Wins | Median loss plies |
|---|---:|---:|---:|---:|---:|
| Selected v0.3 | 188 | 8 | 4 | 0 | 70 |
| Adversarial v0.5a | 181 | 17 | 2 | 0 | 89 |

Among 170 games both models lost, v0.5a was 17.2 plies slower on average. Its
restricted-horizon mean was 27.3 plies worse. On a fresh 40-game adversarial
screen it produced 0 losses, 38 draws, one truncation, and one Shallow Red win.
The matched v0.3 control drew all 40 with no wins. v0.5a is rejected.

### v0.5b: mixed population

Adding the 400-position mixed corpus produced 30.9% validation rank-one and
29.4% test rank-one. It restored the matched adversarial result to 40 draws,
zero losses, and zero wins. On the first 40 games of the frozen random suite,
both v0.3 and v0.5b lost 39 games and drew one, but v0.5b was slower:

| Model | Mean loss plies | Median loss plies |
|---|---:|---:|
| Selected v0.3 | 88.8 | 67 |
| Adversarial v0.5b | 119.4 | 80 |

v0.5b is also rejected. Draw-only hard negatives changed the state
distribution but did not provide the missing action target: a move toward a
provable forced selfmate.

## Next research direction

The opponent population should remain a promotion gate and a source of hard
states, but not be used alone as a teacher. The next useful training corpus
needs positive forced-selfmate ancestry:

1. Generate or import legal selfmate compositions and reachable game-like
   positions.
2. Prove them, then retain every target move on a certified principal tree.
3. Search backward or perturb positions to obtain earlier ancestors that still
   have a proof.
4. Train a proof-distance policy/value head, mixing these labels with the
   current random-opponent corpus to preserve ordinary loss rate and safety.
5. Gate every candidate on the frozen random suite and the trying-to-lose
   population before promotion.

GPU time is appropriate for step 4. Proof enumeration remains branch-heavy CPU
work and did not benefit from allocating an L4.

## Verification

- Ruff: passed.
- mypy over 47 source files: passed.
- pytest: 258 passed, one environment-dependent Stockfish test skipped.
- No new model was promoted or wired into the web deployment.
