# Synthetic loser-generation experiment — 2026-07-22

## Objective

Generate reachable games from the normal initial position in which Shallow Red
faces varied opponents that also try to lose, then mine exact selfmates and
empirical steering moves without waiting for human traffic.

## Synthetic league

The new league switches every six plies among reproducible loser personalities:

- a material-sacrifice heuristic;
- a king-exposure and piece-offering heuristic;
- exhaustive random-reply losing search;
- tactical, stalemate-aware losing search;
- an opponent that predicts and exploits the frozen Shallow Red policy.

Each losing personality receives deterministic exploration. Avoidable moves
that immediately win, draw, or repeat a position are filtered before either the
base policy or an exploratory move is selected. This preserves auditable paired
runs while preventing the old population's immediate repetition treaties.

`--target-exploration` independently adds the same safe exploration mechanism
to Shallow Red for trajectory discovery. It is an experiment control, not a
deployable policy setting.

## Deterministic-target baseline

The selected v0.3 policy played 100 games from the initial array against the
synthetic league, with a 600-ply limit:

| Losses | Draws | Wins | Truncations |
|---:|---:|---:|---:|
| 0 | 5 | 0 | 95 |

The anti-repetition mechanism worked: only two games ended by fivefold
repetition. However, 95 games survived to the limit. The last twelve target
turns from all games produced 1,200 reachable positions; every one was
completely refuted through selfmate-in-two.

This shows that deterministic loser-versus-loser generation is a nonterminal
deadlock, even when repetition is removed.

## Exploration sweep

Three target exploration rates were screened for 20 games each:

| Exploration | Losses | Draws | Wins | Truncations |
|---:|---:|---:|---:|---:|
| 10% | 0 | 2 | 0 | 18 |
| 35% | 0 | 15 | 0 | 5 |
| 50% | 0 | 12 | 0 | 8 |

Higher exploration mostly converted truncations into automatic draws. A
separate 50-game run at 20% produced one loss, 24 draws, zero wins, and 25
truncations, making 20% the only setting with positive loss discovery.

## Scaled 20% screen

An additional 200 games were sharded across one-CPU Modal workers:

| Losses | Draws | Wins | Truncations |
|---:|---:|---:|---:|
| 1 | 88 | 2 | 109 |

Across both 20% runs, the result was two losses, 112 draws, two wins, and 134
truncations in 250 games. Both losses and both wins occurred with Shallow Red
as Black. The observed 0.8% loss rate is sufficient for discovery but not for
direct policy training; the equal number of wins makes raw exploration unsafe.

## Exact reachable selfmates

The last twenty target turns from each synthetic loss were searched through
selfmate-in-three with a 100,000-node budget. Each trajectory produced one
independently proven final position and nineteen complete refutations.

| Position | Forced line |
|---|---|
| Black to move after 49 plies | `...Bg7-e5 Rh8xe5#` |
| Black to move after 61 plies | `...Qc3-d2 Nb1xd2#` |

Both positions are legal and actually reachable from the standard initial
array. They are new examples rather than imported compositions.

## Empirical ancestry

All legal moves from the final three target turns of each successful trajectory
were counterfactually replayed against the stochastic synthetic league. The
same frozen v0.3 policy continued future Shallow Red turns.

### First trajectory

| Distance from observed mate | Best rollout move | Best losses | v0.3 losses |
|---:|---|---:|---:|
| 6 plies | `...Bf8-g7` | 3/4 | 0/4 |
| 4 plies | `...Bf8-g7` | 4/4, mean 4 plies | 4/4, mean 6 plies |
| 2 plies | `...Bg7-e5` | 4/4, mean 2 plies | 4/4, mean 2 plies |

### Second trajectory

| Distance from observed mate | Best rollout move | Best losses | v0.3 losses |
|---:|---|---:|---:|
| 6 plies | `...c5-c4` | 1/2 | 0/2 |
| 4 plies | several tied moves | 1/2 | 0/2 |
| 2 plies | `...Qc3-d2` | 2/2, mean 2 plies | 0/2 |

The rollout teacher ranked both exact proof moves first. More importantly, it
found useful empirical steering moves four and six plies before universal proof
was available. This is the positive-ancestry signal missing from the earlier
draw-only adversarial corpus.

## Decision

The synthetic generation idea works, but the current corpus is not large enough
for GPU training and the exploration policy must never be deployed directly.
Selected v0.3 and the web model remain unchanged.

The next efficient step is not another uniform game dump. It is prioritized
expansion around successful trajectories:

1. generate with 20% exploration only as an off-policy discovery actor;
2. retain losses and wins, discarding most draws and truncations;
3. treat winning trajectories as high-weight negative safety examples;
4. counterfactually expand earlier turns from every loss, first with cheap
   one-rollout screening and then confirm promising moves with more rollouts;
5. mix exact proof moves with empirical ancestry labels;
6. train only after collecting many independent trajectory families, then gate
   against the frozen random suite and zero-win tests.

Full all-legal-move rollouts are expensive: the six focused positions took
roughly ten minutes locally even with multiple workers. Candidate shortlisting
or distributed rollout actors are required before scaling ancestry generation.
