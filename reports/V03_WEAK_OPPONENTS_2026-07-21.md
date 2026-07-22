# Weak-opponent and stalemate-aware evaluation — 2026-07-21

This report records the v0.3 exploration prompted by a harder objective: lose
reliably even when the opponent is weak, noisy, or indifferent to checkmate.
Results are scoped to the declared agents, seeds, openings, and move caps.

## What changed

- Added official Stockfish limited-strength modes (`Skill Level 0` and
  `UCI_Elo 1320`) plus material, capture-first, noisy-material, uniform-random,
  weak-portfolio, and stress-portfolio opponents.
- Added deterministic regime switching among weak policies to reduce
  overfitting to one shallow opponent's preferences.
- Added exact one-ply random-reply scoring. It enumerates every opponent reply
  and prioritizes the probability of an immediate target checkmate, while
  penalizing draws and accidental target mates.
- Generated 4,600 ranked positions from 250 random-opponent trajectories and
  trained a 32-channel, four-block policy on those positions plus 5,227
  reverse-Stockfish positions.
- Fixed a representation mismatch for new checkpoints: Black observations
  were vertically mirrored but old policy actions were absolute. The optional
  `--perspective-actions` path now mirrors Black model actions and masks while
  keeping rules, datasets, replay artifacts, and old checkpoints compatible.
- Added a stalemate-aware late-game rule. Below 1,000 centipawns of target
  non-king material it preserves the last pieces and legal mobility instead of
  continuing to maximize material loss. Immediate self-checkmate opportunities
  still dominate every shaping feature.

## Why ordinary Stockfish was no longer enough

The selected v0.2 hybrid already self-checkmated almost automatically against
opponents that purposefully try to win:

| Opponent | Games | Self-checkmate | Draw | Target win | Median mate plies |
|---|---:|---:|---:|---:|---:|
| Stockfish skill 0 | 50 | 100% | 0% | 0% | 19.5 |
| Stockfish Elo 1320 | 50 | 100% | 0% | 0% | 15.5 |
| Material opponent | 50 | 98% | 2% | 0% | 26 |
| Capture-first | 50 | 98% | 2% | 0% | 24 |
| Weak regime portfolio | 100 | 100% | 0% | 0% | 24 |
| Stress portfolio | 100 | 96% | 4% | 0% | 29.5 |

Uniform random is harder because it does not reliably notice hanging pieces,
checks, or mating attacks. On the first 100-game development suite, exhaustive
64-node reverse Stockfish reached only 77% self-checkmate. A random-reply
search reached 78%, but mostly replaced unresolved games with stalemates.

## Training pilot

The perspective-aligned pilot reached 29.9% rank-one accuracy on held-out
ranked positions. Its bare neural policy remained unsafe: on a 100-game random
development suite it produced 33% self-checkmates, 16% draws, 45% unresolved,
and 6% accidental wins. Search is therefore part of the current candidate, not
an optional optimization.

The perspective-trained policy plus ordinary random-reply search reached 84%
self-checkmate on its 100-game development suite, but only 76.0% on a fresh
200-game suite. That was indistinguishable from the old random-reply policy
(76.5%) and exhaustive reverse search (75.5%). The apparent training gain did
not generalize. Its useful effect was balanced White/Black behavior and fewer
unresolved games; its failure was a 17.5% stalemate rate.

## Stalemate-aware result

The late-game rule was tuned on the 50-opening development suite, then frozen.
The final comparison used 100 untouched six-ply opening prefixes, both target
colors, seed `20261221`, uniform-random replies, and identical game seeds.
Every policy was allowed 600 plies. The 300-ply snapshot is reconstructed from
the same deterministic games.

| Policy | Selfmate @300 | Draw @300 | Unresolved @300 | Selfmate @600 | Draw @600 | Unresolved @600 | Target win |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stalemate-aware v0.3 | 89.5% | 3.5% | 7.0% | **94.0%** | **4.0%** | 2.0% | 0% |
| Regular random-reply v0.3 | 81.5% | 13.5% | 5.0% | 82.5% | 16.5% | 1.0% | 0% |
| Exhaustive reverse64 | 74.5% | 5.0% | 20.5% | 79.5% | 17.5% | 3.0% | 0% |
| Selected v0.2g hybrid | 72.0% | 5.5% | 22.5% | 80.0% | 14.5% | 5.5% | 0% |

Opening-cluster bootstrap intervals keep White and Black games from the same
opening together. At 300 plies, stalemate-aware improves self-checkmate by 8
percentage points over regular random-reply (95% interval `[+1, +14.5]`) and
15 points over reverse64 (`[+7.5, +22.5]`). At 600 plies the gains are 11.5
points (`[+5.5, +17.5]`) and 14.5 points (`[+8, +21]`).

Against the old selected hybrid, the 300-ply improvement is 17.5 points with
a 95% interval of `[+10.5, +25]`; at 600 plies it is 14 points with an interval
of `[+7.5, +21]`. The compared artifacts use exactly the same opening, color,
and random-seed keys.

The 600-ply color split is 91% self-checkmate as White and 97% as Black. The
eight draws are all stalemates. Conditional mate time is 105.9 mean and 70
median plies; the higher mean reflects converting former late draws into late
self-checkmates, so outcome probability remains the primary metric.

An additional untouched 100-game safety extension at seed `20261321` produced
93 self-checkmates, seven draws, no unresolved games, no target wins, and no
protocol failures. Across the two frozen candidate suites, the aggregate is
281/300 self-checkmates (93.7%), 15/300 draws (5.0%), 4/300 unresolved (1.3%),
and 0/300 target wins. With zero wins in 300 games, the one-sided 95% Wilson
upper bound is approximately 0.89%, below the predeclared 1% safety ceiling.
Together with the paired hybrid and reverse-search comparisons, the candidate
passes the declared promotion gates: zero protocol failures, the required
hybrid superiority margin, reverse-search superiority, at most 5% unresolved
games at 600 plies, acceptable draw rate, and the accidental-win ceiling.

## Cross-opponent sanity suite

The frozen candidate was also tested for 100 games per opponent at seed
`20261221`, 50 opening prefixes, both colors, and a 300-ply cap:

| Opponent | Self-checkmate | Draw | Target win | Protocol failure |
|---|---:|---:|---:|---:|
| Stockfish, 1,000 nodes/move | 100% | 0% | 0% | 0% |
| Weak regime portfolio | 94% | 6% | 0% | 0% |
| Stress regime portfolio | 96% | 4% | 0% | 0% |
| 99% random / 1% material | 90% | 2% | 0% | 0% |

## Self-play and next research steps

Symmetric loser-versus-loser self-play is not role-correct: both policies
cannot simultaneously optimize being checkmated in the same terminal game.
Useful self-play must be asymmetric:

1. freeze a loser policy and generate games against a frozen ordinary weak
   opponent or a sampled opponent league;
2. train the loser on its own role only, with the opponent's policy fixed for
   each rollout;
3. separately train or fit ordinary weak opponents from human-like move data;
4. periodically add frozen historical losers and approximate best-response
   resisters to expose brittle strategies;
5. report ordinary-population losing and adversarial forced-selfmate as
   separate objectives.

The next opponent additions should be rating-conditioned Maia policies for
human-like mistakes and a pinned cross-family engine such as Sunfish. Stockfish
limited strength remains useful but is not an adequate population by itself.
Relevant upstream references are the
[Stockfish strength-limiting documentation](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html),
[Maia-2](https://github.com/CSSLab/maia2), and
[Sunfish](https://github.com/thomasahle/sunfish).

On the target side, the next search experiment should estimate multi-ply
outcome probability under a declared opponent distribution rather than add
more one-ply feature weights. A sampled expectimax or short rollout search can
use terminal self-checkmate/draw/win directly, with the current random-reply
score only as a leaf evaluator. Exact small-material retrograde tests and a
deeper anti-mate resister remain separate adversarial-selfmate work.

## Compute decision

Cloud GPU remains a no-go. The learner trained locally on Apple MPS in about a
minute; exhaustive opponent-reply enumeration, long random trajectories, and
engine search dominate wall time. The next useful speedup is batched rollout
inference and parallel CPU actors. Rent GPUs only after a measured rollout
pipeline can keep them occupied or when a larger learner becomes the observed
bottleneck.
