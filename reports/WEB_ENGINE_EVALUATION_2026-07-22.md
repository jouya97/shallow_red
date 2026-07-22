# Web-engine frozen evaluation — 2026-07-22

This report evaluates the exact lightweight TypeScript policy deployed in the
Shallow Red web app. It answers whether that policy loses as reliably as the
selected v0.3 research system.

## Result

No. The web policy remains safe in this sample—it never won—but it converted
fewer games into its own checkmate and drew substantially more often.

| System | Games | Shallow Red losses | Draws | Unresolved | Shallow Red wins |
|---|---:|---:|---:|---:|---:|
| Research v0.3 | 300 | **281 (93.7%)** | 15 (5.0%) | 4 (1.3%) | 0 |
| Web distilled v1 | 300 | **254 (84.7%)** | 44 (14.7%) | 2 (0.7%) | 0 |

The web policy's loss rate is 9.0 percentage points lower. An
opening-clustered paired bootstrap interval for the difference is
`[-13.7, -4.3]` percentage points, so this is not explained by ordinary sample
noise in these suites.

With zero wins in 300 games, the one-sided 95% Wilson upper bound on the web
policy's win probability is approximately 0.89%. This is evidence about the
declared random-opponent distribution, not a guarantee for every possible
player or position.

## Suite breakdown

| Suite | Games | Losses | Draws | Unresolved | Wins |
|---|---:|---:|---:|---:|---:|
| Primary, seed `20261221` | 200 | 169 (84.5%) | 30 (15.0%) | 1 (0.5%) | 0 |
| Safety extension, seed `20261321` | 100 | 85 (85.0%) | 14 (14.0%) | 1 (1.0%) | 0 |

The result was balanced across colors: 128/150 losses as White (85.3%) and
126/150 as Black (84.0%). Conditional on successful self-checkmate, the web
policy took 69.5 median plies and 94.1 mean plies. The research system took 74
median and 104.2 mean plies among its successful games, but that conditional
speed comparison is secondary because the research system successfully lost
27 more games.

## Draw-rule difference

The deployed browser uses `chess.js` game-over behavior, which ends games when
a 50-move or threefold-repetition draw is available. The research harness was
deliberately configured to never claim optional draws and instead continues
until an automatic 75-move or fivefold-repetition draw.

The web result therefore uses the product's real draw behavior. Its 44 draws
were 26 threefold repetitions, 16 stalemates, and two 50-move draws. Testing
past those points would require changing the deployed engine's rules and would
no longer measure the actual web experience. The 9-point aggregate gap mixes
this product-level draw behavior with the policy's weaker long-horizon move
selection; it should not be interpreted as a pure neural-versus-heuristic
ablation.

## Method

- Used the exact exported `chooseLosingMove` function through a persistent
  TypeScript JSON-lines worker; no Python reimplementation of its scoring was
  used.
- Reused the original 150 frozen six-ply openings, both target colors, original
  seeds and game IDs, and the deterministic uniform-random opponent.
- Preserved the 600-ply cap and verified every expected opening/color game key
  exactly once.
- Ran 15 independent ten-opening shards on one-CPU Modal workers. No GPU was
  used.
- Examined 16,923 web-policy decisions. Mean decision time on the cloud workers
  was 781 ms.
- Observed zero illegal moves and zero protocol failures.

## Conclusion

The web build is a good lightweight losing engine, but it is not equivalent to
the research engine. Its strongest result is still the safety result—zero wins
in 300 frozen random-opponent games. Its main weakness is converting difficult
positions: it often repeats or reaches stalemate instead of completing its own
checkmate.

The next web-focused improvement should target repetition and stalemate without
adding enough search to compromise browser latency. The most promising path is
a compact learned move-ordering table or quantized policy used only to shortlist
root moves, while retaining the current exact immediate-reply scan as a tactical
safety layer.
