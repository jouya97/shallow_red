# Adversarial and exact-endgame experiments — 2026-07-21

## Scope and statistical interpretation

The earlier 1,000-position corpus was a training-label experiment, not 1,000
independent evaluation games. A simple random sample of 1,000 independent
Bernoulli outcomes has a worst-case 95% margin of error of about 3.1 percentage
points, but that calculation does not transfer to correlated chess positions,
positions selected from the same trajectories, or distributions not represented
by the sample. Claims below are restricted to their declared position and
opponent distributions.

## Adversary that tries to make Shallow Red win

`selfish-reverse-stockfish` wraps reverse Stockfish from the opponent's own
perspective. It therefore searches for moves that make the opponent lose and
Shallow Red win. In the outer report, a `target_win` is an adversary success;
a `target_was_checkmated` result is a Shallow Red success.

A fresh local screen used ten six-ply openings, both target colors, 32
Stockfish nodes per legal root, and a 300-ply cap:

- 20 games
- 0 Shallow Red selfmates
- 0 Shallow Red wins
- 11 fivefold-repetition draws
- 9 truncations
- 0 protocol failures

The adversary denied every attempted loss, but did not make Shallow Red win.
This is evidence of a frozen-policy interaction, not a proof about all legal
responses or a solved inverted game.

A larger held-out screen ran on Modal CPU at seed 20262321 with 50 new
six-ply openings, both target colors, 32 Stockfish nodes per legal root,
top-12 Shallow Red search, and a 300-ply cap:

- 100 games
- 1 Shallow Red selfmate, at 74 plies
- 0 Shallow Red wins
- 66 fivefold-repetition draws
- 33 truncations
- 0 protocol failures

The two colors from each opening are correlated, so the 100 games are 50
opening clusters rather than 100 independent observations. Zero of 50 clusters
contained an adversary success. The exact cluster-level two-sided 95% upper
bound is 7.11% (the one-sided 95% upper bound is 5.82%). A game-level calculation
would be anti-conservative. This does not prove the true rate is zero or exclude
a 5% opening-family risk. It does show that the tested reverse-Stockfish
adversary more often denies Shallow Red's loss via repetition than forces
Shallow Red to checkmate it. With no observed forcing examples, GPU adversarial
training was not started.

Local artifact:
`artifacts/evaluations/selfish-reverse-stockfish-20g/report.json`.

Modal artifact after download:
`artifacts/evaluations/modal-adversary-screen-100g/modal-adversary-screen-100g/report.json`.

## Exact Syzygy guidance

`SyzygyLosingAgent` evaluates every legal target move using local Syzygy files.
After the target moves, the opponent is the side to move, so the wrapper selects
the successor with the largest opponent standard-chess WDL. Partial coverage is
never mixed with heuristics: if any candidate cannot be probed, the complete
position falls back to the wrapped agent.

This is exact for ordinary chess under an opponent pursuing a normal win. It is
not exact for an opponent trying to avoid checkmating Shallow Red. DTZ is used
only as a same-WDL zeroing-progress tie-break and must not be described as
distance to mate.

The held-out three-piece pilot generated 20 positions in each of KQvK, KRvK,
KBvK, KNvK, and KPvK at seed 20260921:

- 100 generated positions
- 40 automatic insufficient-material draws in KBvK and KNvK
- 60 actionable positions with complete WDL coverage
- 51 positions with complete DTZ coverage across every candidate
- exact guidance improved opponent WDL over the heuristic on 6 positions
- exact guidance tied on 54 and worsened on 0
- selected opponent WDL distribution: 32 wins, 15 draws, 13 losses
- heuristic opponent WDL distribution: 28 wins, 17 draws, 15 losses

The pilot shows that exact guidance fixes real endgame action errors. It does not
estimate behavior over all seven-piece positions. Complete seven-piece Syzygy
data is on the order of tens of terabytes, so future coverage should use named
material families or cached official API responses and report the exact probe
hit rate.

Reproduce locally:

```bash
./scripts/download_syzygy_3piece.sh artifacts/tablebases/syzygy-3
uv run python scripts/syzygy_pilot.py \
  --tablebase artifacts/tablebases/syzygy-3 \
  --positions-per-class 20 --seed 20260921
```

## Modal environment

Modal client 1.5.2 is locked in the optional `cloud` dependency group. The
repository provides CPU, L4, and zero-GPU smoke modes in `modal_app.py`, backed
by the persistent `shallow-red-artifacts` Volume. The zero-GPU image/import test
and a two-game CPU adversarial smoke both passed. No GPU was allocated during
these experiments.
