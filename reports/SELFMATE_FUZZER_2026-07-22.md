# Branching selfmate fuzzer — 2026-07-22

## Objective

Turn the small synthetic-ancestry corpus into many variable legal trajectories
without waiting for human games or spending most compute on 600-ply
loser-versus-loser deadlocks.

## Design

The fuzzer starts from finalized all-legal ancestry labels and skips positions
where only one legal move remains. Black seeds receive reachable color mirrors
so both sides are represented. At each node it branches among:

- the best random-reply losing move;
- the best material-sacrifice move;
- frozen v0.3 policy proposals.

The selected root move is forced once. Future Shallow Red turns use the frozen
v0.3 stalemate-aware policy with deterministic 15% exploration, while the
opponent switches among the synthetic loser league's personalities. Every move
is executed by the ordinary rules-checked match harness.

Nonterminal children are deduplicated by position and retained round-robin
across root seeds. The first version ranks its beam by immediate random-reply
mate probability and the existing random-reply pressure score. Losses and wins
are both stored as PGNs; wins are never relabeled or discarded.

## Near-ancestry pilot

The three-generation pilot began with 24 branchable color-balanced states and
kept a beam of 32:

| Branches | Losses | Wins | Frontiers | Protocol failures |
|---:|---:|---:|---:|---:|
| 522 | 43 (8.2%) | 22 (4.2%) | 457 | 0 |

For comparison, the earlier initial-board 20% exploration run found losses in
0.8% of games. The fuzzer therefore improved local loss discovery by more than
tenfold. The 43 losses covered 13 state-seed roots and 34 unique terminal
checkmate FENs. This is useful variation, although the state seeds ultimately
descend from only four original decisive games and must not be described as 13
independent full-game families.

The final six target turns from every loss deduplicated to 57 reachable
positions. Exact bounded search produced:

| Proven s#1 | Refuted through s#2 | Unknown | White target | Black target |
|---:|---:|---:|---:|---:|
| 36 | 21 | 0 | 17 | 19 |

All 36 proofs are distinct FENs. They contain 13 distinct mating move-pair and
color patterns rather than one terminal trick copied verbatim.

## Deep continuation

The pilot's 32 surviving frontier states had moved beyond the original exact
mates, so a four-generation continuation expanded them with a beam of 64:

| Branches | Losses | Wins | Frontiers | Protocol failures |
|---:|---:|---:|---:|---:|
| 1,998 | 8 (0.4%) | 36 (1.8%) | 1,954 | 0 |

The long-range loss rate fell below the old uniform generator while wins became
4.5 times as common as losses. Immediate mate pressure is therefore a useful
local tactical signal but an unsafe long-range objective.

The eight deep losses still yielded six additional exact s#1 positions, all
with distinct mating move pairs. One requires Black to underpromote with
`...d2-d1=N`, allowing `Kf3-e2#`; this is exactly the kind of strange tactic a
human-authored heuristic is unlikely to propose.

## Combined exact yield

Across the pilot and continuation:

- 42 unique reachable exact selfmate positions were proven;
- 22 have White as the target and 20 have Black as the target;
- 19 distinct mating move-pair/color patterns were found;
- every bounded search completed with no unknown result.

## Decision

Keep the fuzzer as a near-tactic neighborhood augmenter. It is substantially
more efficient than full games when started near confirmed ancestry, generates
balanced White and Black examples, and finds unusual exact tactics.

Do not recursively scale the current pressure-only beam. Its deep continuation
produces more Shallow Red wins than losses. A successor must estimate target
win risk during beam selection and make fewer predicted wins dominate mate
pressure, mirroring the safety-first correction used by the finalized ancestry
dataset. It must also inject fresh initial-board trajectory families; branching
the same four source games can add tactical diversity but not independent
coverage.

The selected v0.3 checkpoint and web policy remain unchanged. The exact proof
reports and decisive PGNs are retained under `artifacts/` for a future proof
book or safety-gated training mixture.

## Safety-first follow-up

The first correction attached the loss and win counts from sibling samples to
every surviving child. Beam selection then ordered fewer sampled wins first,
followed by more sampled losses and tactical pressure. A matched replay of the
1,998-branch deep continuation changed losses from 8 to 8 and wins from 36 to
33. This modest 8% win reduction showed that one segment's sibling outcomes
do not predict most later risk.

To test true independence, a new generator played 64 separate games from the
initial array to 40/41 plies. The frontier was balanced between 32 White and 32
Black targets, with no terminal games or protocol failures during warmup. A
three-generation safety-first search produced:

| Version | Branches | Losses | Wins | Retained source games |
|---|---:|---:|---:|---:|
| sibling safety only | 1,713 | 4 | 8 | 64 |
| safe root proposals | 1,713 | 4 | 2 | 64 |

An audit found that six of the original eight wins occurred on the forced root
move: a neural shortlist refill could reintroduce an immediate move that
checkmated the opponent. Root proposals now discard avoidable immediate wins,
terminal draws, and repetitions after merging every proposal source. The
matched rerun eliminated all six one-ply wins without losing a single generated
selfmate. The two remaining wins occurred nine plies into their segments and
therefore require genuine lookahead.

The four fresh losses came from three independent initial-board games. Their
last six target turns yielded 14 unique candidates, of which exact search
proved five s#1 positions and refuted nine through s#2. Four exact positions
have White as the target and one has Black as the target.

The complete fuzzer campaign now contains 47 unique exact reachable selfmate
positions and 24 mating move-pair/color patterns. The safe-root fuzzer is worth
keeping as a synthetic-data generator: on fresh families it found losses in
0.23% of branches and wins in 0.12%, while preserving every win as safety data.
It is not yet safe enough to become the deployed move policy. Eliminating the
last two wins needs multi-ply win-risk probes or an equivalent learned risk
model before beam admission.

## Independent-family scale run

The corrected generator was then sharded by non-overlapping initial-game
indices. Each game stopped after 40 or 41 warmup plies with the target on move,
and every 64-family shard retained all of its source families through three
fuzzer generations. Corpus growth stopped after crossing a preregistered
roughly 50-family loss gate:

| Initial games | Branches | Losses | Wins | Independent loss families | Independent win families |
|---:|---:|---:|---:|---:|---:|
| 1,536 | 40,938 | 89 | 28 | 54 | 22 |

There were no draws or protocol failures inside the branch segments. The 89
losses covered 72 distinct terminal checkmate FENs; the 28 wins covered 24
distinct terminal FENs. Repeated endings within one initial-game family remain
linked by root provenance and are never counted as independent evidence.

The final six target turns from the losses produced 284 unique reachable
candidates. Bounded exact search through four plies proved 83 positions and
refuted 201, with no unknowns. The proven roots cover all 54 independent loss
families. All-legal labeling then tested 2,339 legal moves:

| Proven moves | Refuted moves | Labeled positions | Source families |
|---:|---:|---:|---:|
| 95 | 2,244 | 83 | 54 |

Unlike the earlier composition corpus, these positions are ordinary reachable
states discovered from the initial array. A fixed family split contains 43
train, 5 validation, and 6 test families.

## Reachable-proof fine-tune

Two v0.3-initialized candidates tested whether reachable proofs transfer better
than composed diagrams. The conservative candidate repeated proof training
families four times with rank temperature 0.25 for five epochs. The aggressive
candidate used twelve copies, temperature 0.10, and eight epochs.

| Model | Proof train rank-one | Proof validation rank-one | Proof test rank-one |
|---|---:|---:|---:|
| selected v0.3 | 19.7% | 57.1% | 40.0% |
| conservative weight 4 | 30.3% | 57.1% | 40.0% |
| aggressive weight 12 | 92.4% | 71.4% | 40.0% |

The aggressive candidate memorized the reachable training diagrams but did not
change a single rank-one decision in the six held-out families. The
conservative candidate barely memorized and also produced no held-out gain.

On the same 100 random-opponent games, selected v0.3 lost 95, the conservative
candidate lost 93, and the aggressive candidate lost 90. None won. Both
candidates also recorded zero losses and zero wins in 40 games against the
synthetic loser league; every game reached the 600-ply cap.

Both checkpoints are rejected. The exact proof corpus remains useful for the
runtime proof book, but proof-only gradient updates again trade away broad loss
reliability without improving held-out selfmate steering.
