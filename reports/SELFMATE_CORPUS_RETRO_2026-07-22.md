# Selfmate corpus and proof-ancestry investigation — 2026-07-22

## Outcome

The proposed composition-to-training pipeline works at useful short depth.
A conservative 84-position YACPDB sample was independently reproduced at
selfmate-in-one/two. Quiet legal retro-expansion then produced 138 unique,
exact six-ply forced-selfmate positions from 36 source compositions. Full
move labeling turned these into 276 color-balanced ranked-policy examples.

This is enough to validate the method and build a small ablation dataset. It
is not yet enough to replace or promote the selected v0.3 model. The next
high-value step is to scale the six-ply corpus into the thousands, then mix a
bounded fraction into the existing random-opponent training data.

## Corpus research and licensing boundary

YACPDB exposes a public query interface with diagram, stipulation, author,
publication, and stable ID metadata. The query
`Stip("s#[1-4]") AND NOT Fairy` returned 33,610 records during this run; the
narrower s#1/s#2 query returned 12,946.

The importer deliberately:

- accepts only standard orthodox piece tokens and s#1 through s#4;
- rejects twins, options, malformed/invalid diagrams, and records marked
  cooked, unsound, or without a solution;
- stores the diagram and attribution but not the published solution;
- writes generated records beneath the gitignored `artifacts/` tree;
- treats every database claim as untrusted until the local AND/OR solver
  reproduces it.

No explicit YACPDB bulk-content license was found. This is not a legal
conclusion, but it is a practical boundary: use small attributed research
samples and do not vendor or redistribute a bulk corpus until permission or a
license is clarified. WFCC's official software page lists YACPDB, PDB,
Problemist collections, Popeye, and several proof-game solvers as relevant
tools. Problemist advertises more than 100,000 downloadable problems and
selfmate support, but its software is shareware and its collection licensing
was not clearer, so it was not imported.

## Independent corpus validation

The first YACPDB page returned 100 records. The conservative parser accepted
84 and rejected 16. The local history-preserving AND/OR solver reproduced all
84 accepted stipulations:

| Result | Positions |
|---|---:|
| Proven | 84 |
| Refuted | 0 |
| Unknown | 0 |

The sample contained one s#1 and 83 s#2 positions. Published solution text was
not used as a label or proof hint.

Popeye remains useful as an optional fast prescreen, but the previous
experiment showed that the adapter missed an independently proven s#1. It is
therefore not used as ground truth here.

## Quiet two-ply retro-expansion

For each proven composition, the generator reverses one quiet opponent move
and one quiet target move. Each proposed predecessor is replayed forward with
python-chess, and only legal histories that exactly reconstruct the child
position are retained. Captures, castling inventions, promotions, and
en-passant inventions are excluded.

The proof query is move-specific: it asks whether the generated target move
survives every legal opponent response. This avoids exploring unrelated root
moves and directly supplies a policy label.

Twelve Modal CPU shards covered all 84 seeds:

| Metric | Count |
|---|---:|
| Generated candidate ancestors | 21,919 |
| Sampled and searched | 3,606 |
| Exact six-ply proofs | 138 |
| Refuted | 3,463 |
| Unknown at 100,000 nodes | 5 |
| Source compositions yielding a proof | 36 / 84 |

All 138 positives are unique. A separate lower-horizon audit completely
refuted the same designated moves at four plies: 138 refuted, 0 proven, 0
unknown. The two-ply distance increase is therefore genuine, not merely a
longer proof found for a move that also wins sooner.

## Recursive eight-ply pilot

A second retro layer was tested on two completed five-root shards. It generated
904 possible ancestors and searched 158:

| Result | Count |
|---|---:|
| Any proof found within eight plies | 13 |
| Refuted | 63 |
| Unknown at 100,000 nodes | 82 |

Seven results initially appeared to increase distance. A mandatory six-ply
recheck showed that six of those also had a six-ply strategy that the capped
deeper search had not finished discovering. Only one was a verified new
eight-ply label.

The pipeline now records a distance gain only when the previous horizon is
completely refuted. This pilot proves recursion is possible, but also shows a
sharp solver cliff: 51.9% of searches were unknown. More brute-force depth is
not the next move; transposition/search improvements or a specialist solver
should come first.

## Honest all-move policy labels

The existing ranked trainer requires a score for every legal move. Treating
all non-demonstrated moves as failures would be wrong, so every legal root move
was independently classified at the six-ply horizon.

Across the 138 original positions:

| Move label | Count |
|---|---:|
| Proven forced selfmate | 338 |
| Refuted | 5,137 |
| Unknown | 8 |
| Total | 5,483 |

Ninety positions had exactly one proven move. Others had multiple valid
choices, including one with 28. The dataset assigns ranks in the order proven,
unknown, refuted; it does not invent a preference between equally proven
moves.

Each position is mirrored vertically with colors swapped, producing:

- 276 unique ranked positions;
- 138 White-to-move and 138 Black-to-move examples;
- a `+1` loser-value target, because at least one forced selfmate is proven;
- 36 attribution/source groups, with mirrored pairs in the same trajectory;
- a leakage-safe example split of 214 train, 30 validation, and 32 test under
  seed 20260722, with no source group crossing partitions.

## Training recommendation

Do not promote a model from this 276-example corpus alone. It is structurally
valuable but small and composition-like. Use it now as an ablation and pipeline
test. Before a serious candidate:

1. Clarify corpus reuse terms, then scale orthodox s#2 imports first.
2. Independently prove every diagram and generate sampled six-ply ancestors.
3. Fully label all legal moves; preserve unknowns rather than treating them as
   negatives.
4. Mirror labels for Black and split by original composition ID.
5. Mix proof labels with the existing random-opponent corpus rather than train
   on compositions alone.
6. Promote only if the candidate preserves the frozen random-opponent loss
   rate and improves against the trying-to-lose population.

Proof work should continue on Modal CPU. A GPU becomes useful only for the
mixed neural-training step.

## Implementation

- `scripts/import_yacpdb_selfmates.py`: conservative attributed importer.
- `scripts/expand_selfmate_ancestors.py`: legal quiet retro-expansion and
  move-specific proof validation.
- `scripts/merge_retro_reports.py`: overlap-aware shard merger.
- `scripts/build_proof_ranked_dataset.py`: all-move proof labels and Black
  mirroring.
- `src/worst_chess/objective/proof_search.py`: designated-first-move proof API.
- `modal_app.py`: CPU-sharded ancestry and ranked-label modes.

## External references

- YACPDB: <https://www.yacpdb.org/>
- WFCC software and database directory: <https://www.wfcc.ch/software/>
- Popeye 4.101: <https://github.com/thomas-maeder/popeye/releases/tag/v4.101>
- Problemist collections: <https://www.sudoktor.com/prb/>

## Verification

- All Modal apps stopped after the experiments.
- Ranked JSONL schema round-trip: passed for all 14 shards.
- mypy: passed over 44 source files.
- pytest: 274 passed, one environment-dependent Stockfish test skipped.
- No checkpoint was promoted and no web deployment was changed.
