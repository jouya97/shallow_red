# Prioritized synthetic ancestry — 2026-07-22

## Question

Can the two synthetic losses and two synthetic wins teach Shallow Red useful
decisions earlier than the already known mate-in-one positions, without
pretending that a tiny stochastic sample labels every legal move?

## Method

The experiment extracted the final twelve Shallow Red turns from each decisive
synthetic game: 24 positions from losses and 24 from wins. For every position,
the first pass evaluated a deduplicated shortlist containing:

- the frozen v0.3 policy's top four moves;
- the top two material-sacrifice heuristic moves;
- the top two exhaustive random-reply moves;
- the move played in the recorded trajectory.

Each candidate was initially rolled out once for 80 plies against the diverse
synthetic loser league. Only positions producing a selfmate advanced to an
independent four-rollout, 120-ply confirmation. Future Shallow Red turns used
the frozen v0.3 stalemate-aware policy in both stages.

The shortlist is never serialized as an all-legal ranked training example.
Confirmed positions receive a final population rollout over every legal move
before becoming training data.

## Shortlist screen

| Positions | Candidate moves | Preliminary positives | Best beat v0.3 |
|---:|---:|---:|---:|
| 48 | 301 | 9 | 9 |

The preliminary positives included positions 20 plies before an observed mate,
but the screen was intentionally permissive and expected to contain sampling
noise.

## Independent confirmation

Seven of nine preliminary positions produced at least one selfmate again. Six
were taken from the two losing trajectories and one was a missed escape from a
trajectory in which Shallow Red eventually won.

| Source | Plies before game end | Confirmed best | Selfmates | Mean loss plies | v0.3 selfmates |
|---|---:|---|---:|---:|---:|
| loss 1 | 2 | `...Bg7-e5` | 4/4 | 2 | 4/4 |
| loss 1 | 4 | `...Bf8-g7` | 4/4 | 4 | 3/4 |
| loss 1 | 6 | `...e7-e6` | 2/4 | 11 | 0/4 |
| loss 2 | 2 | `...Qc3-d2` | 4/4 | 2 | 0/4 |
| loss 2 | 8 | `...c4-c3` | 1/4 | 10 | 0/4 |
| loss 2 | 10 | `...Nf3-e1` | 1/4 | 32 | 1/4 |
| prior win | 7 | `...Qb5-c4` | 1/4 | 14 | 0/4 |

The strongest four positions reproduce or improve the earlier focused search.
The weaker 1/4 signals remain useful only as candidates for the all-legal pass;
they are not treated as proofs.

## Safety construction

The final Shallow Red turn in each winning game had only one legal move. A
negative policy label there cannot teach the engine to avoid winning. The
safety window was therefore expanded to the last four Shallow Red turns in
each winning trajectory. After deduplication with the recovered positive, the
training seed contains:

- six confirmed loss-trajectory steering positions;
- one recovered steering position from a winning trajectory, with no invented
  state-value label;
- seven earlier winning-trajectory safety positions, with loser-perspective
  value `-1`.

This produces fourteen seed positions. The all-legal rollout pass replaces
their placeholder heuristic ordering before any training is considered.

## All-legal result

The final pass evaluated 393 legal moves with four fresh 120-ply population
rollouts per move.

| Measure | Result |
|---|---:|
| Positions | 14 |
| Positions with at least one selfmate | 11 |
| Best move beat frozen v0.3 | 11 |
| Best move beat the recorded trajectory move | 8 |
| Confirmed loss-steering positions with a selfmate | 6/6 |
| Winning-trajectory safety positions with a selfmate alternative | 4/7 |
| Recovered winning-trajectory steering positions with a selfmate | 1/1 |

The final all-move search strengthened several shortlist results. Six plies
before the first loss, `...Ke8-b8` lost in 2/4 rollouts while v0.3 truncated in
all four. Eight plies before the second loss, `...Ra8-b8` lost in 3/4 while
v0.3 again truncated in all four. Ten plies before that loss, `...Bf8-e7`
produced one loss while both v0.3 and the recorded move truncated throughout.

The winning trajectories were especially useful. Four of seven safety states
contained a move that produced a selfmate in fresh rollouts, including one
move with 2/4 losses. In another safety state, the frozen model's move won in
3/4 rollouts while an alternative avoided every observed win.

## Safety-first correction

The ordinary rollout objective ranks loss frequency before target wins. In one
safety state this preferred `...Qg1-h1`, which produced one loss, one Shallow
Red win, and two truncations. Because the project requires wins to be much
harder to tolerate than draws, safety examples now use a different exact
ordering:

1. fewer Shallow Red wins;
2. more Shallow Red losses;
3. faster losses;
4. fewer truncations.

This changed that position's label to `...Qc5-d4`, which produced one loss,
zero wins, and three truncations. The other six safety rank-one labels were
unchanged. Loss-derived and recovered-positive examples retain the original
loss-first ordering.

## Decision

The experiment worked: prioritized ancestry turned four decisive games into
fourteen honest all-legal examples, and eleven positions expose a better route
than frozen v0.3. The final dataset is ready for either exact lookup or a future
fine-tuning mixture.

It is still too small for a new selected neural checkpoint. The fourteen
positions come from only four trajectory families, so repeated gradient steps
would measure memorization of those families rather than improved behavior
against new humans. This is particularly risky because earlier proof-only
fine-tunes already degraded held-out behavior. The selected v0.3 checkpoint
and web policy therefore remain unchanged.

The pipeline should now be reused on every newly discovered synthetic or human
decisive game. Once it contains many independent trajectory families, train on
the safety-finalized all-legal data mixed with the broad v0.3 corpus, then gate
on zero wins before considering promotion.

## Scaled fresh-family run

The safe-root fuzzer later supplied 117 decisive games descended from 1,536
independent initial-board warmups. The ancestry screen considered the final six
target turns from those games:

| Stage | Positions | Candidate moves | Positions with a selfmate |
|---|---:|---:|---:|
| one-rollout screen | 394 | 2,498 | 188 |
| four-rollout confirmation | 188 | 1,269 | 178 |

The screen included 284 positions from losses and 110 from wins. Confirmed
positives cover 61 independent root families and are balanced between target
colors. The training seed retained:

- 173 confirmed loss-steering positions;
- 5 recovered steering positions from games the target won;
- 80 final-four-turn safety positions from 22 win families.

The resulting 258 positions span 73 independent initial-game families.

## Scaled all-legal teacher

The full pass evaluated 6,742 legal moves with four 120-ply population
continuations per move, or 26,968 counterfactual continuations. Future target
turns used the deployed stalemate-aware top-12 policy, not the cheap bare-neural
continuation that biased the earlier random-speed experiment.

At least one legal action produced a selfmate in 201 of 258 positions. Among
the 173 loss-steering states, the final best action produced a selfmate in 172
and avoided every observed target win in 172. Among the 80 explicit safety
states, 55 had an action with zero observed wins and 22 retained a
selfmate-producing alternative.

An audit found one loss-derived state where ordinary loss-first ordering chose
an action with two selfmates and one target win. Because wins are never an
acceptable tie-breaking cost, safety ordering now applies to every state:

1. fewer target wins;
2. more selfmates;
3. faster selfmates;
4. fewer truncations.

This changed eight rank-one labels in total: the seven previously corrected
win-safety states and the newly detected loss-derived risk.

## Leakage correction

The first scaled split used `(source label, trajectory)` as its grouping key.
That allowed steering and safety examples from the same initial game to enter
different partitions. Its v0.8 held-out metrics are invalid and are not used
for model selection. The weight-one v0.8 candidate also failed the matched
random-opponent gate independently, losing only 87 of 100 games versus 95 for
selected v0.3.

The splitter now groups matching trajectory suffixes regardless of source
label and explicitly rejects any cross-partition trajectory overlap. The clean
family split contains:

| Partition | Families | Positions |
|---|---:|---:|
| Train | 58 | 217 |
| Validation | 7 | 19 |
| Test | 8 | 22 |

## Safety-all fine-tune

Two mild v0.3-initialized candidates used the ancestry train set once or twice,
approximately 2.7% or 5.2% of their training streams. Both used three epochs,
learning rate `5e-5`, rank temperature 0.5, and the complete broad v0.3 anchor
corpus.

| Model | Train rank-one | Validation rank-one | Test rank-one | Test MRR |
|---|---:|---:|---:|---:|
| selected v0.3 | 27.6% | 26.3% | 22.7% | 0.363 |
| safety-all weight 1 | 27.6% | 26.3% | 22.7% | 0.360 |
| safety-all weight 2 | 27.6% | 26.3% | 18.2% | 0.326 |

The lower-weight candidate changed no rank-one decisions in any ancestry
partition. The higher-weight candidate made the clean held-out test worse.
Neither advanced to gameplay.

## Final decision

No neural checkpoint is promoted. The selected v0.3 research policy and the
web engine remain unchanged.

The scaled experiment still produced valuable infrastructure and data:

- 258 honest all-legal, win-first-safe labels from 73 reachable families;
- 83 exact proof roots from 54 loss families, with 2,339 legal moves labeled;
- root-family-safe dataset splitting with an explicit leakage assertion;
- shardable role-matched rollout reranking;
- a clear result that low-weight supervised fine-tuning is too weak, while
  heavier proof fine-tuning memorizes and reduces broad loss reliability.

The best immediate use of these assets is the exact proof book plus live
bounded proof search. A future neural attempt needs either substantially more
earlier-game families or a curriculum/objective that can learn the steering
signal without diluting it into the broad policy loss.
