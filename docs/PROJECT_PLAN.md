# Worst Chess Ever: research and implementation plan

Status: Phases 0-2 implemented; weak-opponent v0.3 candidate evaluated
Last updated: 2026-07-21

## 1. Objective

Build a chess agent that obeys orthodox chess rules and is highly effective at
getting its own king checkmated. The project will produce:

1. a rules-verified match and evaluation harness;
2. reproducible non-neural losing baselines;
3. a trained policy/value model;
4. search and opponent-curriculum improvements;
5. benchmark reports, checkpoints, and a UCI-compatible playable engine.

"Best at losing" will always be qualified by a named opponent population and
compute budget. Solving full chess under this objective is not a realistic
claim.

## 2. Current repository state

The repository now contains the rules kernel, action and observation encodings,
baseline and neural agents, Stockfish integration, deterministic hard-label and
all-legal-move ranked datasets, joint policy/value training, paired evaluation,
a resistant opponent, policy-guided reverse search, tests, CI, and a playable
UCI package. Three v0.2 seeds and an independent population suite have been
evaluated; generated datasets and checkpoints remain local artifacts. The v0.3
work adds limited-strength and handcrafted weak opponents, deterministic noisy
and regime-switching portfolios, a random-reply teacher/search policy,
backward-compatible perspective-aligned neural actions, and late-game
stalemate avoidance. See `reports/V03_WEAK_OPPONENTS_2026-07-21.md`.

## 3. Formal game definition

Let `A` be the designated losing agent. Its primary terminal utility is:

```text
A is checkmated                 +1
draw                             0
A checkmates its opponent       -1
protocol failure by A           -1
```

The primary score is undiscounted. Preferences are lexicographic:

1. maximize the probability that `A` is checkmated;
2. among successful self-losses, minimize plies to checkmate;
3. prefer a draw to an accidental win;
4. if an accidental win is unavoidable, delay it.

This ordering must be implemented directly in evaluation. Reward shaping such
as material loss or king exposure may help training, but it will be optional,
logged separately, and ablated. It will never define success.

### 3.1 Three different games

One aggregate "loss Elo" would hide the most important distinction. We will
publish three separate results:

- **Ordinary-opponent score:** the opponent plays normal chess and tries to
  checkmate `A`. Both policies are aligned on `A` being mated.
- **Population score:** expected result against a frozen, declared mixture of
  random, heuristic, human-like, and conventional engine policies.
- **Adversarial selfmate score:** the opponent actively avoids mating `A`.
  This tests whether `A` can force the opponent to mate it against resistance,
  which matches the chess-composition notion of a selfmate.

These modes require different opponent-node operators. Merely negating a
reward or a Stockfish evaluation is not enough.

## 4. Rules contract

Use `python-chess` as the move-generation and adjudication kernel, then verify
our wrapper independently:

- every proposed move must be a member of `board.legal_moves` before `push()`;
- no resignation, time forfeiture, or illegal move can count as a successful
  loss;
- automatic draws include stalemate, dead position, fivefold repetition, and
  the 75-move rule;
- checkmate takes precedence over the 75-move rule;
- optional threefold and 50-move claims are evaluated in two declared modes:
  `never_claim` for the main checkmate benchmark and `claim_available` as a
  draw-avoidance stress test;
- complete PGN, pre-move FEN, legal mask, seed, latency, node budget,
  termination reason, dependency versions, engine binary hash, and network hash
  are stored for every game.

The rules fixtures will cover castling, en passant, every promotion, checks,
mates, stalemate, dead material, repetition, and 50/75-move behavior. This is
grounded in the current [FIDE Laws of Chess](https://handbook.fide.com/chapter/E012023)
and cross-checked against the [`python-chess` core API](https://python-chess.readthedocs.io/en/latest/core.html).

## 5. Framework decision

### Adopt initially

| Component | Choice | Purpose |
|---|---|---|
| Language | Python 3.10+ | Fast iteration, local compatibility, and ML ecosystem |
| Rules/UCI/PGN | `python-chess` | Legal moves, outcomes, UCI processes, PGN, tablebases |
| Conventional oracle | Stockfish over UCI | Opponent ladder, move labels, reverse-evaluation baseline |
| Model/training | PyTorch | Masked policy/value network and supervised training |
| Experiment config | YAML plus typed dataclasses | Reproducible, diffable runs without early framework weight |
| Testing | pytest, Hypothesis, Ruff, mypy | Unit, property, integration, and static checks |
| Tracking | local JSONL/Parquet plus TensorBoard | Reproducible artifacts without requiring a hosted service |

Both `python-chess` and Stockfish are GPL-3.0. Before the first distributable
release we will explicitly choose a compatible repository license and document
how external binaries and model/data licenses are handled. The relevant
sources are the [`python-chess` repository](https://github.com/niklasf/python-chess)
and the [official Stockfish repository](https://github.com/official-stockfish/Stockfish).

### Evaluate as a warm start

[DeepMind Searchless Chess](https://github.com/google-deepmind/searchless_chess)
provides pretrained chess models and all-legal-move outcome analysis. Selecting
the lowest predicted conventional outcome should be a strong no-training
baseline, and its smallest model may be a much cheaper warm start than learning
chess representations from scratch. Its software, model, and dataset licenses
must be recorded separately before reuse.

### Defer until justified

| Framework | What it offers | Decision |
|---|---|---|
| [PettingZoo Chess](https://pettingzoo.farama.org/environments/classic/chess/) | AlphaZero-style `8x8x111` observation, 4,672 actions, legal mask, two-agent API | Reuse or mirror its action encoding; a thin reward wrapper is useful for prototypes |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | Simple PPO/DQN experiments; contrib has masked PPO | Acceptable for one fixed-opponent pilot, but league/self-play orchestration is manual |
| [Ray RLlib](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html) | Multi-policy mapping, PettingZoo/OpenSpiel adapters, distributed rollout workers | Add only when one-machine training works and rollout/league scale is the measured bottleneck |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | Principled games/search/RL abstractions and AlphaZero reference code | Best later research backbone for role-specific or adversarial self-play; its own docs say the reference AlphaZero is not designed for superhuman chess scale |
| [Leela Chess Zero](https://github.com/LeelaChessZero/lc0) | Production policy/value MCTS and distributed self-play | Late-stage option only; reversing targets also requires changing search interpretation, resignation, adjudication, and data generation |

We will not begin by forking Stockfish. Its pruning, terminal values, evaluation,
and alternating-player assumptions all encode ordinary adversarial chess. It is
more useful initially as a pinned external opponent and labeler.

## 6. Proposed code layout

```text
pyproject.toml
README.md
configs/
  train/
  eval/
  opponents/
src/worst_chess/
  chess/
    actions.py
    observations.py
    outcomes.py
  objective/
    rewards.py
    shaping.py
  agents/
    base.py
    random.py
    heuristic.py
    reverse_stockfish.py
    searchless.py
    neural.py
  opponents/
    uci.py
    pools.py
  search/
    cooperative.py
    adversarial.py
    mcts.py
  training/
    dataset.py
    environment.py
    replay.py
    trainer.py
    checkpoints.py
  evaluation/
    match.py
    openings.py
    metrics.py
    tournament.py
    report.py
  cli.py
scripts/
  generate_dataset.py
  train.py
  evaluate.py
  benchmark_compute.py
tests/
  unit/
  property/
  integration/
artifacts/                 # ignored; reports, PGNs, datasets, checkpoints
```

Interfaces will isolate the rules kernel, policy, opponent policy, objective,
and search strategy. This lets the same candidate be evaluated against ordinary
and resistant opponents without silently changing its reward semantics.

## 7. Baseline ladder

Each level must beat or add explanatory value beyond the prior levels:

1. **Uniform legal:** samples uniformly from the current legal moves.
2. **Greedy sacrifice:** prefers hanging high-value pieces and exposing its king.
3. **Reverse Stockfish:** evaluates every legal root move at a fixed node budget
   and selects the worst conventional WDL expectation from `A`'s point of view.
4. **Searchless-minimum:** chooses the legal move with the lowest pretrained
   conventional outcome, subject to license and environment compatibility.
5. **Cooperative lookahead:** at both sides' nodes, chooses continuations that
   increase the chance `A` is mated, modeling an ordinary winning opponent.
6. **Adversarial lookahead:** `A` seeks its mate while the opponent avoids it.
7. **Supervised neural policy/value model:** distills ranked legal moves and
   terminal outcomes from generated trajectories.
8. **Search-enhanced learned agent:** masked policy/value inference plus
   role-correct search.
9. **Opponent-curriculum RL:** trains against a frozen mixture and historical
   checkpoints, then adds approximate best-response resisters.

## 8. Training strategy

### Phase 0: specification and scaffold (CPU)

Deliverables:

- packaging, pinned dependencies, CI, lint/type/test commands;
- canonical terminal utility and draw-claim policy;
- fixed action encoding and legal-action masking;
- deterministic seed and artifact schemas;
- unit and property tests for rules, actions, rewards, and replay.

Exit gate:

- at least `10^6` sampled legal transitions with no illegal accepted action;
- all curated rules fixtures pass;
- saved games replay to identical outcomes.

### Phase 1: match harness and non-neural baselines (CPU)

Deliverables:

- tournament runner with color/opening pairing;
- random, heuristic, Reverse Stockfish, and shallow cooperative/adversarial
  search agents;
- pinned Stockfish opponent tiers using fixed nodes rather than wall time;
- first benchmark report with full PGNs and confidence intervals.

Exit gate:

- deterministic 400-game smoke suite;
- zero protocol failures;
- metrics behave directionally as expected;
- larger paired baseline run is reproducible from a clean checkout.

### Phase 2: dataset and supervised model (mostly CPU, then one GPU)

Generate diverse positions from baseline games and legal opening prefixes.
Label legal moves with terminal results, shallow Stockfish WDL, cooperative
lookahead, and optionally Searchless Chess. Split by whole game and opening
family to prevent adjacent-position leakage.

Train a small masked residual network with:

- policy head over 4,672 actions;
- value/outcome head from the designated loser's point of view;
- optional opponent-profile or mode conditioning;
- ranking/distillation loss over legal moves;
- terminal value loss and calibrated WDL reporting.

Shaping targets remain auxiliary; release selection uses terminal outcomes.

Exit gate:

- checkpoint round trips are deterministic;
- illegal actions have zero probability after masking;
- held-out ranking/value metrics beat an untrained network;
- across three seeds, the trained agent improves held-out self-checkmate rate
  beyond bootstrap noise.

### Phase 3: fixed-population curriculum RL (one GPU initially)

Train the designated loser against frozen opponents in increasing difficulty:

1. random and capture-first;
2. shallow conventional Stockfish tiers;
3. stronger Stockfish tiers and human-like policies;
4. historical loser checkpoints;
5. approximate best-response resisters.

Use a single-agent environment with the opponent embedded for the first pilot.
Move to PettingZoo/RLlib only when policy pools or distributed rollout workers
are required. Compare sparse terminal reward against separately flagged shaping
and remove any shaping that does not improve the frozen terminal benchmark.

### Phase 4: adversarial selfmate research (one to multiple GPUs only if proven)

Implement role-correct MCTS/AlphaZero-style training or an OpenSpiel game
transform. Train the target and resister as different policies, preserving the
identity of the designated loser in observations and backups. Maintain a league
of historical policies to reduce cycling.

This phase is a separate research result. Failure to force selfmate from the
initial position does not invalidate strong practical losing play.

### Phase 5: productization

- UCI-compatible engine;
- reproducible CLI for play, evaluation, and reports;
- model card covering objective, opponents, compute, limitations, and licenses;
- small human playtest, kept separate from the frozen benchmark;
- optional board UI after the engine and evaluation are stable.

## 9. Evaluation design

### 9.1 Opponent strata

- uniform and weighted random legal policies;
- capture-first and conventional heuristic policies;
- pinned Stockfish binary/NNUE at 1k, 10k, 100k, and 1M nodes per move;
- optional frozen [Maia](https://github.com/CSSLab/maia-chess) rating-band models
  for human-like moves;
- optional pinned Lc0 executable/network at fixed visits;
- previous checkpoints and independently trained seeds;
- resistant search policies and learned approximate best responses.

All opponents are isolated from the candidate's logits, reward, search tree,
and random state. Hidden opponent versions and seeds are reserved for release
testing.

### 9.2 Positions and pairing

- standard initial position remains the headline result;
- candidate plays equal paired games as White and Black;
- a frozen suite of legal opening prefixes measures generalization and avoids
  single-line memorization;
- development and release opening suites are disjoint;
- small-piece curated selfmate/endgame positions provide exact or retrograde
  cross-checks where feasible;
- ordinary Syzygy WDL/DTZ is used only as a diagnostic because it assumes both
  players optimize ordinary chess.

### 9.3 Primary metrics

- self-checkmate rate;
- mean terminal utility;
- accidental target-win rate;
- total draw rate and breakdown by cause;
- protocol-failure rate, which must be zero;
- results by opponent, target color, opening family, and compute budget.

Secondary metrics include median plies conditional on self-checkmate,
competing-risk survival curves, worst-stratum performance, adversarial score,
policy diversity, and solved-position regret. Material loss, king exposure, and
ordinary engine evaluation are diagnostics only.

### 9.4 Statistics and promotion gates

- pair candidate/baseline games on opponent, color, opening, and seed;
- use paired or cluster bootstrap 95% confidence intervals;
- use McNemar's test for matched binary self-checkmate outcomes where useful;
- freeze the primary opponent mixture, stopping rule, and practical margin
  before a release run;
- use sequential testing only during development, then a fixed release sample.

Candidate gate: about 2,400 paired games, with the candidate's lower 95%
confidence bound exceeding the strongest baseline on the predeclared
population metric and zero legality/protocol failures.

Release gate: 10,000 frozen games plus the solved-position suite, with no hidden
opponent or opening-family collapse. "Best at losing" is scoped to this named
suite. "Selfmate-capable" additionally requires positive results against
resisters.

## 10. Compute policy and when to contact the owner

No cloud GPU is needed for Phases 0 or 1. Rules, Stockfish labeling, search, and
match generation are primarily CPU-bound. We will measure actual throughput
before requesting any rental.

### GPU request A: one technical pilot

Contact the owner only when all are true:

- Phase 0 rules/replay gate passes;
- random, heuristic, and Reverse Stockfish baselines are reproducible;
- train/validation/test datasets are materialized and leakage-checked;
- a small local training smoke test completes without invalid masks, NaNs, or
  checkpoint errors;
- profiling identifies learner throughput, not position generation, as the
  bottleneck.

Expected request: one NVIDIA GPU with 16-24 GB VRAM, plus 8-32 CPU cores and
enough local SSD for datasets/checkpoints. The exact instance and rental hours
will be based on a benchmark, not guessed in advance.

The request will include:

```text
objective and experiment ID
recommended GPU type/count and acceptable substitutes
CPU/RAM/storage requirements
estimated wall time and maximum spend
dataset/checkpoint sizes and transfer plan
automatic checkpoint interval
success metric and early-stop rule
commands needed to reproduce the run
```

### GPU request B: scale-out

Multi-GPU compute is requested only if:

- one-GPU runs show a learning signal across three seeds;
- the trained agent beats the strongest non-neural baseline on held-out games;
- a fitted learning curve projects reaching the candidate gate within the
  declared compute cap;
- scaling from one to two/four GPUs achieves at least 70% parallel efficiency.

Expected late-stage range, only if justified: 2-8 GPUs with at least 24 GB each
and 32-64 CPU cores for distributed inference/self-play.

### Stop rules

Do not scale, or stop an active scale-up, if:

- actor/search throughput starves the GPU after tuning;
- two successive compute doublings do not improve the frozen population score;
- gains disappear against hidden opponents or resisters;
- only shaping diagnostics improve while terminal self-checkmate rate does not;
- instability, OOMs, invalid masks, or irreproducible checkpoints remain.

## 11. Subagent workflow

Subagents will be used for bounded, independently verifiable work, not for
unsupervised architectural divergence. The primary agent owns integration,
objective consistency, and final validation.

Planned assignments:

1. **Rules/harness agent:** implement terminal semantics, draw policies,
   property tests, replay, and PGN artifacts.
2. **Baseline/search agent:** implement heuristic, Reverse Stockfish,
   cooperative, and adversarial search behind common interfaces.
3. **Evaluation agent:** implement opponent pools, paired tournaments,
   bootstrap statistics, and report generation.
4. **Training agent:** implement action/observation encoding, datasets,
   PyTorch model, masking, checkpoints, and supervised training.
5. **RL/scale agent:** added only after the supervised gate; implement the
   fixed-population environment and later distributed league integration.

For each work package, the primary agent will define files/interfaces and
acceptance tests before delegation. Agents will work in parallel only on
non-overlapping modules, report exact files changed and tests run, and have
their work integrated and re-tested centrally. Framework and evaluation
research has already been parallelized this way for this plan.

## 12. Immediate implementation sequence

1. Create `pyproject.toml`, package skeleton, developer commands, and CI.
2. Lock the formal objective and draw-claim configuration in code and tests.
3. Implement fixed action encoding, legal masks, and random legal agent.
4. Build deterministic matches, PGN/FEN artifacts, and replay validation.
5. Add heuristic and Reverse Stockfish agents plus pinned engine discovery.
6. Run the 400-game CPU smoke benchmark and publish the first report.
7. Implement shallow cooperative/adversarial search and larger baseline study.
8. Decide whether Searchless Chess materially improves the baseline/warm start.
9. Generate the supervised dataset and run the local training smoke test.
10. Only then request a precisely sized one-GPU pilot.

## 13. Primary research references

- [FIDE Laws of Chess](https://handbook.fide.com/chapter/E012023)
- [`python-chess` documentation](https://python-chess.readthedocs.io/en/latest/)
- [`python-chess` UCI engine API](https://python-chess.readthedocs.io/en/latest/engine.html)
- [Stockfish repository and GPL license](https://github.com/official-stockfish/Stockfish)
- [PettingZoo chess environment](https://pettingzoo.farama.org/environments/classic/chess/)
- [OpenSpiel AlphaZero documentation](https://openspiel.readthedocs.io/en/stable/alpha_zero.html)
- [OpenSpiel algorithms](https://github.com/google-deepmind/open_spiel/blob/master/docs/algorithms.md)
- [RLlib multi-agent environments](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html)
- [Leela Chess Zero development overview](https://lczero.org/dev/overview/)
- [DeepMind Searchless Chess](https://github.com/google-deepmind/searchless_chess)
- [World Federation for Chess Composition codex](https://www.wfcc.ch/rules/codex/)
- [Lichess open database](https://database.lichess.org/)
