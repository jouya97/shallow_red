import type { Metadata } from "next";
import Link from "next/link";
import { SiteHeader } from "../SiteHeader";

export const metadata: Metadata = {
  title: "The Technical Stuff — Shallow Red",
  description:
    "How Shallow Red was trained, evaluated, and distilled into a browser chess engine that tries to lose.",
};

export default function TechnicalPage() {
  return (
    <main>
      <SiteHeader current="technical" />

      <section className="technical-hero">
        <p className="kicker">Inside the world&apos;s least ambitious chess engine</p>
        <h1>The Technical Stuff</h1>
        <p className="technical-byline">By <strong>Jian Ouyang</strong></p>
        <p className="lede">
          Shallow Red began with a simple question: what happens when a chess
          engine values its own checkmate above everything else?
        </p>
      </section>

      <article className="technical-article">
        <section className="technical-section">
          <p className="section-number">01</p>
          <div>
            <h2>What does “losing” mean?</h2>
            <p>Shallow Red succeeds when it gets checkmated.</p>
            <p>
              Winning is a failure. Draws are also undesirable. Given two
              successful ways to lose, it prefers the faster one—but reliability
              always comes before speed.
            </p>
            <p>
              The objective is reliability first and speed second. A system that
              loses 95% of its games slowly is better than one that loses 85%
              quickly and draws the rest. Several faster experimental versions
              were rejected for exactly that reason.
            </p>
          </div>
        </section>

        <section className="technical-section">
          <p className="section-number">02</p>
          <div>
            <h2>The policy is not the engine</h2>
            <p>
              Shallow Red has two layers. A neural <em>policy</em> ranks moves.
              A deterministic tactical evaluator makes the final decision.
            </p>
            <p>
              The <strong>research policy</strong> is the larger 32-channel
              neural network. The <strong>research system</strong> combines that
              policy with the Python stalemate-aware reply evaluator.
            </p>
            <p>
              The <strong>browser policy</strong> is the smaller quantized
              network downloaded by this page. The <strong>browser engine</strong>
              combines it with a TypeScript version of the tactical evaluator.
              In both systems, the network proposes and the evaluator decides.
            </p>
          </div>
        </section>

        <section className="technical-section">
          <p className="section-number">03</p>
          <div>
            <h2>From a board to 4,672 actions</h2>
            <p>
              A position becomes 21 planes of 8 × 8 values. Twelve planes locate
              Shallow Red&apos;s six piece types and its opponent&apos;s six piece
              types. The remaining planes encode side to move, four castling
              rights, en passant, the halfmove clock, and repetition state.
            </p>
            <p>
              Black positions are mirrored by rank, so the network always sees
              Shallow Red&apos;s home rank at the bottom. Policy actions are
              mirrored the same way.
            </p>
            <p>
              The policy head emits 4,672 logits: 64 origin squares multiplied
              by 73 move planes. Those planes describe sliding moves, knight
              moves, and underpromotions. Castling, en passant, and queen
              promotions fit into the same geometry. Illegal actions are masked
              before ranking.
            </p>
            <div className="technical-table-wrap">
              <table className="technical-table">
                <thead>
                  <tr><th>Policy</th><th>Channels</th><th>Residual blocks</th><th>Policy parameters</th></tr>
                </thead>
                <tbody>
                  <tr><td>Research</td><td>32</td><td>4</td><td>82,473</td></tr>
                  <tr><td>Browser</td><td>24</td><td>3</td><td>37,633</td></tr>
                </tbody>
              </table>
            </div>
            <p>
              Each residual block contains two 3 × 3 convolutions, ReLU
              activations, and a skip connection. A 1 × 1 policy head converts
              the final features into the 73 move planes.
            </p>
          </div>
        </section>

        <section className="technical-section">
          <p className="section-number">04</p>
          <div>
            <h2>Learning a ranking, not one move</h2>
            <p>
              Training did not label a position with one supposedly correct
              move. A teacher scored every legal move and assigned a complete
              ranking. The policy learned a soft distribution over those ranks,
              so near-best moves still contributed to the objective.
            </p>
            <p>
              The selected v0.3 dataset contains 9,827 positions: 5,227 labeled
              by reverse-Stockfish analysis and 4,600 labeled by a random-reply
              evaluator. It was split into 7,937 training, 967 validation, and
              923 test positions, keeping related game families together.
            </p>
            <p>
              The 32 × 4 policy trained for 20 epochs on Apple MPS with a rank
              temperature of 2. Actions were aligned to Shallow Red&apos;s
              perspective. Although the architecture includes a value head, the
              selected run set its value-loss weight to zero: the final system
              uses the policy for move ordering, not as an outcome oracle.
            </p>
          </div>
        </section>

        <section className="technical-section">
          <p className="section-number">05</p>
          <div>
            <h2>The research system</h2>
            <p>
              On each turn, the policy keeps its twelve highest-ranked legal
              moves. For every retained move, the evaluator enumerates every
              legal immediate opponent reply.
            </p>
            <p>
              Its leading term is exact under a uniform-reply assumption: what
              fraction of those replies checkmate Shallow Red immediately? It
              also measures draws, checks, captured material, attacks around the
              king, king escape squares, and Shallow Red&apos;s resulting mobility.
            </p>
            <pre className="technical-formula"><code>{`score(move) =
  1e9 × P(checkmated next)
− 1e8 × P(draw next)
+ 1e6 × P(checked next)
+ capture, king-pressure, and mobility terms`}</code></pre>
            <div className="technical-table-wrap">
              <table className="technical-table">
                <thead>
                  <tr><th>Event</th><th>Primary effect</th></tr>
                </thead>
                <tbody>
                  <tr><td>Shallow Red checkmates its opponent</td><td>−1 trillion</td></tr>
                  <tr><td>Opponent can checkmate Shallow Red</td><td>+1 billion × probability</td></tr>
                  <tr><td>Opponent reply creates a draw</td><td>−100 million × probability</td></tr>
                  <tr><td>Opponent reply gives check</td><td>+1 million × probability</td></tr>
                </tbody>
              </table>
            </div>
            <p>
              When Shallow Red has at most 1,000 centipawns of non-king material,
              several smaller incentives reverse. Sacrificing the last pieces
              becomes dangerous, so the evaluator starts preserving material,
              king escapes, and legal mobility to reduce stalemates.
            </p>
          </div>
        </section>

        <section className="technical-section technical-section-red">
          <p className="section-number">06</p>
          <div>
            <h2>What runs on this website</h2>
            <p>
              The browser uses the same two-stage design with a smaller policy:
              24 channels, three residual blocks, and 37,633 policy parameters.
              Per-tensor int8 quantization compresses it into a self-describing
              39.7 KB artifact. The unused value head is not shipped.
            </p>
            <ol className="technical-flow">
              <li><span>01</span>Encode the 21-plane position</li>
              <li><span>02</span>Run the neural policy</li>
              <li><span>03</span>Mask illegal actions and retain the top 12</li>
              <li><span>04</span>Enumerate every immediate opponent reply</li>
              <li><span>05</span>Apply the tactical score and play the winner</li>
            </ol>
            <p>
              The neural logits determine which moves reach the tactical stage;
              they do not directly choose the move. If the policy cannot load or
              inference fails, the same tactical scorer evaluates every legal
              move instead.
            </p>
            <p>
              Model loading, inference, legal-move generation, reply enumeration,
              and scoring all happen locally in your browser. No GPU or permanent
              chess server is required.
            </p>
          </div>
        </section>

        <section className="technical-section">
          <p className="section-number">07</p>
          <div>
            <h2>How bad is it?</h2>
            <p>
              Against a uniform-random opponent, the selected research system was
              checkmated in 281 of 300 frozen games:
            </p>
            <div className="technical-stats" aria-label="Research-system evaluation results">
              <div><strong>93.7%</strong><span>Losses</span></div>
              <div><strong>5%</strong><span>Draws</span></div>
              <div><strong>1.3%</strong><span>Unfinished</span></div>
              <div><strong>0</strong><span>Wins</span></div>
            </div>
            <p>
              It also lost all 100 games against Stockfish at 1,000 nodes per
              move. Successful random-opponent losses took a median of 70
              plies—about 35 full moves.
            </p>
            <p>
              Before browser export, the smaller 24 × 3 policy was tested in the
              equivalent Python hybrid across random, Stockfish, weak, and stress
              opponents. It self-checkmated in 474 of 500 games, with 25 draws,
              one unresolved game, and zero wins.
            </p>
            <p>
              After quantization, its held-out rank-one accuracy was 30.4% and
              its top-12 set agreed with the float policy 92.4% of the time. The
              dependency-free TypeScript forward pass measured 6.24 ms median in
              Node on the development Mac.
            </p>
            <aside className="technical-note">
              The 500-game candidate suite used the equivalent Python hybrid,
              not the exact production TypeScript path. Eight parity fixtures
              cover colors, castling, en passant, repetition, low material, and
              promotions, but the deployed engine still needs its own long
              gameplay evaluation.
            </aside>
          </div>
        </section>

        <section className="technical-section">
          <p className="section-number">08</p>
          <div>
            <h2>Why not just reverse Stockfish?</h2>
            <p>
              Stockfish is very helpful when the opponent wants to win. It
              eagerly accepts hanging pieces and finds mating attacks. Against
              that cooperative objective, Shallow Red can lose extremely well.
            </p>
            <p>
              Random and weak opponents are harder. They overlook free material,
              ignore attacks, miss checkmates, and accidentally create
              stalemates. Shallow Red therefore needs an explicit model of
              unhelpful replies rather than simply negating a normal chess
              evaluation.
            </p>
          </div>
        </section>

        <section className="technical-section">
          <p className="section-number">09</p>
          <div>
            <h2>Limits and negative results</h2>
            <p>
              Shallow Red is not a general selfmate solver. Its tactical layer
              looks through one opponent reply, assumes a particular reply
              distribution, and can only inspect moves retained by the neural
              shortlist.
            </p>
            <ul>
              <li>
                The bare v0.3 neural policy self-checkmated in only 33% of a
                100-game random-opponent development suite and accidentally won
                6%. Search is essential.
              </li>
              <li>Models trained to lose faster became less reliable.</li>
              <li>Rollout search looked faster only because it converted difficult losses into draws.</li>
              <li>Repetition penalties reduced overall performance.</li>
              <li>Two copies of Shallow Red produced 40 repetition draws in symmetric self-play.</li>
              <li>Standard chess tablebases rarely affected actual games.</li>
              <li>
                An exact 2.8-million-position bishop-versus-rook analysis found
                no nonterminal position where the losing side could force its
                own checkmate against perfect resistance.
              </li>
            </ul>
            <p>
              The website and research harness also use different draw behavior.
              The browser ends games under chess.js&apos;s claimable threefold
              repetition and 50-move rules. Some research evaluations continued
              until automatic fivefold repetition or the 75-move rule.
            </p>
            <p>
              Every result is conditional on its opponents, openings, seeds,
              move limits, and rules. Zero observed wins is evidence, not a proof
              that Shallow Red can never win.
            </p>
          </div>
        </section>

        <section className="technical-principle">
          <p className="eyebrow">The guiding rule</p>
          <blockquote>
            First, lose consistently.<br />
            Then, and only then, lose faster.
          </blockquote>
        </section>

        <section className="technical-credit">
          <p className="eyebrow">Credit</p>
          <p>
            Shallow Red was researched, trained, evaluated, built, and deployed
            in collaboration with <strong>Codex using gpt-5.6-sol-high</strong>.
          </p>
          <Link className="button technical-back" href="/">Back to the game</Link>
        </section>
      </article>

      <footer>
        <span>Legal chess. Deeply questionable goals.</span>
      </footer>
    </main>
  );
}
