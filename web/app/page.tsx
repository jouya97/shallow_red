import { ShallowRedGame } from "./ShallowRedGame";

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a href="#top" className="wordmark" aria-label="Shallow Red home">
          <span className="mark">SR</span>
          <span>Shallow Red</span>
        </a>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="kicker">The world&apos;s least ambitious chess engine</p>
          <h1>You can&apos;t lose.</h1>
          <p className="lede">
            Shallow Red follows every rule of chess and tries to get checkmated
            as quickly as possible.
          </p>
        </div>
      </section>

      <ShallowRedGame />

      <footer>
        <span>Legal chess. Deeply questionable goals.</span>
      </footer>
    </main>
  );
}
