import Link from "next/link";

type SiteHeaderProps = {
  current: "play" | "technical";
};

export function SiteHeader({ current }: SiteHeaderProps) {
  return (
    <header className="site-header">
      <Link href="/" className="wordmark" aria-label="Shallow Red home">
        <span className="mark">SR</span>
        <span>Shallow Red</span>
      </Link>
      <nav className="site-nav" aria-label="Primary navigation">
        <Link href="/" aria-current={current === "play" ? "page" : undefined}>
          Play
        </Link>
        <Link
          href="/technical"
          aria-current={current === "technical" ? "page" : undefined}
        >
          The Technical Stuff
        </Link>
      </nav>
    </header>
  );
}
