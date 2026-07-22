"use client";

import { useEffect, useRef, useState } from "react";
import { Chess, type Square } from "chess.js";
import { opposite, type Side } from "../lib/game-result";
import { chooseLosingMove } from "../lib/shallow-red";

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const PIECES = {
  wk: "♔",
  wq: "♕",
  wr: "♖",
  wb: "♗",
  wn: "♘",
  wp: "♙",
  bk: "♚",
  bq: "♛",
  br: "♜",
  bb: "♝",
  bn: "♞",
  bp: "♟",
} as const;

type Outcome = "loss" | "win" | "draw";
type GlobalStats = { losses: number; wins: number };

export function ShallowRedGame() {
  const [game, setGame] = useState(() => new Chess());
  const recordedOutcome = useRef<Outcome | null>(null);
  const gameId = useRef<string | null>(null);
  const engineTimer = useRef<number | null>(null);
  const [, setFen] = useState(() => game.fen());
  const [humanColor, setHumanColor] = useState<Side>("w");
  const [selected, setSelected] = useState<Square | null>(null);
  const [thinking, setThinking] = useState(false);
  const [stats, setStats] = useState<GlobalStats | null>(null);

  useEffect(() => () => {
    if (engineTimer.current !== null) window.clearTimeout(engineTimer.current);
  }, []);

  useEffect(() => {
    let active = true;

    async function refreshStats() {
      try {
        const response = await fetch("/api/stats", { cache: "no-store" });
        if (!response.ok) return;
        const nextStats = (await response.json()) as GlobalStats;
        if (active) setStats(nextStats);
      } catch {
        // The board stays playable if the public counter is temporarily offline.
      }
    }

    void refreshStats();
    const timer = window.setInterval(refreshStats, 10_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const engineColor = opposite(humanColor);
  const displayFiles = humanColor === "w" ? FILES : [...FILES].reverse();
  const displayRanks = humanColor === "w" ? RANKS : [...RANKS].reverse();
  const legalTargets = (() => {
    if (!selected) return new Set<Square>();
    return new Set(
      game.moves({ square: selected, verbose: true }).map((move) => move.to),
    );
  })();

  const history = game.history();
  const canUndo = game
    .history({ verbose: true })
    .some((move) => move.color === humanColor);
  const status = describeStatus(game, thinking, humanColor, engineColor);

  function playSquare(square: Square) {
    if (thinking || game.isGameOver() || game.turn() !== humanColor) return;
    const piece = game.get(square);
    if (!selected) {
      if (piece?.color === humanColor) setSelected(square);
      return;
    }
    if (piece?.color === humanColor) {
      setSelected(square);
      return;
    }

    try {
      game.move({ from: selected, to: square, promotion: "q" });
    } catch {
      setSelected(null);
      return;
    }
    gameId.current ??= window.crypto.randomUUID();
    setSelected(null);
    setFen(game.fen());
    if (game.isGameOver()) recordFinishedGame(game, engineColor);
    else scheduleEngineMove(game, engineColor);
  }

  function scheduleEngineMove(activeGame: Chess, activeEngineColor: Side) {
    setThinking(true);
    engineTimer.current = window.setTimeout(() => {
      if (activeGame.isGameOver() || activeGame.turn() !== activeEngineColor) {
        setThinking(false);
        engineTimer.current = null;
        return;
      }
      const nextDecision = chooseLosingMove(activeGame, activeEngineColor);
      activeGame.move(nextDecision.move);
      setFen(activeGame.fen());
      setThinking(false);
      recordFinishedGame(activeGame, activeEngineColor);
      engineTimer.current = null;
    }, 40);
  }

  function startGame(playerColor: Side) {
    if (engineTimer.current !== null) window.clearTimeout(engineTimer.current);
    engineTimer.current = null;
    const nextGame = new Chess();
    setGame(nextGame);
    setHumanColor(playerColor);
    recordedOutcome.current = null;
    gameId.current = null;
    setFen(nextGame.fen());
    setSelected(null);
    setThinking(false);
    const nextEngineColor = opposite(playerColor);
    if (nextEngineColor === "w") scheduleEngineMove(nextGame, nextEngineColor);
  }

  function resetGame() {
    startGame(humanColor);
  }

  function undoTurn() {
    if (thinking || !canUndo) return;
    while (game.history().length > 0) {
      const undone = game.undo();
      if (undone?.color === humanColor) break;
    }
    recordedOutcome.current = null;
    gameId.current = null;
    setFen(game.fen());
    setSelected(null);
  }

  function recordFinishedGame(finishedGame: Chess, activeEngineColor: Side) {
    const outcome = gameOutcome(finishedGame, activeEngineColor);
    if (!outcome || recordedOutcome.current) return;
    recordedOutcome.current = outcome;
    if (outcome === "draw") return;

    const id = gameId.current ?? window.crypto.randomUUID();
    gameId.current = id;
    void fetch("/api/stats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engineColor: activeEngineColor,
        gameId: id,
        moves: finishedGame.history(),
      }),
      keepalive: true,
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((nextStats: GlobalStats | null) => {
        if (nextStats) setStats(nextStats);
      })
      .catch(() => {
        // A later page refresh will retrieve the authoritative total.
      });
  }

  return (
    <>
      <section className="game-shell" aria-label="Play Shallow Red">
      <div className="board-wrap">
        <div className="board" role="grid" aria-label={`Chess board, ${colorName(humanColor)} perspective`}>
          {displayRanks.flatMap((rank, rankIndex) =>
            displayFiles.map((file, fileIndex) => {
              const square = `${file}${rank}` as Square;
              const piece = game.get(square);
              const isLight = (rankIndex + fileIndex) % 2 === 0;
              const isSelected = selected === square;
              const isTarget = legalTargets.has(square);
              return (
                <button
                  type="button"
                  role="gridcell"
                  aria-label={squareLabel(square, piece ? `${piece.color}${piece.type}` : null)}
                  className={`square ${isLight ? "light" : "dark"}${isSelected ? " selected" : ""}${isTarget ? " legal-target" : ""}`}
                  key={square}
                  onClick={() => playSquare(square)}
                >
                  {piece ? (
                    <span className={`piece ${piece.color === "w" ? "white-piece" : "black-piece"}`}>
                      {PIECES[`${piece.color}${piece.type}` as keyof typeof PIECES]}
                    </span>
                  ) : null}
                  {fileIndex === 0 ? <span className="rank-label">{rank}</span> : null}
                  {rankIndex === 7 ? <span className="file-label">{file}</span> : null}
                </button>
              );
            }),
          )}
        </div>
        <div className="board-actions">
          <button type="button" className="button primary" onClick={resetGame}>New game</button>
          <button type="button" className="button quiet" onClick={undoTurn} disabled={thinking || !canUndo}>Undo turn</button>
        </div>
      </div>

      <aside className="game-panel">
        <div className="side-picker">
          <p className="eyebrow">Play as</p>
          <div className="side-options" role="group" aria-label="Choose your color">
            {(["w", "b"] as const).map((color) => (
              <button
                type="button"
                className="side-option"
                aria-pressed={humanColor === color}
                key={color}
                onClick={() => startGame(color)}
              >
                {colorName(color)}
              </button>
            ))}
          </div>
        </div>

        <div className="status-card" aria-live="polite">
          <span className={`status-dot${thinking ? " thinking" : ""}`} />
          <div>
            <p className="eyebrow">Current position</p>
            <h2>{status.title}</h2>
            <p>{status.detail}</p>
          </div>
        </div>

        <div className="rule-card">
          <p className="eyebrow">Your assignment</p>
          <p className="big-rule">Play {colorName(humanColor)}. Try to lose.</p>
          <p>Shallow Red plays {colorName(engineColor)} and wants you to checkmate it. Normal chess rules apply.</p>
        </div>

        <div className="move-log">
          <div className="move-log-heading">
            <p className="eyebrow">Moves</p>
            <span>{history.length} plies</span>
          </div>
          <div className="moves" aria-label="Move history">
            {history.length === 0 ? <span className="empty-log">Make the first move.</span> : history.map((move, index) => (
              <span key={`${index}-${move}`}><b>{index % 2 === 0 ? `${Math.floor(index / 2) + 1}.` : ""}</b>{move}</span>
            ))}
          </div>
        </div>
        </aside>
      </section>

      <section className="record-section" aria-labelledby="record-heading">
        <div className="record-copy">
          <p className="eyebrow">All-time record</p>
          <h2 id="record-heading">Shallow Red&apos;s record</h2>
        </div>
        <div className="record-totals">
          <div><span>Losses</span><strong>{stats?.losses ?? "—"}</strong></div>
          <div><span>Wins</span><strong>{stats?.wins ?? "—"}</strong></div>
        </div>
      </section>
    </>
  );
}

function gameOutcome(game: Chess, engineColor: Side): Outcome | null {
  if (!game.isGameOver()) return null;
  if (!game.isCheckmate()) return "draw";
  return game.turn() === engineColor ? "loss" : "win";
}

function describeStatus(
  game: Chess,
  thinking: boolean,
  humanColor: Side,
  engineColor: Side,
) {
  if (game.isCheckmate()) {
    return game.turn() === engineColor
      ? { title: "Shallow Red lost.", detail: "Exactly as designed. You checkmated the engine." }
      : { title: "You lost.", detail: "Shallow Red won. Please preserve the evidence." };
  }
  if (game.isDraw()) return { title: "Draw.", detail: "Neither side succeeded at losing." };
  if (thinking) return { title: "Plotting defeat…", detail: "Shallow Red is choosing a move." };
  if (game.isCheck()) return { title: "Check.", detail: game.turn() === humanColor ? "Your king is under attack." : "Shallow Red's king is under attack." };
  return game.turn() === humanColor
    ? { title: "Your move.", detail: `Select a ${colorName(humanColor).toLowerCase()} piece, then choose a highlighted square.` }
    : { title: "Shallow Red's move.", detail: "It is looking for the cleanest way to fail." };
}

function colorName(color: Side) {
  return color === "w" ? "White" : "Black";
}

function squareLabel(square: Square, piece: string | null) {
  if (!piece) return `${square}, empty`;
  const colors = piece[0] === "w" ? "white" : "black";
  const names: Record<string, string> = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };
  return `${square}, ${colors} ${names[piece[1]]}`;
}
