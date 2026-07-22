import { Chess } from "chess.js";

export type Side = "w" | "b";
export type GameOutcome = "loss" | "win" | "draw";

export function opposite(side: Side): Side {
  return side === "w" ? "b" : "w";
}

export function replayGameOutcome(
  moves: readonly string[],
  engineColor: Side,
): GameOutcome | null {
  const game = new Chess();
  for (const move of moves) game.move(move);
  if (!game.isGameOver()) return null;
  if (!game.isCheckmate()) return "draw";
  return game.turn() === engineColor ? "loss" : "win";
}
