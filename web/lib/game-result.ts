import { Chess } from "chess.js";

export type Side = "w" | "b";
export type DecisiveOutcome = "loss" | "win";

export function opposite(side: Side): Side {
  return side === "w" ? "b" : "w";
}

export function replayCheckmateOutcome(
  moves: readonly string[],
  engineColor: Side,
): DecisiveOutcome | null {
  const game = new Chess();
  for (const move of moves) game.move(move);
  if (!game.isCheckmate()) return null;
  return game.turn() === engineColor ? "loss" : "win";
}
