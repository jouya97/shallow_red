import { Chess } from "chess.js";
import { chooseLosingMove, type EngineDecision } from "./shallow-red";

export type EngineProtocolDecision = EngineDecision & {
  moveUci: string;
};

/** Run the exact browser policy from a serialized orthodox-chess position. */
export function chooseLosingMoveFromFen(fen: string): EngineProtocolDecision {
  const game = new Chess(fen);
  const decision = chooseLosingMove(game, game.turn());
  return {
    ...decision,
    moveUci: `${decision.move.from}${decision.move.to}${decision.move.promotion ?? ""}`,
  };
}
