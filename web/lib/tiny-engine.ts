import { type Chess, type Color } from "chess.js";
import {
  chooseLosingMoveFromCandidates,
  type EngineDecision,
} from "./shallow-red";
import {
  rankTinyPolicyMoves,
  type TinyPolicy,
} from "./tiny-policy";

export const TINY_POLICY_SHORTLIST = 12;

export function chooseTinyPolicyMove(
  game: Chess,
  policy: TinyPolicy,
  targetColor: Color = game.turn(),
  shortlistSize = TINY_POLICY_SHORTLIST,
): EngineDecision {
  if (!Number.isInteger(shortlistSize) || shortlistSize < 1) {
    throw new Error("Tiny policy shortlist size must be a positive integer.");
  }
  const candidates = rankTinyPolicyMoves(
    policy,
    game,
    targetColor,
    shortlistSize,
  ).map(({ move }) => move);
  return chooseLosingMoveFromCandidates(game, candidates, targetColor);
}
