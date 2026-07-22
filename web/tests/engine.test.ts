import assert from "node:assert/strict";
import test from "node:test";
import { Chess } from "chess.js";
import { replayGameOutcome } from "../lib/game-result";
import { chooseLosingMoveFromFen } from "../lib/engine-protocol";
import { rewardClaimMailto } from "../lib/reward-claim";
import { chooseLosingMove } from "../lib/shallow-red";

test("the browser engine is deterministic, legal, and non-mutating", () => {
  const game = new Chess();
  game.move("e4");
  const original = game.fen();

  const first = chooseLosingMove(game, "b");
  const second = chooseLosingMove(game, "b");

  assert.deepEqual(first.move, second.move);
  assert.equal(game.fen(), original);
  const moved = new Chess(original);
  assert.doesNotThrow(() => moved.move(first.move));
  assert.ok(first.candidates > 0);
  assert.ok(first.repliesExamined > 0);
});

test("the engine avoids checkmating its opponent when another move exists", () => {
  const game = new Chess("7k/5KQ1/8/8/8/8/8/8 w - - 0 1");

  const decision = chooseLosingMove(game, "w");

  assert.notEqual(decision.lan, "g7g8");
  game.move(decision.move);
  assert.equal(game.isCheckmate(), false);
});

test("the engine finds an immediate selfmate opportunity", () => {
  const game = new Chess(
    "3n4/r1k1b3/p1p5/PpPp4/6p1/2KP2qb/6N1/5r2 w - - 0 70",
  );

  const decision = chooseLosingMove(game, "w");

  assert.equal(decision.lan, "c3d4");
  assert.ok(decision.immediateMateProbability > 0);
});

test("finished games are counted from either Shallow Red color", () => {
  const whiteCheckmates = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"];
  const blackCheckmates = ["f3", "e5", "g4", "Qh4#"];

  assert.equal(replayGameOutcome(whiteCheckmates, "b"), "loss");
  assert.equal(replayGameOutcome(whiteCheckmates, "w"), "win");
  assert.equal(replayGameOutcome(blackCheckmates, "w"), "loss");
  assert.equal(replayGameOutcome(blackCheckmates, "b"), "win");
});

test("drawn games are counted", () => {
  const threefoldRepetition = [
    "Nf3", "Nf6", "Ng1", "Ng8",
    "Nf3", "Nf6", "Ng1", "Ng8",
  ];

  assert.equal(replayGameOutcome(threefoldRepetition, "w"), "draw");
  assert.equal(replayGameOutcome(threefoldRepetition, "b"), "draw");
  assert.equal(replayGameOutcome(["e4"], "b"), null);
});

test("reward claims open a pre-addressed email with the game ID", () => {
  const mailto = rewardClaimMailto({
    claimantEmail: "winner@example.com",
    gameId: "12345678-abcd-1234-abcd-1234567890ab",
    name: "Ada Winner",
    note: "I have done the impossible.",
  });

  assert.match(mailto, /^mailto:jianouyang001@gmail\.com\?/);
  assert.match(decodeURIComponent(mailto), /Ada Winner/);
  assert.match(decodeURIComponent(mailto), /12345678-abcd-1234-abcd-1234567890ab/);
  assert.match(decodeURIComponent(mailto), /winner@example\.com/);
});

test("the evaluation protocol runs the exact browser policy from FEN", () => {
  const game = new Chess();
  game.move("e4");

  const protocolDecision = chooseLosingMoveFromFen(game.fen());
  const directDecision = chooseLosingMove(game, "b");

  assert.equal(protocolDecision.moveUci, directDecision.lan);
  assert.equal(protocolDecision.score, directDecision.score);
  assert.doesNotThrow(() => game.move(protocolDecision.moveUci));
});
