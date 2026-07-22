import assert from "node:assert/strict";
import test from "node:test";
import { Chess } from "chess.js";
import { replayCheckmateOutcome } from "../lib/game-result";
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

  assert.equal(replayCheckmateOutcome(whiteCheckmates, "b"), "loss");
  assert.equal(replayCheckmateOutcome(whiteCheckmates, "w"), "win");
  assert.equal(replayCheckmateOutcome(blackCheckmates, "w"), "loss");
  assert.equal(replayCheckmateOutcome(blackCheckmates, "b"), "win");
});
