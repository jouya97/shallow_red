import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { Chess } from "chess.js";
import {
  decodeTinyPolicy,
  encodeTinyPolicyMove,
  encodeTinyPolicyObservation,
  inferTinyPolicy,
  rankTinyPolicyMoves,
} from "../lib/tiny-policy";
import fixtures from "./fixtures/tiny-policy-parity.json";

const modelUrl = new URL("../public/tiny-policy-v1.bin", import.meta.url);

function decodeFloat32(encoded: string) {
  const bytes = Buffer.from(encoded, "base64");
  const copy = Uint8Array.from(bytes);
  return new Float32Array(copy.buffer);
}

function rankLegal(logits: Float32Array, legalActions: number[]) {
  return legalActions
    .slice()
    .sort((left, right) => logits[right] - logits[left] || left - right);
}

test("decodes the versioned 24x3 int8 policy", async () => {
  const bytes = await readFile(modelUrl);
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  const policy = decodeTinyPolicy(buffer);

  assert.equal(
    createHash("sha256").update(bytes).digest("hex"),
    fixtures.modelSha256,
  );
  assert.equal(policy.header.channels, 24);
  assert.equal(policy.header.residualBlocks, 3);
  assert.equal(policy.header.policyParameters, 37_633);
  assert.equal(policy.header.orientation, "perspective_vertical_mirror");
});

test("matches Python observation and legal-action coordinates", async () => {
  const bytes = await readFile(modelUrl);
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  const policy = decodeTinyPolicy(buffer);

  for (const fixture of fixtures.cases) {
    const game = new Chess(fixture.initialFen);
    for (const move of fixture.movesUci) game.move(move);
    assert.equal(game.fen(), fixture.fen, fixture.name);
    const perspective = fixture.targetColor === "white" ? "w" : "b";
    assert.deepEqual(
      encodeTinyPolicyObservation(game, perspective),
      decodeFloat32(fixture.observationF32Base64),
      fixture.name,
    );
    const actions = game
      .moves({ verbose: true })
      .map((move) => encodeTinyPolicyMove(game, move))
      .sort((left, right) => left - right);
    assert.deepEqual(actions, fixture.legalActions, fixture.name);
    assert.deepEqual(
      rankTinyPolicyMoves(policy, game, perspective, 12).map(
        ({ action }) => action,
      ),
      fixture.top12Actions,
      fixture.name,
    );
  }
});

test("matches quantized PyTorch logits and legal top-12 rankings", async () => {
  const bytes = await readFile(modelUrl);
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  const policy = decodeTinyPolicy(buffer);

  for (const fixture of fixtures.cases) {
    const observation = decodeFloat32(fixture.observationF32Base64);
    const expected = decodeFloat32(fixture.expectedLogitsF32Base64);
    const actual = inferTinyPolicy(policy, observation);
    let maxAbsoluteError = 0;
    for (let index = 0; index < actual.length; index += 1) {
      maxAbsoluteError = Math.max(
        maxAbsoluteError,
        Math.abs(actual[index] - expected[index]),
      );
    }

    assert.ok(
      maxAbsoluteError < 0.000_2,
      `${fixture.name} max logit error was ${maxAbsoluteError}`,
    );
    assert.deepEqual(
      rankLegal(actual, fixture.legalActions).slice(0, 12),
      fixture.top12Actions,
      fixture.name,
    );
  }
});

test("rejects corrupt policy payloads", () => {
  assert.throws(
    () => decodeTinyPolicy(new ArrayBuffer(11)),
    /truncated/,
  );
  assert.throws(
    () => decodeTinyPolicy(new TextEncoder().encode("not-a-policy!").buffer),
    /invalid magic/,
  );
});
