import { readFile } from "node:fs/promises";
import {
  decodeTinyPolicy,
  inferTinyPolicy,
} from "../lib/tiny-policy";
import fixtures from "../tests/fixtures/tiny-policy-parity.json";

const modelUrl = new URL("../public/tiny-policy-v1.bin", import.meta.url);

function decodeFloat32(encoded: string) {
  const bytes = Buffer.from(encoded, "base64");
  const copy = Uint8Array.from(bytes);
  return new Float32Array(copy.buffer);
}

const bytes = await readFile(modelUrl);
const buffer = bytes.buffer.slice(
  bytes.byteOffset,
  bytes.byteOffset + bytes.byteLength,
);
const policy = decodeTinyPolicy(buffer);
const observation = decodeFloat32(fixtures.cases[0].observationF32Base64);

for (let iteration = 0; iteration < 20; iteration += 1) {
  inferTinyPolicy(policy, observation);
}

const timings: number[] = [];
for (let iteration = 0; iteration < 200; iteration += 1) {
  const start = performance.now();
  inferTinyPolicy(policy, observation);
  timings.push(performance.now() - start);
}
timings.sort((left, right) => left - right);

let maxAbsoluteError = 0;
for (const fixture of fixtures.cases) {
  const actual = inferTinyPolicy(
    policy,
    decodeFloat32(fixture.observationF32Base64),
  );
  const expected = decodeFloat32(fixture.expectedLogitsF32Base64);
  for (let index = 0; index < actual.length; index += 1) {
    maxAbsoluteError = Math.max(
      maxAbsoluteError,
      Math.abs(actual[index] - expected[index]),
    );
  }
}

console.log(JSON.stringify({
  fixtures: fixtures.cases.length,
  maxAbsoluteError,
  meanMs:
    timings.reduce((sum, duration) => sum + duration, 0) / timings.length,
  modelBytes: bytes.byteLength,
  p50Ms: timings[Math.floor(timings.length * 0.5)],
  p95Ms: timings[Math.floor(timings.length * 0.95)],
  runs: timings.length,
}, null, 2));
