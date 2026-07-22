import { createInterface } from "node:readline";
import { chooseLosingMoveFromFen } from "../lib/engine-protocol";

type Request = {
  fen: string;
};

const lines = createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

for await (const line of lines) {
  if (!line.trim()) continue;
  try {
    const request = JSON.parse(line) as Request;
    if (typeof request.fen !== "string") {
      throw new Error("request.fen must be a string");
    }
    process.stdout.write(`${JSON.stringify(chooseLosingMoveFromFen(request.fen))}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stdout.write(`${JSON.stringify({ error: message })}\n`);
  }
}
