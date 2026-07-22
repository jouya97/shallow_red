import { Chess } from "chess.js";
import { getDb } from "../../../db";
import { gameCounters, gameResults } from "../../../db/schema";

type RecordedOutcome = "loss" | "win";

async function readCounters() {
  const db = await getDb();
  const [counters] = await db.select().from(gameCounters).limit(1);

  if (!counters) {
    throw new Error("The game counter row is unavailable.");
  }

  return { losses: counters.losses, wins: counters.wins };
}

function errorResponse(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected error";
  const detail =
    error instanceof Error && error.cause instanceof Error ? error.cause.message : "";
  const combined = `${message}\n${detail}`;

  if (combined.includes("no such table") || combined.includes("game_counters")) {
    return Response.json(
      { error: "The game statistics database has not been migrated yet." },
      { status: 503 },
    );
  }

  return Response.json({ error: message }, { status: 500 });
}

export async function GET() {
  try {
    return Response.json(await readCounters(), {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as { gameId?: string; moves?: unknown };

    if (!payload.gameId || !/^[0-9a-f-]{36}$/i.test(payload.gameId)) {
      return Response.json({ error: "A valid gameId is required." }, { status: 400 });
    }
    if (
      !Array.isArray(payload.moves) ||
      payload.moves.length > 512 ||
      payload.moves.some((move) => typeof move !== "string" || move.length > 16)
    ) {
      return Response.json({ error: "A valid move list is required." }, { status: 400 });
    }

    const game = new Chess();
    try {
      for (const move of payload.moves as string[]) game.move(move);
    } catch {
      return Response.json({ error: "The submitted game contains an illegal move." }, { status: 400 });
    }
    if (!game.isCheckmate()) {
      return Response.json({ error: "Only completed checkmates are counted." }, { status: 400 });
    }

    const outcome: RecordedOutcome = game.turn() === "b" ? "loss" : "win";

    const db = await getDb();
    await db
      .insert(gameResults)
      .values({ id: payload.gameId, outcome })
      .onConflictDoNothing();

    return Response.json(await readCounters(), {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    return errorResponse(error);
  }
}
