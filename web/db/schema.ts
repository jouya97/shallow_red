import { sql } from "drizzle-orm";
import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const gameCounters = sqliteTable("game_counters", {
  id: integer("id").primaryKey(),
  losses: integer("losses").notNull().default(0),
  wins: integer("wins").notNull().default(0),
  draws: integer("draws").notNull().default(0),
  totalGames: integer("total_games").notNull().default(0),
  updatedAt: integer("updated_at")
    .notNull()
    .default(sql`(unixepoch())`),
});

export const gameResults = sqliteTable("game_results", {
  id: text("id").primaryKey(),
  outcome: text("outcome", { enum: ["loss", "win", "draw"] }).notNull(),
  engineColor: text("engine_color", { enum: ["w", "b"] }),
  moves: text("moves").notNull().default("[]"),
  finishedAt: integer("finished_at")
    .notNull()
    .default(sql`(unixepoch())`),
});
