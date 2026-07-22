CREATE TABLE `game_counters` (
	`id` integer PRIMARY KEY NOT NULL,
	`losses` integer DEFAULT 0 NOT NULL,
	`wins` integer DEFAULT 0 NOT NULL,
	`updated_at` integer DEFAULT (unixepoch()) NOT NULL
);
--> statement-breakpoint
CREATE TABLE `game_results` (
	`id` text PRIMARY KEY NOT NULL,
	`outcome` text NOT NULL CHECK (`outcome` IN ('loss', 'win')),
	`finished_at` integer DEFAULT (unixepoch()) NOT NULL
);
--> statement-breakpoint
INSERT INTO `game_counters` (`id`, `losses`, `wins`) VALUES (1, 0, 0);
--> statement-breakpoint
CREATE TRIGGER `count_finished_game`
AFTER INSERT ON `game_results`
BEGIN
	UPDATE `game_counters`
	SET
		`losses` = `losses` + CASE WHEN NEW.`outcome` = 'loss' THEN 1 ELSE 0 END,
		`wins` = `wins` + CASE WHEN NEW.`outcome` = 'win' THEN 1 ELSE 0 END,
		`updated_at` = unixepoch()
	WHERE `id` = 1;
END;
