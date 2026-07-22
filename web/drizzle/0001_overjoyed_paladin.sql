DROP TRIGGER `count_finished_game`;--> statement-breakpoint
ALTER TABLE `game_counters` ADD `draws` integer DEFAULT 0 NOT NULL;--> statement-breakpoint
ALTER TABLE `game_counters` ADD `total_games` integer DEFAULT 0 NOT NULL;--> statement-breakpoint
UPDATE `game_counters`
SET `total_games` = `losses` + `wins`
WHERE `id` = 1;--> statement-breakpoint
CREATE TABLE `__new_game_results` (
	`id` text PRIMARY KEY NOT NULL,
	`outcome` text NOT NULL CHECK (`outcome` IN ('loss', 'win', 'draw')),
	`finished_at` integer DEFAULT (unixepoch()) NOT NULL
);--> statement-breakpoint
INSERT INTO `__new_game_results` (`id`, `outcome`, `finished_at`)
SELECT `id`, `outcome`, `finished_at` FROM `game_results`;--> statement-breakpoint
DROP TABLE `game_results`;--> statement-breakpoint
ALTER TABLE `__new_game_results` RENAME TO `game_results`;--> statement-breakpoint
CREATE TRIGGER `count_finished_game`
AFTER INSERT ON `game_results`
BEGIN
	UPDATE `game_counters`
	SET
		`losses` = `losses` + CASE WHEN NEW.`outcome` = 'loss' THEN 1 ELSE 0 END,
		`wins` = `wins` + CASE WHEN NEW.`outcome` = 'win' THEN 1 ELSE 0 END,
		`draws` = `draws` + CASE WHEN NEW.`outcome` = 'draw' THEN 1 ELSE 0 END,
		`total_games` = `total_games` + 1,
		`updated_at` = unixepoch()
	WHERE `id` = 1;
END;
