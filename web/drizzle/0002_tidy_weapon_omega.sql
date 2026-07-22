ALTER TABLE `game_results` ADD `engine_color` text;--> statement-breakpoint
ALTER TABLE `game_results` ADD `moves` text DEFAULT '[]' NOT NULL;