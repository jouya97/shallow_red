import { Chess, type Color, type Move, type PieceSymbol, type Square } from "chess.js";

const PIECE_VALUES: Record<PieceSymbol, number> = {
  p: 100,
  n: 320,
  b: 330,
  r: 500,
  q: 900,
  k: 0,
};

const WEIGHTS = {
  targetCheckmate: -1_000_000_000_000,
  immediateMateProbability: 1_000_000_000,
  immediateDrawProbability: -100_000_000,
  checkingReplyProbability: 1_000_000,
  capturedTargetValue: 2_000,
  targetRingAttack: 20_000,
  targetKingEscape: -10_000,
  targetMobility: -100,
  capturedOpponentValue: -20_000,
  lowMaterialThreshold: 1_000,
  lowMaterialCapturedTargetValue: -4_000,
  lowMaterialTargetKingEscape: 5_000,
  lowMaterialTargetMobility: 2_000,
} as const;

export type LosingMove = {
  from: Square;
  to: Square;
  promotion?: PieceSymbol;
};

export type EngineDecision = {
  move: LosingMove;
  lan: string;
  score: number;
  immediateMateProbability: number;
  candidates: number;
  repliesExamined: number;
  elapsedMs: number;
};

type MoveEvaluation = {
  score: number;
  immediateMateProbability: number;
  repliesExamined: number;
};

export function chooseLosingMove(
  game: Chess,
  targetColor: Color = game.turn(),
): EngineDecision {
  if (game.turn() !== targetColor) {
    throw new Error("Shallow Red can only move for its designated color");
  }
  if (game.isGameOver()) {
    throw new Error("Shallow Red cannot move from a finished game");
  }

  const started = performance.now();
  const candidates = game
    .moves({ verbose: true })
    .slice()
    .sort((left, right) => left.lan.localeCompare(right.lan));
  let selected = candidates[0];
  let selectedEvaluation = evaluateMove(game, selected, targetColor);
  let repliesExamined = selectedEvaluation.repliesExamined;

  for (const candidate of candidates.slice(1)) {
    const evaluation = evaluateMove(game, candidate, targetColor);
    repliesExamined += evaluation.repliesExamined;
    if (evaluation.score > selectedEvaluation.score) {
      selected = candidate;
      selectedEvaluation = evaluation;
    }
  }

  return {
    move: moveDescriptor(selected),
    lan: selected.lan,
    score: selectedEvaluation.score,
    immediateMateProbability: selectedEvaluation.immediateMateProbability,
    candidates: candidates.length,
    repliesExamined,
    elapsedMs: performance.now() - started,
  };
}

function evaluateMove(
  game: Chess,
  candidate: Move,
  targetColor: Color,
): MoveEvaluation {
  const position = new Chess(game.fen());
  const capturedOpponentValue = candidate.captured
    ? PIECE_VALUES[candidate.captured]
    : 0;
  position.move(moveDescriptor(candidate));

  if (position.isCheckmate()) {
    return {
      score: WEIGHTS.targetCheckmate,
      immediateMateProbability: 0,
      repliesExamined: 0,
    };
  }
  if (position.isGameOver()) {
    return {
      score: WEIGHTS.immediateDrawProbability,
      immediateMateProbability: 0,
      repliesExamined: 0,
    };
  }

  const replies = position.moves({ verbose: true });
  const opponentColor = opposite(targetColor);
  const targetMaterial = material(position, targetColor);
  const lowMaterial = targetMaterial <= WEIGHTS.lowMaterialThreshold;
  let mateCount = 0;
  let drawCount = 0;
  let checkCount = 0;
  let capturedTarget = 0;
  let ringAttacks = 0;
  let kingEscapes = 0;
  let targetMobility = 0;

  for (const reply of replies) {
    if (reply.captured) {
      capturedTarget += PIECE_VALUES[reply.captured];
    }
    position.move(moveDescriptor(reply));
    if (position.isCheckmate()) {
      mateCount += 1;
      position.undo();
      continue;
    }
    if (position.isGameOver()) {
      drawCount += 1;
      position.undo();
      continue;
    }
    if (position.isCheck()) {
      checkCount += 1;
    }

    const king = position.findPiece({ type: "k", color: targetColor })[0];
    if (king) {
      for (const square of kingRing(king)) {
        ringAttacks += position.attackers(square, opponentColor).length;
      }
    }
    const targetMoves = position.moves({ verbose: true });
    kingEscapes += targetMoves.filter((move) => move.piece === "k").length;
    targetMobility += targetMoves.length;
    position.undo();
  }

  const denominator = replies.length;
  const immediateMateProbability = mateCount / denominator;
  const drawProbability = drawCount / denominator;
  const checkProbability = checkCount / denominator;
  const capturedTargetWeight = lowMaterial
    ? WEIGHTS.lowMaterialCapturedTargetValue
    : WEIGHTS.capturedTargetValue;
  const kingEscapeWeight = lowMaterial
    ? WEIGHTS.lowMaterialTargetKingEscape
    : WEIGHTS.targetKingEscape;
  const mobilityWeight = lowMaterial
    ? WEIGHTS.lowMaterialTargetMobility
    : WEIGHTS.targetMobility;

  return {
    score:
      WEIGHTS.immediateMateProbability * immediateMateProbability +
      WEIGHTS.immediateDrawProbability * drawProbability +
      WEIGHTS.checkingReplyProbability * checkProbability +
      (capturedTargetWeight * capturedTarget) / denominator +
      (WEIGHTS.targetRingAttack * ringAttacks) / denominator +
      (kingEscapeWeight * kingEscapes) / denominator +
      (mobilityWeight * targetMobility) / denominator +
      WEIGHTS.capturedOpponentValue * capturedOpponentValue,
    immediateMateProbability,
    repliesExamined: replies.length,
  };
}

function material(game: Chess, color: Color): number {
  let total = 0;
  for (const piece of ["p", "n", "b", "r", "q"] as const) {
    total += game.findPiece({ type: piece, color }).length * PIECE_VALUES[piece];
  }
  return total;
}

function kingRing(square: Square): Square[] {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  const ring: Square[] = [];
  for (let fileOffset = -1; fileOffset <= 1; fileOffset += 1) {
    for (let rankOffset = -1; rankOffset <= 1; rankOffset += 1) {
      if (fileOffset === 0 && rankOffset === 0) continue;
      const nextFile = file + fileOffset;
      const nextRank = rank + rankOffset;
      if (nextFile < 0 || nextFile > 7 || nextRank < 0 || nextRank > 7) continue;
      ring.push(`${String.fromCharCode(97 + nextFile)}${nextRank + 1}` as Square);
    }
  }
  return ring;
}

function moveDescriptor(move: Move): LosingMove {
  return {
    from: move.from,
    to: move.to,
    ...(move.promotion ? { promotion: move.promotion } : {}),
  };
}

function opposite(color: Color): Color {
  return color === "w" ? "b" : "w";
}

