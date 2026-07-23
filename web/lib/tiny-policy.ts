import {
  type Chess,
  type Color,
  type Move,
  type PieceSymbol,
  type Square,
} from "chess.js";

const MAGIC = "SRPOLICY";
const PREFIX_BYTES = 12;
const BOARD_SQUARES = 64;
const ACTION_PLANES = 73;
const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;
const PIECE_PLANES: Record<PieceSymbol, number> = {
  p: 0,
  n: 1,
  b: 2,
  r: 3,
  q: 4,
  k: 5,
};
const SLIDING_DIRECTIONS = [
  [0, 1],
  [1, 1],
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, -1],
  [-1, 0],
  [-1, 1],
] as const;
const KNIGHT_DIRECTIONS = [
  [1, 2],
  [2, 1],
  [2, -1],
  [1, -2],
  [-1, -2],
  [-2, -1],
  [-2, 1],
  [-1, 2],
] as const;
const UNDERPROMOTIONS = ["n", "b", "r"] as const;

type TensorHeader = {
  length: number;
  name: string;
  offset: number;
  scale: number;
  shape: number[];
};

type PolicyHeader = {
  actionLayout: string;
  actionPlanes: number;
  actionSpaceSize: number;
  channels: number;
  format: string;
  observationShape: number[];
  orientation: string;
  payloadBytes: number;
  policyParameters: number;
  residualBlocks: number;
  tensors: TensorHeader[];
  version: number;
};

type PolicyTensor = {
  data: Float32Array;
  shape: number[];
};

export type TinyPolicy = {
  header: PolicyHeader;
  tensors: ReadonlyMap<string, PolicyTensor>;
};

export type TinyPolicyMove = {
  action: number;
  logit: number;
  move: Move;
};

export function decodeTinyPolicy(buffer: ArrayBuffer): TinyPolicy {
  if (buffer.byteLength < PREFIX_BYTES) {
    throw new Error("Tiny policy is truncated.");
  }
  const prefix = new Uint8Array(buffer, 0, 8);
  const magic = new TextDecoder().decode(prefix);
  if (magic !== MAGIC) {
    throw new Error("Tiny policy has an invalid magic header.");
  }

  const view = new DataView(buffer);
  const headerLength = view.getUint32(8, true);
  const payloadStart = PREFIX_BYTES + headerLength;
  if (payloadStart > buffer.byteLength) {
    throw new Error("Tiny policy header is truncated.");
  }

  let header: PolicyHeader;
  try {
    const encoded = new Uint8Array(buffer, PREFIX_BYTES, headerLength);
    header = JSON.parse(new TextDecoder().decode(encoded)) as PolicyHeader;
  } catch {
    throw new Error("Tiny policy header is invalid.");
  }
  validateHeader(header, buffer.byteLength - payloadStart);

  const quantized = new Int8Array(buffer, payloadStart);
  const tensors = new Map<string, PolicyTensor>();
  for (const tensor of header.tensors) {
    const count = tensor.shape.reduce((product, size) => product * size, 1);
    if (tensor.length !== count) {
      throw new Error(`Tiny policy tensor ${tensor.name} has an invalid length.`);
    }
    const values = new Float32Array(count);
    for (let index = 0; index < count; index += 1) {
      values[index] = quantized[tensor.offset + index] * tensor.scale;
    }
    tensors.set(tensor.name, { data: values, shape: tensor.shape });
  }
  return { header, tensors };
}

export function inferTinyPolicy(
  policy: TinyPolicy,
  observation: Float32Array,
): Float32Array {
  const expectedObservation = policy.header.observationShape.reduce(
    (product, size) => product * size,
    1,
  );
  if (observation.length !== expectedObservation) {
    throw new Error(
      `Tiny policy expected ${expectedObservation} observation values, got ${observation.length}.`,
    );
  }

  const stemWeight = requireTensor(policy, "stem.0.weight");
  const stemBias = requireTensor(policy, "stem.0.bias");
  let features = relu(
    conv8x8(observation, stemWeight, stemBias, 1),
  );

  for (let block = 0; block < policy.header.residualBlocks; block += 1) {
    const residual = features;
    const first = relu(
      conv8x8(
        features,
        requireTensor(policy, `residual_tower.${block}.conv1.weight`),
        requireTensor(policy, `residual_tower.${block}.conv1.bias`),
        1,
      ),
    );
    const second = conv8x8(
      first,
      requireTensor(policy, `residual_tower.${block}.conv2.weight`),
      requireTensor(policy, `residual_tower.${block}.conv2.bias`),
      1,
    );
    features = addRelu(residual, second);
  }

  const planes = conv8x8(
    features,
    requireTensor(policy, "policy_head.weight"),
    requireTensor(policy, "policy_head.bias"),
    0,
  );
  const logits = new Float32Array(policy.header.actionSpaceSize);
  for (let square = 0; square < BOARD_SQUARES; square += 1) {
    for (let plane = 0; plane < policy.header.actionPlanes; plane += 1) {
      logits[square * policy.header.actionPlanes + plane] =
        planes[plane * BOARD_SQUARES + square];
    }
  }
  return logits;
}

export function encodeTinyPolicyObservation(
  game: Chess,
  perspective: Color,
): Float32Array {
  const observation = new Float32Array(21 * BOARD_SQUARES);
  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const square = `${FILES[file]}${rank + 1}` as Square;
      const piece = game.get(square);
      if (!piece) continue;
      const ownerOffset = piece.color === perspective ? 0 : 6;
      const plane = ownerOffset + PIECE_PLANES[piece.type];
      const orientedRank = perspective === "w" ? rank : 7 - rank;
      observation[plane * BOARD_SQUARES + orientedRank * 8 + file] = 1;
    }
  }

  fillPlane(observation, 12, game.turn() === perspective);
  const ownCastling = game.getCastlingRights(perspective);
  const opponentCastling = game.getCastlingRights(opposite(perspective));
  fillPlane(observation, 13, ownCastling.k);
  fillPlane(observation, 14, ownCastling.q);
  fillPlane(observation, 15, opponentCastling.k);
  fillPlane(observation, 16, opponentCastling.q);

  const fenParts = game.fen({ forceEnpassantSquare: true }).split(" ");
  const enPassant = rawEnPassantSquare(game, fenParts[3]);
  if (enPassant && enPassant !== "-") {
    const file = enPassant.charCodeAt(0) - 97;
    const rank = Number(enPassant[1]) - 1;
    const orientedRank = perspective === "w" ? rank : 7 - rank;
    observation[17 * BOARD_SQUARES + orientedRank * 8 + file] = 1;
  }
  const halfmoveClock = Math.min(Number(fenParts[4]), 150) / 150;
  observation.fill(halfmoveClock, 18 * BOARD_SQUARES, 19 * BOARD_SQUARES);

  const repetitions = currentPositionCount(game);
  fillPlane(observation, 19, repetitions >= 2);
  fillPlane(observation, 20, repetitions >= 3);
  return observation;
}

export function encodeTinyPolicyMove(game: Chess, move: Move): number {
  const mirrorRanks = game.turn() === "b";
  const [fromFile, absoluteFromRank] = squareCoordinates(move.from);
  const [toFile, absoluteToRank] = squareCoordinates(move.to);
  const fromRank = mirrorRanks ? 7 - absoluteFromRank : absoluteFromRank;
  const toRank = mirrorRanks ? 7 - absoluteToRank : absoluteToRank;
  const fileDelta = toFile - fromFile;
  const rankDelta = toRank - fromRank;
  let plane: number;

  const underpromotion = UNDERPROMOTIONS.indexOf(
    move.promotion as (typeof UNDERPROMOTIONS)[number],
  );
  if (underpromotion >= 0) {
    if (rankDelta !== 1 || ![-1, 0, 1].includes(fileDelta)) {
      throw new Error(`Invalid tiny-policy underpromotion: ${move.lan}`);
    }
    plane = 64 + (fileDelta + 1) * 3 + underpromotion;
  } else {
    const knight = KNIGHT_DIRECTIONS.findIndex(
      ([file, rank]) => file === fileDelta && rank === rankDelta,
    );
    if (knight >= 0) {
      plane = 56 + knight;
    } else {
      const distance = Math.max(Math.abs(fileDelta), Math.abs(rankDelta));
      if (
        distance < 1 ||
        distance > 7 ||
        !(
          fileDelta === 0 ||
          rankDelta === 0 ||
          Math.abs(fileDelta) === Math.abs(rankDelta)
        )
      ) {
        throw new Error(`Move has no tiny-policy action plane: ${move.lan}`);
      }
      const fileStep = fileDelta / distance;
      const rankStep = rankDelta / distance;
      const direction = SLIDING_DIRECTIONS.findIndex(
        ([file, rank]) => file === fileStep && rank === rankStep,
      );
      if (direction < 0) {
        throw new Error(`Move has no tiny-policy direction: ${move.lan}`);
      }
      plane = direction * 7 + distance - 1;
    }
  }
  return (fromRank * 8 + fromFile) * ACTION_PLANES + plane;
}

export function rankTinyPolicyMoves(
  policy: TinyPolicy,
  game: Chess,
  targetColor: Color = game.turn(),
  topK?: number,
): TinyPolicyMove[] {
  if (game.turn() !== targetColor) {
    throw new Error("Tiny policy can only rank moves for its target color.");
  }
  const moves = game.moves({ verbose: true });
  if (moves.length === 0) {
    throw new Error("Tiny policy cannot rank a finished position.");
  }
  const logits = inferTinyPolicy(
    policy,
    encodeTinyPolicyObservation(game, targetColor),
  );
  const ranked = moves
    .map((move) => {
      const action = encodeTinyPolicyMove(game, move);
      return { action, logit: logits[action], move };
    })
    .sort((left, right) => right.logit - left.logit || left.action - right.action);
  return topK === undefined ? ranked : ranked.slice(0, Math.max(0, topK));
}

function validateHeader(header: PolicyHeader, availablePayload: number) {
  if (
    header.format !== "shallow-red.policy-int8" ||
    header.version !== 1 ||
    header.actionLayout !== "from_square*73+plane" ||
    header.actionPlanes !== 73 ||
    header.actionSpaceSize !== 4_672 ||
    header.orientation !== "perspective_vertical_mirror" ||
    header.observationShape.join(",") !== "21,8,8" ||
    !Number.isInteger(header.channels) ||
    header.channels < 1 ||
    !Number.isInteger(header.residualBlocks) ||
    header.residualBlocks < 1 ||
    header.payloadBytes !== availablePayload
  ) {
    throw new Error("Tiny policy header is unsupported.");
  }

  const names = new Set<string>();
  for (const tensor of header.tensors) {
    if (
      !tensor.name ||
      names.has(tensor.name) ||
      !Number.isFinite(tensor.scale) ||
      tensor.scale <= 0 ||
      !Number.isInteger(tensor.offset) ||
      !Number.isInteger(tensor.length) ||
      tensor.offset < 0 ||
      tensor.length < 1 ||
      tensor.offset + tensor.length > header.payloadBytes ||
      !Array.isArray(tensor.shape) ||
      tensor.shape.some((size) => !Number.isInteger(size) || size < 1)
    ) {
      throw new Error("Tiny policy tensor metadata is invalid.");
    }
    names.add(tensor.name);
  }
}

function requireTensor(policy: TinyPolicy, name: string): PolicyTensor {
  const tensor = policy.tensors.get(name);
  if (!tensor) throw new Error(`Tiny policy is missing tensor ${name}.`);
  return tensor;
}

function conv8x8(
  input: Float32Array,
  weights: PolicyTensor,
  bias: PolicyTensor,
  padding: 0 | 1,
): Float32Array {
  const [outputChannels, inputChannels, kernelRows, kernelColumns] =
    weights.shape;
  if (
    !outputChannels ||
    !inputChannels ||
    !kernelRows ||
    !kernelColumns ||
    kernelRows !== kernelColumns ||
    input.length !== inputChannels * BOARD_SQUARES ||
    bias.shape.length !== 1 ||
    bias.shape[0] !== outputChannels
  ) {
    throw new Error("Tiny policy convolution tensor shape is invalid.");
  }

  const output = new Float32Array(outputChannels * BOARD_SQUARES);
  for (let out = 0; out < outputChannels; out += 1) {
    const outputOffset = out * BOARD_SQUARES;
    const outputBias = bias.data[out];
    for (let row = 0; row < 8; row += 1) {
      for (let column = 0; column < 8; column += 1) {
        let sum = outputBias;
        for (let inputChannel = 0; inputChannel < inputChannels; inputChannel += 1) {
          const inputOffset = inputChannel * BOARD_SQUARES;
          const weightChannelOffset =
            (out * inputChannels + inputChannel) * kernelRows * kernelColumns;
          for (let kernelRow = 0; kernelRow < kernelRows; kernelRow += 1) {
            const sourceRow = row + kernelRow - padding;
            if (sourceRow < 0 || sourceRow >= 8) continue;
            for (let kernelColumn = 0; kernelColumn < kernelColumns; kernelColumn += 1) {
              const sourceColumn = column + kernelColumn - padding;
              if (sourceColumn < 0 || sourceColumn >= 8) continue;
              sum +=
                input[inputOffset + sourceRow * 8 + sourceColumn] *
                weights.data[
                  weightChannelOffset + kernelRow * kernelColumns + kernelColumn
                ];
            }
          }
        }
        output[outputOffset + row * 8 + column] = sum;
      }
    }
  }
  return output;
}

function relu(values: Float32Array): Float32Array {
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] < 0) values[index] = 0;
  }
  return values;
}

function addRelu(
  left: Float32Array,
  right: Float32Array,
): Float32Array {
  if (left.length !== right.length) {
    throw new Error("Tiny policy residual tensors do not align.");
  }
  const result = new Float32Array(left.length);
  for (let index = 0; index < left.length; index += 1) {
    result[index] = Math.max(0, left[index] + right[index]);
  }
  return result;
}

function fillPlane(
  observation: Float32Array,
  plane: number,
  enabled: boolean,
) {
  if (!enabled) return;
  observation.fill(1, plane * BOARD_SQUARES, (plane + 1) * BOARD_SQUARES);
}

function squareCoordinates(square: Square): [number, number] {
  return [square.charCodeAt(0) - 97, Number(square[1]) - 1];
}

function opposite(color: Color): Color {
  return color === "w" ? "b" : "w";
}

function currentPositionCount(game: Chess): number {
  const history = game.history({ verbose: true });
  const current = positionKey(game.fen());
  if (history.length === 0) return 1;
  let count = Number(positionKey(history[0].before) === current);
  for (const move of history) {
    count += Number(positionKey(move.after) === current);
  }
  return count;
}

function rawEnPassantSquare(game: Chess, fenSquare: string): string {
  if (fenSquare !== "-") return fenSquare;
  const lastMove = game.history({ verbose: true }).at(-1);
  if (
    !lastMove ||
    lastMove.piece !== "p" ||
    Math.abs(Number(lastMove.to[1]) - Number(lastMove.from[1])) !== 2
  ) {
    return "-";
  }
  const middleRank = (Number(lastMove.to[1]) + Number(lastMove.from[1])) / 2;
  return `${lastMove.from[0]}${middleRank}`;
}

function positionKey(fen: string) {
  return fen.split(" ").slice(0, 4).join(" ");
}
