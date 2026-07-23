"""Agent interfaces and baseline implementations."""

from worst_chess.agents.adapters import SelfishLoserOpponentAgent
from worst_chess.agents.base import Agent, AgentError, MoveContext
from worst_chess.agents.exploit import (
    FrozenExploitWeights,
    FrozenTargetExploitOpponentAgent,
)
from worst_chess.agents.heuristic import GreedySacrificeAgent, HeuristicAgent
from worst_chess.agents.neural import NeuralAgent, PolicyMove
from worst_chess.agents.opponent_model import (
    OpportunisticHybridAgent,
    RandomReplyEvaluation,
    RandomReplySearchAgent,
    RandomReplyWeights,
    SampledExpectimaxConfig,
    StalemateAwareRandomReplySearchAgent,
    TwoTurnRandomReplyAgent,
)
from worst_chess.agents.policy_search import PolicyGuidedReverseSearchAgent
from worst_chess.agents.portfolio import RegimeSwitchingOpponentAgent
from worst_chess.agents.proof_hybrid import (
    ProofGuidedSelfmateAgent,
    ProofHybridStats,
    SelfmateBookEntry,
    SelfmateProofBook,
)
from worst_chess.agents.random import RandomAgent
from worst_chess.agents.resistant import ResistantOpponentAgent, ResistantWeights
from worst_chess.agents.rollout_search import NeuralShortlistRolloutAgent
from worst_chess.agents.stockfish import (
    LimitedStrengthStockfishAgent,
    ReverseMoveScore,
    ReverseStockfishAgent,
    StockfishAgent,
)
from worst_chess.agents.synthetic_loser import (
    ExploringLoserAgent,
    build_synthetic_loser_league,
)
from worst_chess.agents.tablebase import SyzygyLosingAgent, SyzygyMoveScore
from worst_chess.agents.weak import (
    CaptureFirstOpponentAgent,
    MaterialOpponentAgent,
    MaterialOpponentWeights,
    NoisyOpponentAgent,
)
from worst_chess.agents.web import WebEngineAgent

__all__ = [
    "Agent",
    "AgentError",
    "GreedySacrificeAgent",
    "FrozenExploitWeights",
    "FrozenTargetExploitOpponentAgent",
    "HeuristicAgent",
    "LimitedStrengthStockfishAgent",
    "MaterialOpponentAgent",
    "MaterialOpponentWeights",
    "MoveContext",
    "NeuralAgent",
    "NeuralShortlistRolloutAgent",
    "NoisyOpponentAgent",
    "OpportunisticHybridAgent",
    "PolicyGuidedReverseSearchAgent",
    "PolicyMove",
    "ProofGuidedSelfmateAgent",
    "ProofHybridStats",
    "RandomAgent",
    "RandomReplyEvaluation",
    "RandomReplySearchAgent",
    "RandomReplyWeights",
    "StalemateAwareRandomReplySearchAgent",
    "SampledExpectimaxConfig",
    "TwoTurnRandomReplyAgent",
    "ResistantOpponentAgent",
    "ResistantWeights",
    "RegimeSwitchingOpponentAgent",
    "ReverseMoveScore",
    "ReverseStockfishAgent",
    "StockfishAgent",
    "SyzygyLosingAgent",
    "SyzygyMoveScore",
    "SelfishLoserOpponentAgent",
    "SelfmateBookEntry",
    "SelfmateProofBook",
    "CaptureFirstOpponentAgent",
    "ExploringLoserAgent",
    "WebEngineAgent",
    "build_synthetic_loser_league",
]
