"""Agent interfaces and baseline implementations."""

from worst_chess.agents.base import Agent, AgentError, MoveContext
from worst_chess.agents.heuristic import GreedySacrificeAgent, HeuristicAgent
from worst_chess.agents.neural import NeuralAgent, PolicyMove
from worst_chess.agents.policy_search import PolicyGuidedReverseSearchAgent
from worst_chess.agents.random import RandomAgent
from worst_chess.agents.resistant import ResistantOpponentAgent, ResistantWeights
from worst_chess.agents.stockfish import (
    ReverseMoveScore,
    ReverseStockfishAgent,
    StockfishAgent,
)

__all__ = [
    "Agent",
    "AgentError",
    "GreedySacrificeAgent",
    "HeuristicAgent",
    "MoveContext",
    "NeuralAgent",
    "PolicyGuidedReverseSearchAgent",
    "PolicyMove",
    "RandomAgent",
    "ResistantOpponentAgent",
    "ResistantWeights",
    "ReverseMoveScore",
    "ReverseStockfishAgent",
    "StockfishAgent",
]
