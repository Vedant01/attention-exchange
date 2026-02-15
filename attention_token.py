"""
Attention Token — The Economic Primitive
=========================================
Defines the AttentionToken class: a fungible, depreciating unit of
tokenized human attention with minting rules and decay dynamics.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttentionToken:
    """
    A single unit of tokenized attention.

    Properties
    ----------
    token_id : int
        Unique identifier.
    initial_value : float
        Value at minting (based on engagement quality).
    timestamp_minted : int
        Simulation timestep when minted.
    owner_id : int
        Current owner's agent ID.
    decay_rate : float
        λ in the exponential decay function v(t) = v₀ × e^(-λt).
    quality_score : float
        Quality of the engagement that generated this token [0, 1].
    """

    token_id: int
    initial_value: float
    timestamp_minted: int
    owner_id: int
    decay_rate: float = 0.005
    quality_score: float = 0.5
    is_expired: bool = False

    def current_value(self, current_time: int) -> float:
        """
        Compute the current value of the token with exponential decay.
        v(t) = v₀ × e^(-λ × Δt) × quality_score
        """
        if self.is_expired:
            return 0.0
        dt = max(0, current_time - self.timestamp_minted)
        value = self.initial_value * math.exp(-self.decay_rate * dt)
        return max(value, 0.0)

    def has_expired(self, current_time: int, min_value: float = 0.01) -> bool:
        """Check if token value has decayed below the minimum threshold."""
        if self.is_expired:
            return True
        if self.current_value(current_time) < min_value:
            self.is_expired = True
            return True
        return False

    def transfer(self, new_owner_id: int):
        """Transfer token ownership."""
        self.owner_id = new_owner_id


class TokenMinter:
    """
    Mints new AttentionTokens based on verified engagement.

    Minting Formula
    ---------------
    tokens_minted = floor(engagement_score × mint_rate × quality_multiplier)

    where quality_multiplier = 1 + quality_bonus for high-quality engagement
    """

    def __init__(self, mint_rate: float = 0.1, decay_rate: float = 0.005,
                 quality_bonus: float = 0.5):
        self.mint_rate = mint_rate
        self.decay_rate = decay_rate
        self.quality_bonus = quality_bonus
        self._next_token_id = 0

    def mint(self, owner_id: int, engagement_score: float,
             quality_score: float, current_time: int) -> list:
        """
        Mint tokens for verified engagement.

        Parameters
        ----------
        owner_id : int
            Agent who earned the attention.
        engagement_score : float
            Raw engagement metric (e.g., time spent, interactions).
        quality_score : float
            Quality of the content/interaction [0, 1].
        current_time : int
            Current simulation timestep.

        Returns
        -------
        list[AttentionToken]
            Newly minted tokens.
        """
        quality_multiplier = 1.0 + self.quality_bonus * quality_score
        num_tokens = max(1, int(engagement_score * self.mint_rate * quality_multiplier))
        token_value = engagement_score * quality_score / max(num_tokens, 1)

        tokens = []
        for _ in range(num_tokens):
            token = AttentionToken(
                token_id=self._next_token_id,
                initial_value=max(token_value, 0.01),
                timestamp_minted=current_time,
                owner_id=owner_id,
                decay_rate=self.decay_rate,
                quality_score=quality_score,
            )
            tokens.append(token)
            self._next_token_id += 1

        return tokens

    @property
    def total_minted(self) -> int:
        return self._next_token_id


class TokenLedger:
    """
    Tracks all tokens in the system: active supply, ownership, and expiry.
    """

    def __init__(self, min_value: float = 0.01):
        self.tokens: dict[int, AttentionToken] = {}
        self.min_value = min_value
        self._total_expired = 0

    def add_tokens(self, tokens: list):
        """Register newly minted tokens."""
        for t in tokens:
            self.tokens[t.token_id] = t

    def get_agent_tokens(self, agent_id: int, current_time: int) -> list:
        """Get all active tokens owned by an agent."""
        return [
            t for t in self.tokens.values()
            if t.owner_id == agent_id and not t.has_expired(current_time, self.min_value)
        ]

    def get_agent_balance(self, agent_id: int, current_time: int) -> float:
        """Get total token value for an agent."""
        return sum(t.current_value(current_time)
                   for t in self.get_agent_tokens(agent_id, current_time))

    def get_agent_token_count(self, agent_id: int, current_time: int) -> int:
        """Get number of active tokens for an agent."""
        return len(self.get_agent_tokens(agent_id, current_time))

    def transfer_tokens(self, from_id: int, to_id: int,
                        quantity: int, current_time: int) -> list:
        """Transfer tokens between agents. Returns transferred tokens."""
        available = self.get_agent_tokens(from_id, current_time)
        # Sort by value descending — transfer highest-value tokens first
        available.sort(key=lambda t: t.current_value(current_time), reverse=True)
        transferred = []
        for t in available[:quantity]:
            t.transfer(to_id)
            transferred.append(t)
        return transferred

    def cleanup_expired(self, current_time: int) -> int:
        """Remove expired tokens from the ledger. Returns count removed."""
        expired_ids = [
            tid for tid, t in self.tokens.items()
            if t.has_expired(current_time, self.min_value)
        ]
        for tid in expired_ids:
            del self.tokens[tid]
        self._total_expired += len(expired_ids)
        return len(expired_ids)

    @property
    def active_supply(self) -> int:
        return len(self.tokens)

    @property
    def total_expired(self) -> int:
        return self._total_expired

    def total_value(self, current_time: int) -> float:
        """Total value of all active tokens."""
        return sum(t.current_value(current_time) for t in self.tokens.values()
                   if not t.is_expired)
