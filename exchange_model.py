"""
Exchange Model — The Attention Exchange (AXP Protocol)
=======================================================
Novel market mechanism for trading attention as a liquid asset.
Uses a continuous double auction with token minting, decay,
and anti-manipulation safeguards.

This is the core PATENTABLE component.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import SimulationConfig
from attention_token import AttentionToken, TokenMinter, TokenLedger
from order_book import OrderBook, Side, Trade
from agents import (Agent, ContentProducer, ContentConsumer, Advertiser,
                    Speculator, MarketMaker, create_agent_population)


@dataclass
class ExchangeMetrics:
    """Per-timestep metrics from the exchange model."""
    timestep: int = 0
    price: float = 0.0
    spread: float = 0.0
    volume: int = 0
    active_supply: int = 0
    total_value: float = 0.0
    tokens_minted: int = 0
    tokens_expired: int = 0
    num_trades: int = 0
    circuit_breaker_triggered: bool = False


class AttentionExchangeModel:
    """
    The Attention Exchange Protocol (AXP).

    Core Innovation:
    1. Attention tokens are minted when users engage with content
    2. Tokens are traded on a continuous double auction
    3. Prices reflect true attention value (quality × scarcity)
    4. Anti-manipulation: position limits, circuit breakers, wash-trade detection
    5. Token decay ensures attention value is time-sensitive

    Flow:
    - Producers create content → consumers engage → tokens minted to producers
    - Producers sell tokens on exchange to monetize attention
    - Consumers/advertisers buy tokens to allocate attention budgets
    - Speculators provide liquidity and price discovery
    - Market maker ensures tight spreads
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

        # Core components
        self.order_book = OrderBook(
            tick_size=config.tick_size,
            circuit_breaker_pct=config.circuit_breaker_pct,
        )
        self.minter = TokenMinter(
            mint_rate=config.token_mint_rate,
            decay_rate=config.token_decay_rate,
        )
        self.ledger = TokenLedger(min_value=config.min_token_value)

        # Agents
        self.agents = create_agent_population(config, self.rng)

        # Metrics history
        self.metrics_history: List[ExchangeMetrics] = []

        # Anti-manipulation tracking
        self._agent_positions: Dict[int, int] = {}  # agent_id -> net position
        self._wash_trade_pairs: set = set()

        # Per-agent surplus tracking
        self.agent_surplus: Dict[int, float] = {}
        self.agent_revenue: Dict[int, float] = {}
        for aid in self.agents["all"]:
            self.agent_surplus[aid] = 0.0
            self.agent_revenue[aid] = 0.0

    def run(self) -> List[ExchangeMetrics]:
        """
        Run the full simulation.

        Returns
        -------
        List[ExchangeMetrics]
            Per-timestep metrics for the entire run.
        """
        # Initial token distribution: seed producers with tokens
        self._seed_initial_tokens()

        for t in range(self.config.num_timesteps):
            metrics = self._step(t)
            self.metrics_history.append(metrics)

        return self.metrics_history

    def _seed_initial_tokens(self):
        """Give each producer initial tokens proportional to quality."""
        for producer in self.agents["producers"]:
            tokens_count = int(
                self.config.initial_token_supply /
                self.config.num_producers * producer.quality
            ) + 5
            tokens = self.minter.mint(
                owner_id=producer.agent_id,
                engagement_score=tokens_count * 2.0,
                quality_score=producer.quality,
                current_time=0,
            )
            self.ledger.add_tokens(tokens)
            producer.state.tokens_held = len(tokens)
            self._agent_positions[producer.agent_id] = len(tokens)

        # Give market maker tokens and cash
        mm = self.agents["market_maker"]
        mm_tokens = self.minter.mint(
            owner_id=mm.agent_id,
            engagement_score=200,
            quality_score=0.5,
            current_time=0,
        )
        self.ledger.add_tokens(mm_tokens)
        mm.state.tokens_held = len(mm_tokens)
        self._agent_positions[mm.agent_id] = len(mm_tokens)

    def _step(self, t: int) -> ExchangeMetrics:
        """Execute one simulation timestep."""
        metrics = ExchangeMetrics(timestep=t)

        # 1. Content creation & engagement → mint new tokens
        tokens_minted = self._mint_phase(t)
        metrics.tokens_minted = tokens_minted

        # 2. Reset circuit breaker periodically
        if t % 50 == 0:
            self.order_book.reset_circuit_breaker()

        # 3. Market maker quotes
        self._market_maker_phase(t)

        # 4. Agent order submission (randomized order)
        all_agents = list(self.agents["all"].values())
        self.rng.shuffle(all_agents)

        for agent in all_agents:
            if isinstance(agent, MarketMaker):
                continue  # Already handled

            current_price = self.order_book.get_last_price()
            tokens_available = self.ledger.get_agent_token_count(
                agent.agent_id, t
            )

            order_params = agent.generate_order(
                current_price=current_price,
                timestamp=t,
                tokens_available=tokens_available,
                position_limit=self.config.position_limit,
            )

            if order_params is None:
                continue

            # Enforce position limits
            if not self._check_position_limit(
                agent.agent_id, order_params["side"], order_params["quantity"]
            ):
                continue

            # Submit order to book
            order, trades = self.order_book.submit_order(**order_params)

            # Process trades (settlement)
            for trade in trades:
                self._settle_trade(trade, t)

        # 5. Token decay & cleanup
        expired = self.ledger.cleanup_expired(t)
        metrics.tokens_expired = expired

        # 6. Record metrics
        metrics.price = self.order_book.get_last_price() or 0.0
        spread = self.order_book.get_spread()
        metrics.spread = spread if spread is not None else 0.0
        metrics.volume = sum(
            v for ts, v in self.order_book.volume_history if ts == t
        )
        metrics.active_supply = self.ledger.active_supply
        metrics.total_value = self.ledger.total_value(t)
        metrics.num_trades = len([
            trade for trade in self.order_book.trade_tape
            if trade.timestamp == t
        ])
        metrics.circuit_breaker_triggered = self.order_book._circuit_breaker_active

        return metrics

    def _mint_phase(self, t: int) -> int:
        """Simulate content creation and engagement, mint tokens to producers."""
        total_minted = 0
        for producer in self.agents["producers"]:
            # Engagement proportional to quality + randomness
            engagement = (producer.quality * 5.0 +
                         self.rng.exponential(1.0)) * self.config.producer_content_rate
            tokens = self.minter.mint(
                owner_id=producer.agent_id,
                engagement_score=engagement,
                quality_score=producer.quality,
                current_time=t,
            )
            self.ledger.add_tokens(tokens)
            producer.state.tokens_held += len(tokens)
            self._agent_positions[producer.agent_id] = \
                self._agent_positions.get(producer.agent_id, 0) + len(tokens)
            total_minted += len(tokens)
        return total_minted

    def _market_maker_phase(self, t: int):
        """Market maker provides liquidity."""
        mm = self.agents["market_maker"]
        current_price = self.order_book.get_last_price()
        tokens_avail = self.ledger.get_agent_token_count(mm.agent_id, t)
        quotes = mm.generate_quotes(current_price, t, tokens_avail)
        for q in quotes:
            order, trades = self.order_book.submit_order(**q)
            for trade in trades:
                self._settle_trade(trade, t)

    def _check_position_limit(self, agent_id: int, side: Side,
                               quantity: int) -> bool:
        """Enforce position limits (anti-manipulation)."""
        current = self._agent_positions.get(agent_id, 0)
        if side == Side.BID:
            return (current + quantity) <= self.config.position_limit
        return True  # No lower limit for selling

    def _settle_trade(self, trade: Trade, t: int):
        """Settle a trade: transfer tokens and cash."""
        buyer = self.agents["all"].get(trade.buyer_id)
        seller = self.agents["all"].get(trade.seller_id)

        if buyer is None or seller is None:
            return

        total_cost = trade.price * trade.quantity

        # Transfer cash
        if hasattr(buyer, 'state'):
            buyer.state.cash -= total_cost
            buyer.state.total_cost += total_cost
            buyer.state.trades_count += 1
        if hasattr(seller, 'state'):
            seller.state.cash += total_cost
            seller.state.total_revenue += total_cost
            seller.state.trades_count += 1

        # Transfer tokens
        transferred = self.ledger.transfer_tokens(
            trade.seller_id, trade.buyer_id, trade.quantity, t
        )

        # Update positions
        self._agent_positions[trade.buyer_id] = \
            self._agent_positions.get(trade.buyer_id, 0) + trade.quantity
        self._agent_positions[trade.seller_id] = \
            self._agent_positions.get(trade.seller_id, 0) - trade.quantity

        # Update token counts
        if hasattr(buyer, 'state'):
            buyer.state.tokens_held += trade.quantity
        if hasattr(seller, 'state'):
            seller.state.tokens_held -= trade.quantity

        # Compute per-agent surplus
        if isinstance(buyer, ContentConsumer):
            valuation = buyer.get_valuation(quality=0.5)
            surplus = (valuation - trade.price) * trade.quantity
            self.agent_surplus[trade.buyer_id] = \
                self.agent_surplus.get(trade.buyer_id, 0.0) + surplus
        elif isinstance(buyer, Advertiser):
            valuation = buyer.get_valuation() * 50
            surplus = (valuation - trade.price) * trade.quantity
            self.agent_surplus[trade.buyer_id] = \
                self.agent_surplus.get(trade.buyer_id, 0.0) + surplus

        if isinstance(seller, ContentProducer):
            surplus = (trade.price - seller.production_cost) * trade.quantity
            self.agent_surplus[trade.seller_id] = \
                self.agent_surplus.get(trade.seller_id, 0.0) + surplus
            self.agent_revenue[trade.seller_id] = \
                self.agent_revenue.get(trade.seller_id, 0.0) + total_cost

    def get_final_results(self) -> dict:
        """Compile final simulation results."""
        prices = [m.price for m in self.metrics_history if m.price > 0]
        spreads = [m.spread for m in self.metrics_history]
        volumes = [m.volume for m in self.metrics_history]

        producer_revenues = [
            self.agent_revenue.get(p.agent_id, 0.0)
            for p in self.agents["producers"]
        ]
        producer_surplus = [
            self.agent_surplus.get(p.agent_id, 0.0)
            for p in self.agents["producers"]
        ]
        consumer_surplus = [
            self.agent_surplus.get(c.agent_id, 0.0)
            for c in self.agents["consumers"]
        ]

        return {
            "model": "exchange",
            "num_timesteps": self.config.num_timesteps,
            "total_trades": self.order_book.total_trades,
            "total_volume": self.order_book.total_volume,
            "final_price": prices[-1] if prices else 0,
            "avg_price": np.mean(prices) if prices else 0,
            "price_std": np.std(prices) if prices else 0,
            "avg_spread": np.mean(spreads),
            "avg_volume": np.mean(volumes),
            "total_tokens_minted": self.minter.total_minted,
            "final_active_supply": self.ledger.active_supply,
            "total_expired": self.ledger.total_expired,
            "producer_revenues": producer_revenues,
            "producer_surplus": producer_surplus,
            "consumer_surplus": consumer_surplus,
            "total_producer_surplus": sum(producer_surplus),
            "total_consumer_surplus": sum(consumer_surplus),
            "total_surplus": sum(producer_surplus) + sum(consumer_surplus),
            "price_history": prices,
            "spread_history": spreads,
            "volume_history": volumes,
            "metrics_history": self.metrics_history,
            "agent_surplus": self.agent_surplus,
            "producer_qualities": [p.quality for p in self.agents["producers"]],
        }
