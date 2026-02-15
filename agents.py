"""
Agents — Heterogeneous Market Participants
============================================
Defines agent types for both the Attention Exchange and Ad-Driven models:
ContentProducer, ContentConsumer, Advertiser, Speculator, MarketMaker.
Each agent has behavioral strategies: ZI, ZIP, or RL (Q-learning).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum

from order_book import Side


class AgentType(Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    ADVERTISER = "advertiser"
    SPECULATOR = "speculator"
    MARKET_MAKER = "market_maker"


class Strategy(Enum):
    ZI = "zero_intelligence"           # Random within budget
    ZIP = "zero_intelligence_plus"     # Adaptive ZI with learning
    RL = "reinforcement_learning"      # Q-learning


@dataclass
class AgentState:
    """Tracks an agent's economic state across the simulation."""
    cash: float = 0.0
    tokens_held: int = 0
    total_revenue: float = 0.0
    total_cost: float = 0.0
    total_surplus: float = 0.0
    trades_count: int = 0
    attention_received: float = 0.0
    attention_spent: float = 0.0


class Agent:
    """Base class for all agent types."""

    def __init__(self, agent_id: int, agent_type: AgentType,
                 strategy: Strategy = Strategy.ZI, rng: np.random.RandomState = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.strategy = strategy
        self.state = AgentState()
        self.rng = rng if rng is not None else np.random.RandomState()

        # ZIP learning parameters
        self._zip_margin = 0.05
        self._zip_momentum = 0.0
        self._zip_learning_rate = 0.1
        self._zip_beta = 0.3

        # RL Q-table (discretized)
        self._q_table: Dict[tuple, Dict[str, float]] = {}
        self._rl_epsilon = 0.2
        self._rl_alpha = 0.1
        self._rl_gamma = 0.95
        self._rl_last_state = None
        self._rl_last_action = None

    def get_valuation(self, **kwargs) -> float:
        """Override: agent's private valuation for attention."""
        raise NotImplementedError

    def generate_order(self, current_price: Optional[float],
                       timestamp: int, **kwargs) -> Optional[dict]:
        """Override: generate a buy/sell order."""
        raise NotImplementedError


class ContentProducer(Agent):
    """
    Produces content and earns attention tokens.
    In the exchange model: sells attention tokens on the exchange.
    In the ad model: earns revenue from platform ad-revenue sharing.
    """

    def __init__(self, agent_id: int, quality: float, production_cost: float,
                 strategy: Strategy = Strategy.ZIP, rng: np.random.RandomState = None):
        super().__init__(agent_id, AgentType.PRODUCER, strategy, rng)
        self.quality = quality               # Content quality [0, 1]
        self.production_cost = production_cost
        self.state.cash = 0.0
        # Reserve price: won't sell below cost
        self._reserve_price = production_cost * 0.8

    def get_valuation(self, **kwargs) -> float:
        return self.quality * 2.0  # Valuation proportional to quality

    def generate_order(self, current_price: Optional[float],
                       timestamp: int, **kwargs) -> Optional[dict]:
        """Generate a sell (ASK) order for attention tokens."""
        tokens_available = kwargs.get("tokens_available", 0)
        if tokens_available <= 0:
            return None

        if self.strategy == Strategy.ZI:
            price = self._reserve_price + self.rng.uniform(0, 1.0)
        elif self.strategy == Strategy.ZIP:
            price = self._zip_ask_price(current_price)
        else:  # RL
            price = self._rl_price(current_price, Side.ASK)

        quantity = min(tokens_available, max(1, int(tokens_available * 0.3)))
        return {
            "agent_id": self.agent_id,
            "side": Side.ASK,
            "price": max(price, self.tick_size if hasattr(self, 'tick_size') else 0.01),
            "quantity": quantity,
            "timestamp": timestamp,
        }

    def _zip_ask_price(self, current_price: Optional[float]) -> float:
        """ZIP strategy for asks: adapt margin based on market."""
        target = current_price if current_price else self._reserve_price * 1.2
        error = target - (self._reserve_price * (1 + self._zip_margin))
        self._zip_momentum = self._zip_beta * self._zip_momentum + \
                             (1 - self._zip_beta) * error
        self._zip_margin += self._zip_learning_rate * self._zip_momentum
        self._zip_margin = max(0.01, min(self._zip_margin, 1.0))
        return self._reserve_price * (1 + self._zip_margin)

    def _rl_price(self, current_price: Optional[float], side: Side) -> float:
        """Simple Q-learning pricing."""
        state = self._discretize_state(current_price)
        action = self._rl_select_action(state)
        self._rl_last_state = state
        self._rl_last_action = action
        base = current_price if current_price else self._reserve_price
        # Actions: 5 price levels relative to base
        offsets = {0: -0.2, 1: -0.1, 2: 0.0, 3: 0.1, 4: 0.2}
        return max(self._reserve_price, base * (1 + offsets.get(action, 0)))

    def _discretize_state(self, price: Optional[float]) -> tuple:
        if price is None:
            return (0, 0)
        p_bucket = int(price * 10) % 20
        t_bucket = self.state.tokens_held // 5
        return (p_bucket, min(t_bucket, 10))

    def _rl_select_action(self, state: tuple) -> int:
        if self.rng.random() < self._rl_epsilon:
            return self.rng.randint(0, 5)
        if state not in self._q_table:
            self._q_table[state] = {i: 0.0 for i in range(5)}
        return max(self._q_table[state], key=self._q_table[state].get)

    def update_rl(self, reward: float, new_price: Optional[float]):
        """Update Q-table after trade result."""
        if self._rl_last_state is None:
            return
        new_state = self._discretize_state(new_price)
        if self._rl_last_state not in self._q_table:
            self._q_table[self._rl_last_state] = {i: 0.0 for i in range(5)}
        if new_state not in self._q_table:
            self._q_table[new_state] = {i: 0.0 for i in range(5)}
        old_q = self._q_table[self._rl_last_state][self._rl_last_action]
        max_future = max(self._q_table[new_state].values())
        new_q = old_q + self._rl_alpha * (reward + self._rl_gamma * max_future - old_q)
        self._q_table[self._rl_last_state][self._rl_last_action] = new_q


class ContentConsumer(Agent):
    """
    Consumes content and allocates attention.
    In the exchange model: buys attention tokens to allocate to preferred content.
    In the ad model: attention is allocated by the platform (ad auction winner).
    """

    def __init__(self, agent_id: int, attention_budget: float,
                 quality_sensitivity: float = 0.7,
                 price_sensitivity: float = 0.3,
                 strategy: Strategy = Strategy.ZIP, rng: np.random.RandomState = None):
        super().__init__(agent_id, AgentType.CONSUMER, strategy, rng)
        self.attention_budget = attention_budget
        self.quality_sensitivity = quality_sensitivity
        self.price_sensitivity = price_sensitivity
        self.state.cash = attention_budget * 2.0  # Starting cash
        self._max_willingness_to_pay = attention_budget * 0.3

    def get_valuation(self, quality: float = 0.5, **kwargs) -> float:
        """Valuation = quality-weighted willingness to pay."""
        return self._max_willingness_to_pay * (
            self.quality_sensitivity * quality +
            self.price_sensitivity * (1 - quality) * 0.5
        )

    def generate_order(self, current_price: Optional[float],
                       timestamp: int, **kwargs) -> Optional[dict]:
        """Generate a buy (BID) order for attention tokens."""
        quality = kwargs.get("quality", 0.5)
        valuation = self.get_valuation(quality=quality)

        if self.state.cash < valuation * 0.1:
            return None

        if self.strategy == Strategy.ZI:
            price = self.rng.uniform(0.1, valuation)
        elif self.strategy == Strategy.ZIP:
            price = self._zip_bid_price(current_price, valuation)
        else:
            price = self._rl_price(current_price, Side.BID)

        quantity = max(1, int(self.attention_budget * 0.1))
        price = min(price, self.state.cash / max(quantity, 1))

        return {
            "agent_id": self.agent_id,
            "side": Side.BID,
            "price": max(price, 0.01),
            "quantity": quantity,
            "timestamp": timestamp,
        }

    def _zip_bid_price(self, current_price: Optional[float],
                       valuation: float) -> float:
        target = current_price if current_price else valuation * 0.8
        error = target - (valuation * (1 - self._zip_margin))
        self._zip_momentum = self._zip_beta * self._zip_momentum + \
                             (1 - self._zip_beta) * error
        self._zip_margin += self._zip_learning_rate * self._zip_momentum
        self._zip_margin = max(0.01, min(self._zip_margin, 0.5))
        return valuation * (1 - self._zip_margin)

    def _rl_price(self, current_price: Optional[float], side: Side) -> float:
        state = self._discretize_state(current_price)
        action = self._rl_select_action(state)
        self._rl_last_state = state
        self._rl_last_action = action
        base = current_price if current_price else self._max_willingness_to_pay * 0.5
        offsets = {0: -0.2, 1: -0.1, 2: 0.0, 3: 0.1, 4: 0.2}
        return max(0.01, base * (1 + offsets.get(action, 0)))

    def _discretize_state(self, price: Optional[float]) -> tuple:
        if price is None:
            return (0, 0)
        p_bucket = int(price * 10) % 20
        cash_bucket = int(self.state.cash) // 5
        return (p_bucket, min(cash_bucket, 10))

    def _rl_select_action(self, state: tuple) -> int:
        if self.rng.random() < self._rl_epsilon:
            return self.rng.randint(0, 5)
        if state not in self._q_table:
            self._q_table[state] = {i: 0.0 for i in range(5)}
        return max(self._q_table[state], key=self._q_table[state].get)

    def compute_satisfaction(self, allocated_quality: float,
                             price_paid: float) -> float:
        """
        User satisfaction = quality-weighted utility minus price.
        Higher quality content → higher satisfaction.
        """
        utility = self.quality_sensitivity * allocated_quality * self.attention_budget
        return utility - price_paid


class Advertiser(Agent):
    """
    Buys attention/ad slots to reach consumers.
    In the exchange model: buys attention tokens directly.
    In the ad model: bids in second-price ad auctions.
    """

    def __init__(self, agent_id: int, budget: float,
                 valuation_per_click: float = 0.5,
                 strategy: Strategy = Strategy.ZIP, rng: np.random.RandomState = None):
        super().__init__(agent_id, AgentType.ADVERTISER, strategy, rng)
        self.budget = budget
        self.valuation_per_click = valuation_per_click
        self.state.cash = budget
        self._target_ctr = 0.02

    def get_valuation(self, ctr: float = 0.02, **kwargs) -> float:
        """Valuation = expected value per impression."""
        return self.valuation_per_click * ctr

    def generate_ad_bid(self, quality_score: float = 1.0,
                        timestamp: int = 0) -> Optional[dict]:
        """Generate a bid for the ad auction (ad model)."""
        if self.state.cash <= 0:
            return None
        # Bid = valuation * quality_score + noise
        valuation = self.get_valuation(ctr=self._target_ctr)
        bid = valuation * quality_score * (1 + self.rng.uniform(-0.2, 0.2))
        bid = min(bid, self.state.cash)
        return {
            "agent_id": self.agent_id,
            "bid": max(bid, 0.001),
            "quality_score": quality_score,
            "timestamp": timestamp,
        }

    def generate_order(self, current_price: Optional[float],
                       timestamp: int, **kwargs) -> Optional[dict]:
        """Generate a buy order on the attention exchange."""
        if self.state.cash <= 0:
            return None
        valuation = self.get_valuation(ctr=self._target_ctr) * 50  # Scale up
        if self.strategy == Strategy.ZI:
            price = self.rng.uniform(0.1, valuation)
        else:
            price = valuation * 0.8 if current_price is None else \
                    min(current_price * 1.05, valuation)

        quantity = max(1, int(self.state.cash / max(price, 0.01) * 0.1))
        quantity = min(quantity, 10)

        return {
            "agent_id": self.agent_id,
            "side": Side.BID,
            "price": max(price, 0.01),
            "quantity": quantity,
            "timestamp": timestamp,
        }


class Speculator(Agent):
    """
    Trades attention tokens for profit (exchange model only).
    Uses momentum and mean-reversion strategies.
    """

    def __init__(self, agent_id: int, capital: float,
                 strategy: Strategy = Strategy.ZIP, rng: np.random.RandomState = None):
        super().__init__(agent_id, AgentType.SPECULATOR, strategy, rng)
        self.state.cash = capital
        self.capital = capital
        self._price_history: List[float] = []
        self._position = 0
        self._entry_price = 0.0

    def generate_order(self, current_price: Optional[float],
                       timestamp: int, **kwargs) -> Optional[dict]:
        """Generate speculative order based on price momentum."""
        position_limit = kwargs.get("position_limit", 500)

        if current_price is None:
            return None

        self._price_history.append(current_price)

        if len(self._price_history) < 5:
            return None

        # Simple momentum: compare short MA vs long MA
        short_ma = np.mean(self._price_history[-5:])
        long_ma = np.mean(self._price_history[-min(20, len(self._price_history)):])

        if short_ma > long_ma * 1.02:  # Uptrend → buy
            if self._position < position_limit and self.state.cash > current_price:
                price = current_price * (1 + self.rng.uniform(0, 0.05))
                qty = min(5, int(self.state.cash / price * 0.1))
                if qty > 0:
                    return {
                        "agent_id": self.agent_id,
                        "side": Side.BID,
                        "price": price,
                        "quantity": qty,
                        "timestamp": timestamp,
                    }

        elif short_ma < long_ma * 0.98:  # Downtrend → sell
            tokens_held = kwargs.get("tokens_available", 0)
            if tokens_held > 0:
                price = current_price * (1 - self.rng.uniform(0, 0.05))
                qty = min(5, tokens_held)
                return {
                    "agent_id": self.agent_id,
                    "side": Side.ASK,
                    "price": max(price, 0.01),
                    "quantity": qty,
                    "timestamp": timestamp,
                }

        return None


class MarketMaker(Agent):
    """
    Provides liquidity by continuously quoting bid and ask prices.
    Earns the bid-ask spread.
    """

    def __init__(self, agent_id: int, capital: float, spread: float = 0.05,
                 depth: int = 50, rng: np.random.RandomState = None):
        super().__init__(agent_id, AgentType.MARKET_MAKER, Strategy.ZI, rng)
        self.state.cash = capital
        self.spread = spread
        self.depth = depth
        self._fair_value = 1.0

    def generate_quotes(self, current_price: Optional[float],
                        timestamp: int, tokens_available: int = 100) -> List[dict]:
        """Generate both bid and ask quotes around fair value."""
        if current_price is not None:
            # Exponential moving average of fair value
            self._fair_value = 0.9 * self._fair_value + 0.1 * current_price

        half_spread = self.spread / 2.0
        bid_price = self._fair_value * (1 - half_spread)
        ask_price = self._fair_value * (1 + half_spread)

        orders = []
        bid_qty = min(self.depth, int(self.state.cash / max(bid_price, 0.01) * 0.5))
        if bid_qty > 0:
            orders.append({
                "agent_id": self.agent_id,
                "side": Side.BID,
                "price": max(bid_price, 0.01),
                "quantity": bid_qty,
                "timestamp": timestamp,
            })

        ask_qty = min(self.depth, tokens_available)
        if ask_qty > 0:
            orders.append({
                "agent_id": self.agent_id,
                "side": Side.ASK,
                "price": max(ask_price, 0.01),
                "quantity": ask_qty,
                "timestamp": timestamp,
            })

        return orders


# ── Agent Factory ──────────────────────────────────────────────────────

def create_agent_population(config, rng: np.random.RandomState = None):
    """
    Create the full agent population from config.

    Returns
    -------
    dict with keys: 'producers', 'consumers', 'advertisers',
                    'speculators', 'market_maker', 'all'
    """
    if rng is None:
        rng = np.random.RandomState(config.random_seed)

    agents = {"producers": [], "consumers": [], "advertisers": [],
              "speculators": [], "market_maker": None, "all": {}}
    agent_id = 0

    # Content Producers
    qualities = rng.beta(config.quality_alpha, config.quality_beta,
                         size=config.num_producers)
    costs = np.abs(rng.normal(config.producer_cost_mean, config.producer_cost_std,
                              size=config.num_producers))
    strategies = [Strategy.ZIP] * int(config.num_producers * 0.6) + \
                 [Strategy.ZI] * int(config.num_producers * 0.2) + \
                 [Strategy.RL] * (config.num_producers - int(config.num_producers * 0.6)
                                  - int(config.num_producers * 0.2))
    rng.shuffle(strategies)

    for i in range(config.num_producers):
        agent = ContentProducer(
            agent_id=agent_id,
            quality=float(qualities[i]),
            production_cost=float(costs[i]),
            strategy=strategies[i],
            rng=np.random.RandomState(rng.randint(0, 2**31)),
        )
        agents["producers"].append(agent)
        agents["all"][agent_id] = agent
        agent_id += 1

    # Content Consumers
    budgets = np.abs(rng.normal(config.consumer_attention_budget_mean,
                                config.consumer_attention_budget_std,
                                size=config.num_consumers))
    for i in range(config.num_consumers):
        agent = ContentConsumer(
            agent_id=agent_id,
            attention_budget=float(budgets[i]),
            quality_sensitivity=config.consumer_quality_sensitivity,
            price_sensitivity=config.consumer_price_sensitivity,
            strategy=Strategy.ZIP,
            rng=np.random.RandomState(rng.randint(0, 2**31)),
        )
        agents["consumers"].append(agent)
        agents["all"][agent_id] = agent
        agent_id += 1

    # Advertisers
    ad_budgets = np.abs(rng.normal(config.advertiser_budget_mean,
                                   config.advertiser_budget_std,
                                   size=config.num_advertisers))
    for i in range(config.num_advertisers):
        agent = Advertiser(
            agent_id=agent_id,
            budget=float(ad_budgets[i]),
            valuation_per_click=config.advertiser_valuation_per_click,
            strategy=Strategy.ZIP,
            rng=np.random.RandomState(rng.randint(0, 2**31)),
        )
        agents["advertisers"].append(agent)
        agents["all"][agent_id] = agent
        agent_id += 1

    # Speculators (exchange model only)
    for i in range(config.num_speculators):
        capital = float(rng.uniform(20, 100))
        agent = Speculator(
            agent_id=agent_id,
            capital=capital,
            strategy=Strategy.ZIP,
            rng=np.random.RandomState(rng.randint(0, 2**31)),
        )
        agents["speculators"].append(agent)
        agents["all"][agent_id] = agent
        agent_id += 1

    # Market Maker
    mm = MarketMaker(
        agent_id=agent_id,
        capital=5000.0,
        spread=config.market_maker_spread,
        depth=config.market_maker_depth,
        rng=np.random.RandomState(rng.randint(0, 2**31)),
    )
    agents["market_maker"] = mm
    agents["all"][agent_id] = mm

    return agents
