"""
Order Book — Continuous Double Auction Engine
==============================================
Implements a price-time priority limit order book for trading
attention tokens via a continuous double auction (CDA).
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class Side(Enum):
    BID = "BID"
    ASK = "ASK"


@dataclass
class Order:
    """A single limit order in the order book."""
    order_id: int
    agent_id: int
    side: Side
    price: float
    quantity: int
    timestamp: int
    remaining: int = -1  # -1 means not yet initialized

    def __post_init__(self):
        if self.remaining == -1:
            self.remaining = self.quantity

    @property
    def is_filled(self) -> bool:
        return self.remaining <= 0

    def __lt__(self, other):
        return self.order_id < other.order_id

    def __le__(self, other):
        return self.order_id <= other.order_id


@dataclass
class Trade:
    """A completed trade between a buyer and seller."""
    trade_id: int
    buyer_id: int
    seller_id: int
    price: float
    quantity: int
    timestamp: int
    buyer_order_id: int
    seller_order_id: int


class OrderBook:
    """
    Continuous Double Auction (CDA) Order Book.

    Implements price-time priority matching with the following features:
    - Limit orders only (market orders simulated via aggressive pricing)
    - Price-time priority: best price first, then earliest timestamp
    - Partial fills supported
    - Trade tape recording
    - Spread, depth, and midprice analytics

    Anti-Manipulation Features:
    - Circuit breaker: halts trading if price moves > threshold
    - Position limits: enforced externally by the exchange model
    """

    def __init__(self, tick_size: float = 0.01,
                 circuit_breaker_pct: float = 0.15):
        # Bids: max-heap (negate price for heapq min-heap)
        # Format: (-price, timestamp, order)
        self._bids: list = []
        # Asks: min-heap
        # Format: (price, timestamp, order)
        self._asks: list = []

        self.tick_size = tick_size
        self.circuit_breaker_pct = circuit_breaker_pct

        self._trade_tape: List[Trade] = []
        self._next_trade_id = 0
        self._next_order_id = 0
        self._reference_price: Optional[float] = None
        self._circuit_breaker_active = False

        # Analytics
        self._price_history: List[Tuple[int, float]] = []
        self._spread_history: List[Tuple[int, float]] = []
        self._volume_history: List[Tuple[int, int]] = []

    def submit_order(self, agent_id: int, side: Side, price: float,
                     quantity: int, timestamp: int) -> Tuple[Order, List[Trade]]:
        """
        Submit a limit order and attempt to match.

        Returns
        -------
        (order, trades) : Tuple[Order, List[Trade]]
            The submitted order and any resulting trades.
        """
        # Round price to tick size
        price = round(price / self.tick_size) * self.tick_size
        price = max(price, self.tick_size)

        order = Order(
            order_id=self._next_order_id,
            agent_id=agent_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )
        self._next_order_id += 1

        # Check circuit breaker
        if self._circuit_breaker_active:
            # Still add to book but don't match
            self._insert_order(order)
            return order, []

        # Try to match
        trades = self._match(order, timestamp)

        # If order not fully filled, add remainder to book
        if not order.is_filled:
            self._insert_order(order)

        # Update analytics
        if trades:
            last_price = trades[-1].price
            self._price_history.append((timestamp, last_price))
            total_vol = sum(t.quantity for t in trades)
            self._volume_history.append((timestamp, total_vol))

            # Check circuit breaker
            if self._reference_price is not None:
                pct_change = abs(last_price - self._reference_price) / self._reference_price
                if pct_change > self.circuit_breaker_pct:
                    self._circuit_breaker_active = True

            self._reference_price = last_price

        # Record spread
        spread = self.get_spread()
        if spread is not None:
            self._spread_history.append((timestamp, spread))

        return order, trades

    def _insert_order(self, order: Order):
        """Insert order into the appropriate side of the book."""
        if order.side == Side.BID:
            heapq.heappush(self._bids, (-order.price, order.timestamp, order))
        else:
            heapq.heappush(self._asks, (order.price, order.timestamp, order))

    def _match(self, incoming: Order, timestamp: int) -> List[Trade]:
        """Match an incoming order against the book."""
        trades = []

        if incoming.side == Side.BID:
            # Match against asks (ascending price)
            while self._asks and not incoming.is_filled:
                ask_price, ask_ts, ask_order = self._asks[0]
                if ask_order.is_filled:
                    heapq.heappop(self._asks)
                    continue
                # Check price crossing
                if incoming.price >= ask_price:
                    # Trade at the resting order's price (price-time priority)
                    trade_price = ask_price
                    trade_qty = min(incoming.remaining, ask_order.remaining)
                    trade = Trade(
                        trade_id=self._next_trade_id,
                        buyer_id=incoming.agent_id,
                        seller_id=ask_order.agent_id,
                        price=trade_price,
                        quantity=trade_qty,
                        timestamp=timestamp,
                        buyer_order_id=incoming.order_id,
                        seller_order_id=ask_order.order_id,
                    )
                    self._next_trade_id += 1
                    trades.append(trade)
                    self._trade_tape.append(trade)

                    incoming.remaining -= trade_qty
                    ask_order.remaining -= trade_qty

                    if ask_order.is_filled:
                        heapq.heappop(self._asks)
                else:
                    break

        else:  # ASK
            # Match against bids (descending price)
            while self._bids and not incoming.is_filled:
                neg_bid_price, bid_ts, bid_order = self._bids[0]
                bid_price = -neg_bid_price
                if bid_order.is_filled:
                    heapq.heappop(self._bids)
                    continue
                if incoming.price <= bid_price:
                    trade_price = bid_price
                    trade_qty = min(incoming.remaining, bid_order.remaining)
                    trade = Trade(
                        trade_id=self._next_trade_id,
                        buyer_id=bid_order.agent_id,
                        seller_id=incoming.agent_id,
                        price=trade_price,
                        quantity=trade_qty,
                        timestamp=timestamp,
                        buyer_order_id=bid_order.order_id,
                        seller_order_id=incoming.order_id,
                    )
                    self._next_trade_id += 1
                    trades.append(trade)
                    self._trade_tape.append(trade)

                    incoming.remaining -= trade_qty
                    bid_order.remaining -= trade_qty

                    if bid_order.is_filled:
                        heapq.heappop(self._bids)
                else:
                    break

        return trades

    def reset_circuit_breaker(self):
        """Reset the circuit breaker (called at start of new period)."""
        self._circuit_breaker_active = False
        if self._price_history:
            self._reference_price = self._price_history[-1][1]

    # ── Analytics ──────────────────────────────────────────────────────

    def get_best_bid(self) -> Optional[float]:
        """Best (highest) bid price."""
        self._clean_bids()
        if self._bids:
            return -self._bids[0][0]
        return None

    def get_best_ask(self) -> Optional[float]:
        """Best (lowest) ask price."""
        self._clean_asks()
        if self._asks:
            return self._asks[0][0]
        return None

    def get_midprice(self) -> Optional[float]:
        """Midpoint between best bid and best ask."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    def get_spread(self) -> Optional[float]:
        """Bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None

    def get_depth(self, levels: int = 5) -> dict:
        """
        Order book depth at N price levels.

        Returns dict with 'bids' and 'asks' lists of (price, total_quantity).
        """
        self._clean_bids()
        self._clean_asks()

        bid_depth = {}
        for neg_p, _, order in self._bids:
            if not order.is_filled:
                p = -neg_p
                bid_depth[p] = bid_depth.get(p, 0) + order.remaining

        ask_depth = {}
        for p, _, order in self._asks:
            if not order.is_filled:
                ask_depth[p] = ask_depth.get(p, 0) + order.remaining

        bids = sorted(bid_depth.items(), key=lambda x: -x[0])[:levels]
        asks = sorted(ask_depth.items(), key=lambda x: x[0])[:levels]

        return {"bids": bids, "asks": asks}

    def get_last_price(self) -> Optional[float]:
        """Last traded price."""
        if self._trade_tape:
            return self._trade_tape[-1].price
        return None

    def get_vwap(self, n_trades: int = 50) -> Optional[float]:
        """Volume-weighted average price of last N trades."""
        recent = self._trade_tape[-n_trades:]
        if not recent:
            return None
        total_value = sum(t.price * t.quantity for t in recent)
        total_volume = sum(t.quantity for t in recent)
        return total_value / total_volume if total_volume > 0 else None

    @property
    def trade_tape(self) -> List[Trade]:
        return self._trade_tape

    @property
    def price_history(self) -> List[Tuple[int, float]]:
        return self._price_history

    @property
    def spread_history(self) -> List[Tuple[int, float]]:
        return self._spread_history

    @property
    def volume_history(self) -> List[Tuple[int, int]]:
        return self._volume_history

    @property
    def total_trades(self) -> int:
        return len(self._trade_tape)

    @property
    def total_volume(self) -> int:
        return sum(t.quantity for t in self._trade_tape)

    def _clean_bids(self):
        """Remove filled orders from top of bid heap."""
        while self._bids and self._bids[0][2].is_filled:
            heapq.heappop(self._bids)

    def _clean_asks(self):
        """Remove filled orders from top of ask heap."""
        while self._asks and self._asks[0][2].is_filled:
            heapq.heappop(self._asks)
