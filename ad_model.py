"""
Ad-Driven Model — Baseline Comparison
=======================================
Traditional ad-driven attention allocation using Vickrey ad auctions
with platform intermediation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config import SimulationConfig
from agents import ContentProducer, ContentConsumer, Advertiser, create_agent_population


@dataclass
class AdSlot:
    slot_id: int
    content_id: int
    content_quality: float
    timestamp: int


@dataclass
class AdAuctionResult:
    slot_id: int
    winner_id: int
    winning_bid: float
    price_paid: float
    platform_revenue: float
    producer_revenue: float
    consumer_exposed_to: int
    content_quality: float
    timestamp: int


@dataclass
class AdModelMetrics:
    timestep: int = 0
    total_impressions: int = 0
    total_clicks: int = 0
    avg_ctr: float = 0.0
    platform_revenue: float = 0.0
    advertiser_spend: float = 0.0
    producer_revenue: float = 0.0
    avg_bid: float = 0.0
    avg_content_quality_shown: float = 0.0
    consumer_satisfaction: float = 0.0


class AdDrivenModel:
    """Traditional Ad-Driven Attention Allocation Model (Baseline)."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        self.agents = create_agent_population(config, self.rng)
        self.metrics_history: List[AdModelMetrics] = []
        self.producer_revenue: Dict[int, float] = {}
        self.advertiser_spend: Dict[int, float] = {}
        self.consumer_satisfaction_scores: Dict[int, List[float]] = {}
        self.platform_total_revenue = 0.0
        self.agent_surplus: Dict[int, float] = {aid: 0.0 for aid in self.agents["all"]}
        for p in self.agents["producers"]:
            self.producer_revenue[p.agent_id] = 0.0
        for a in self.agents["advertisers"]:
            self.advertiser_spend[a.agent_id] = 0.0

    def run(self) -> List[AdModelMetrics]:
        for t in range(self.config.num_timesteps):
            self.metrics_history.append(self._step(t))
        return self.metrics_history

    def _step(self, t: int) -> AdModelMetrics:
        metrics = AdModelMetrics(timestep=t)
        slots = self._generate_ad_slots(t)
        auction_results = []
        for slot in slots:
            result = self._run_ad_auction(slot, t)
            if result is not None:
                auction_results.append(result)
        impressions, clicks, satisfaction = self._simulate_exposure(auction_results, t)
        metrics.total_impressions = impressions
        metrics.total_clicks = clicks
        metrics.avg_ctr = clicks / max(impressions, 1)
        metrics.platform_revenue = sum(r.platform_revenue for r in auction_results)
        metrics.advertiser_spend = sum(r.price_paid for r in auction_results)
        metrics.producer_revenue = sum(r.producer_revenue for r in auction_results)
        metrics.avg_bid = np.mean([r.winning_bid for r in auction_results]) if auction_results else 0
        metrics.avg_content_quality_shown = np.mean([r.content_quality for r in auction_results]) if auction_results else 0
        metrics.consumer_satisfaction = satisfaction
        self.platform_total_revenue += metrics.platform_revenue
        return metrics

    def _generate_ad_slots(self, t: int) -> List[AdSlot]:
        slots = []
        sid = 0
        for producer in self.agents["producers"]:
            engagement = producer.quality * 0.4 + self.rng.uniform(0, 1) * 0.6
            n = max(1, int(engagement * self.config.ad_slots_per_timestep / len(self.agents["producers"])))
            for _ in range(n):
                slots.append(AdSlot(slot_id=sid, content_id=producer.agent_id, content_quality=producer.quality, timestamp=t))
                sid += 1
        if len(slots) > self.config.ad_slots_per_timestep:
            self.rng.shuffle(slots)
            slots = slots[:self.config.ad_slots_per_timestep]
        return slots

    def _run_ad_auction(self, slot: AdSlot, t: int) -> Optional[AdAuctionResult]:
        bids = []
        for adv in self.agents["advertisers"]:
            if adv.state.cash <= 0:
                continue
            bid_info = adv.generate_ad_bid(quality_score=1.0, timestamp=t)
            if bid_info:
                rank = bid_info["bid"] * (1.0 - self.config.quality_score_weight + self.config.quality_score_weight * bid_info["quality_score"])
                bids.append((rank, bid_info["bid"], adv))
        if not bids:
            return None
        bids.sort(key=lambda x: x[0], reverse=True)
        _, winner_bid, winner = bids[0]
        price = bids[1][1] if self.config.ad_auction_type == "second_price" and len(bids) > 1 else winner_bid
        platform_cut = price * self.config.platform_cut
        producer_share = price - platform_cut
        winner.state.cash -= price
        winner.state.total_cost += price
        self.advertiser_spend[winner.agent_id] = self.advertiser_spend.get(winner.agent_id, 0) + price
        producer = self.agents["all"].get(slot.content_id)
        if producer and hasattr(producer, 'state'):
            producer.state.cash += producer_share
            producer.state.total_revenue += producer_share
            self.producer_revenue[slot.content_id] = self.producer_revenue.get(slot.content_id, 0) + producer_share
            self.agent_surplus[slot.content_id] = self.agent_surplus.get(slot.content_id, 0) + producer_share - producer.production_cost * 0.1
        adv_val = winner.get_valuation(ctr=self.config.base_ctr) * 50
        self.agent_surplus[winner.agent_id] = self.agent_surplus.get(winner.agent_id, 0) + adv_val - price
        return AdAuctionResult(slot_id=slot.slot_id, winner_id=winner.agent_id, winning_bid=winner_bid, price_paid=price,
                               platform_revenue=platform_cut, producer_revenue=producer_share, consumer_exposed_to=winner.agent_id,
                               content_quality=slot.content_quality, timestamp=t)

    def _simulate_exposure(self, results: List[AdAuctionResult], t: int) -> Tuple[int, int, float]:
        total_imp, total_clicks = 0, 0
        sat_scores = []
        consumers = self.agents["consumers"]
        per_c = max(1, len(results) // max(len(consumers), 1))
        for consumer in consumers:
            shown = self.rng.choice(len(results), size=min(per_c, len(results)), replace=False) if results else []
            for idx in shown:
                r = results[idx]
                total_imp += 1
                ctr = self.config.base_ctr * (0.5 + 0.5 * r.content_quality)
                if self.rng.random() < ctr:
                    total_clicks += 1
                sat = consumer.compute_satisfaction(r.content_quality, 0)
                sat_scores.append(sat)
                self.agent_surplus[consumer.agent_id] = self.agent_surplus.get(consumer.agent_id, 0) + sat * 0.01
        return total_imp, total_clicks, np.mean(sat_scores) if sat_scores else 0

    def get_final_results(self) -> dict:
        pr = [self.producer_revenue.get(p.agent_id, 0.0) for p in self.agents["producers"]]
        cs = [self.agent_surplus.get(c.agent_id, 0.0) for c in self.agents["consumers"]]
        ps = [self.agent_surplus.get(p.agent_id, 0.0) for p in self.agents["producers"]]
        advs = [self.agent_surplus.get(a.agent_id, 0.0) for a in self.agents["advertisers"]]
        return {
            "model": "ad_driven", "num_timesteps": self.config.num_timesteps,
            "total_impressions": sum(m.total_impressions for m in self.metrics_history),
            "total_clicks": sum(m.total_clicks for m in self.metrics_history),
            "overall_ctr": sum(m.total_clicks for m in self.metrics_history) / max(sum(m.total_impressions for m in self.metrics_history), 1),
            "platform_total_revenue": self.platform_total_revenue,
            "total_advertiser_spend": sum(self.advertiser_spend.values()),
            "total_producer_revenue": sum(pr), "producer_revenues": pr,
            "producer_surplus": ps, "consumer_surplus": cs, "advertiser_surplus": advs,
            "total_producer_surplus": sum(ps), "total_consumer_surplus": sum(cs),
            "total_advertiser_surplus": sum(advs),
            "total_surplus": sum(ps) + sum(cs) + sum(advs),
            "avg_content_quality_shown": np.mean([m.avg_content_quality_shown for m in self.metrics_history if m.avg_content_quality_shown > 0]) if self.metrics_history else 0,
            "avg_satisfaction": np.mean([m.consumer_satisfaction for m in self.metrics_history]) if self.metrics_history else 0,
            "producer_qualities": [p.quality for p in self.agents["producers"]],
            "metrics_history": self.metrics_history, "agent_surplus": self.agent_surplus,
        }
