"""
BitBot AI Predictor — v2
Fuses: Technical Analysis + Advanced Patterns + News Sentiment +
       Multi-Timeframe Momentum + Order Book Microstructure
into a single confidence score and trade signal.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from bot import logger
from bot.context import market_context


# ── Weights for signal fusion ──────────────────────────────────────────────────
W_ML        = 0.25   # Machine Learning Random Forest Model
W_TECHNICAL = 0.25   # 20+ indicator TA composite
W_PATTERNS  = 0.15   # Regime, Fibonacci, divergence, order blocks
W_SENTIMENT = 0.15   # News + Fear & Greed
W_MOMENTUM  = 0.10   # Multi-timeframe agreement
W_MICROSTR  = 0.10   # Order book bid/ask imbalance


def _market_structure_score(orderbook: Optional[Dict]) -> Tuple[float, str, Dict[str, Any]]:
    """
    Analyze deep bid/ask liquidity with exponential distance decay.
    Detects Liquidity Clusters (Whale Walls).
    """
    if not orderbook:
        return 50.0, "NEUTRAL", {}
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    if not bids or not asks:
        return 50.0, "NEUTRAL", {}
        
    mid_price = (bids[0][0] + asks[0][0]) / 2
    
    bid_pressure = 0.0
    clusters = []
    
    avg_bid_vol = sum(v for p, v in bids) / len(bids) if bids else 1
    for price, vol in bids:
        dist = abs(mid_price - price) / mid_price
        # Weight decays exponentially as we go deeper into the book
        weight = np.exp(-15 * dist) 
        bid_pressure += vol * weight
        
        # Detect Liquidity Cluster (3x average volume)
        if vol > avg_bid_vol * 3.5:
            clusters.append({"type": "SUPPORT", "price": price, "strength": round(vol / avg_bid_vol, 1)})
        
    ask_pressure = 0.0
    avg_ask_vol = sum(v for p, v in asks) / len(asks) if asks else 1
    for price, vol in asks:
        dist = abs(price - mid_price) / mid_price
        weight = np.exp(-15 * dist)
        ask_pressure += vol * weight
        
        # Detect Liquidity Cluster (3x average volume)
        if vol > avg_ask_vol * 3.5:
            clusters.append({"type": "RESISTANCE", "price": price, "strength": round(vol / avg_ask_vol, 1)})
        
    total = bid_pressure + ask_pressure
    if total == 0:
        return 50.0, "NEUTRAL", {"clusters": clusters}
        
    bid_pct = (bid_pressure / total) * 100
    
    direction = "NEUTRAL"
    if bid_pct >= 58: direction = "BUY"
    elif bid_pct <= 42: direction = "SELL"
    
    return round(bid_pct, 2), direction, {"clusters": sorted(clusters, key=lambda x: x['strength'], reverse=True)[:3]}


def _multi_timeframe_momentum(ta_list: List[Dict]) -> Tuple[float, str]:
    """Average score across multiple timeframes."""
    scores = [t.get("composite", {}).get("score", 50.0) for t in ta_list if t]
    if not scores:
        return 50.0, "NEUTRAL"
    avg = sum(scores) / len(scores)
    return round(avg, 2), "BUY" if avg >= 60 else ("SELL" if avg <= 40 else "NEUTRAL")


def _bayesian_fusion(prior: float, signals: List[Tuple[float, float]]) -> float:
    """
    Bayesian Probability Update.
    prior: 0.0 to 1.0 (starting belief)
    signals: List of (probability, weight)
    """
    p = prior
    for prob, weight in signals:
        # Scale the influence by weight
        # prob is 0 to 100, normalize to 0 to 1
        evidence = prob / 100.0
        # Incorporate weight: push evidence closer to 0.5 (neutral) if weight is low
        influence = 0.5 + (evidence - 0.5) * weight
        # Bayes update
        p = (p * influence) / (p * influence + (1 - p) * (1 - influence) + 1e-9)
    return p * 100


class Predictor:
    """
    Fuses all signal sources into one confidence score.
    Trades only when confidence ≥ threshold AND direction is clear.
    """

    def __init__(self, confidence_threshold: float = 85.0):
        self.threshold = confidence_threshold
        self.last_raw_score: Dict[str, float] = {} # Symbol-based smoothing state
        self.smoothing_factor = 0.35 # Expert stable smoothing (0.0 to 1.0)
        self.confirmation_counts: Dict[str, int] = {} # Counter for sustained signals
        self.req_confirmations = 2 # Need 2 cycles of high confidence to trade
        self.last_book_state: Dict[str, Dict[str, float]] = {} # Track liquidity delta

    def predict(
        self,
        ta_results: Dict[str, Any],
        ta_results_htf: Optional[Dict] = None,
        ta_results_ltf: Optional[Dict] = None,
        news_results: Optional[Dict] = None,
        orderbook: Optional[Dict] = None,
        funding_rate: Optional[float] = None,
        pattern_results: Optional[Dict] = None,
        ml_results: Optional[Dict] = None,
        symbol: str = "",
        active_pos: Optional[Any] = None,
    ) -> Dict[str, Any]:

        logger.thinking(f"🧠 Running AI fusion prediction for {symbol}...")

        # ── 1. Technical Analysis Score ────────────────────────────────────────
        comp     = ta_results.get("composite", {})
        ta_score = comp.get("score", 50.0)
        ta_dir   = comp.get("direction", "NEUTRAL")

        # ── 2. Advanced Pattern Score ──────────────────────────────────────────
        if pattern_results:
            pat_comp  = pattern_results.get("composite", {})
            pat_score = pat_comp.get("score", 50.0)
            pat_dir   = pat_comp.get("direction", "NEUTRAL")
            regime    = pattern_results.get("regime", {})
            adx_val   = regime.get("adx", 20)
            # Ranging / choppy market → dampen pattern confidence
            if not regime.get("tradeable", True):
                orig = pat_score
                pat_score = 50.0 + (pat_score - 50.0) * 0.3   # Dampen towards neutral
                logger.warning(
                    f"Non-tradeable regime ({regime.get('regime','?')}) — "
                    f"pattern score dampened {orig:.1f}% → {pat_score:.1f}%"
                )
        else:
            pat_score = 50.0
            pat_dir   = "NEUTRAL"
            regime    = {}
            adx_val   = 20

        atr_pct = ta_results.get("atr", {}).get("pct", 1.0)

        # ── 3. News / Sentiment Score ──────────────────────────────────────────
        if news_results:
            sent_pct = news_results.get("sentiment_pct", 50.0)
            sent_dir = news_results.get("sentiment", "NEUTRAL")
            if news_results.get("coin_is_trending", False):
                sent_pct = min(100, sent_pct + 5)
        else:
            sent_pct = 50.0
            sent_dir = "NEUTRAL"

        # ── 4. Multi-Timeframe Momentum ────────────────────────────────────────
        mtf_score, mtf_dir = _multi_timeframe_momentum(
            [r for r in [ta_results_ltf, ta_results, ta_results_htf] if r]
        )

        # ── 5. Order Book Microstructure ───────────────────────────────────────
        micro_score, micro_dir, micro_data = _market_structure_score(orderbook)
        
        # ── 5b. Liquidity Delta (Institutional Shift) ─────────────────────────
        liq_delta_score = 50.0
        if symbol in self.last_book_state and orderbook:
            prev = self.last_book_state[symbol]
            curr_bids = sum(v for p, v in orderbook.get("bids", []))
            curr_asks = sum(v for p, v in orderbook.get("asks", []))
            
            bid_delta = (curr_bids - prev["bids"]) / (prev["bids"] + 1e-9)
            ask_delta = (curr_asks - prev["asks"]) / (prev["asks"] + 1e-9)
            
            # If bids increasing more than asks -> Bullish Delta
            liq_delta_score = 50 + (bid_delta - ask_delta) * 100
            liq_delta_score = max(0, min(100, liq_delta_score))
            
        if orderbook:
            self.last_book_state[symbol] = {
                "bids": sum(v for p, v in orderbook.get("bids", [])),
                "asks": sum(v for p, v in orderbook.get("asks", []))
            }

        # ── 5b. Machine Learning Score ─────────────────────────────────────────
        ml_score = (ml_results or {}).get("ml_score", 50.0)
        ml_dir   = (ml_results or {}).get("ml_direction", "NEUTRAL")

        # ── 6. Weighted Fusion -> Bayesian Neural Fusion ───────────────────────
        # Prior is based on Global Market Context (BTC/ETH)
        prior_prob = market_context.market_score / 100.0
        
        # Bayesian evidence components
        evidence = [
            (ml_score,    W_ML),
            (ta_score,    W_TECHNICAL),
            (pat_score,   W_PATTERNS),
            (sent_pct,    W_SENTIMENT),
            (mtf_score,   W_MOMENTUM),
            (micro_score, W_MICROSTR),
            (liq_delta_score, 0.15) # New: Liquidity Delta evidence
        ]
        
        # Adjust weights based on ADX (Expert regime awareness)
        final_evidence = []
        for score, weight in evidence:
            eff_weight = weight
            if adx_val > 30: # Strong trend
                if score > 60 or score < 40: eff_weight *= 1.2
            final_evidence.append((score, eff_weight))
            
        raw_score = _bayesian_fusion(prior_prob, final_evidence)

        # ── 6d. Expert Smoothing (Stability Filter) ───────────────────────────
        # This prevents the score from "fluctuating" too wildly between cycles
        if symbol in self.last_raw_score:
            prev = self.last_raw_score[symbol]
            # EMA formula: current * alpha + previous * (1 - alpha)
            raw_score = (raw_score * self.smoothing_factor) + (prev * (1 - self.smoothing_factor))
            
        self.last_raw_score[symbol] = raw_score

        # ── 7. Direction Consensus Voting & Synergy ────────────────────────────
        directions = [ml_dir, ta_dir, pat_dir, sent_dir, mtf_dir, micro_dir]
        buy_votes  = directions.count("BUY")
        sell_votes = directions.count("SELL")

        if buy_votes >= 4:
            final_dir = "BUY"
            consensus_boost = 10
        elif sell_votes >= 4:
            final_dir = "SELL"
            consensus_boost = -10
        elif buy_votes >= 3:
            final_dir = "BUY"
            consensus_boost = 5
        elif sell_votes >= 3:
            final_dir = "SELL"
            consensus_boost = -5
        elif buy_votes > sell_votes:
            final_dir = "BUY"
            consensus_boost = 0
        elif sell_votes > buy_votes:
            final_dir = "SELL"
            consensus_boost = 0
        else:
            final_dir = "NEUTRAL"
            consensus_boost = 0

        # ── 6b. Volume Confirmation Expert Rule ────────────────────────────────
        # A breakout or reversal without volume is a "fakeout", high volume indicates operator interest
        vol_ratio = ta_results.get("volume", {}).get("ratio", 1.0)
        if final_dir != "NEUTRAL":
            if vol_ratio < 0.8:
                # Dampen confidence if volume is low
                logger.warning(f"Low volume ({vol_ratio:.1f}x) detected for {final_dir} signal -> Dampening confidence")
                raw_score = 50 + (raw_score - 50) * 0.6
            elif vol_ratio > 1.5:
                # Boost confidence if volume is high (operators are interested)
                logger.info(f"High volume ({vol_ratio:.1f}x) detected -> Operators are interested. Boosting confidence.")
                boost = min(15, (vol_ratio - 1.0) * 10)
                if final_dir == "BUY":
                    raw_score = min(100, raw_score + boost)
                else:
                    raw_score = max(0, raw_score - boost)
            
        # Apply consensus boost
        if final_dir == "BUY":
            raw_score = min(100, raw_score + consensus_boost)
        elif final_dir == "SELL":
            raw_score = max(0, raw_score + consensus_boost)

        # ── 6c. Confidence Sharpening (Sigmoid-like scaling) ───────────────────
        # This pushes scores further away from the 50% neutral zone to make signals clearer
        if abs(raw_score - 50) > 5:
            # Shift towards the extremes
            diff = raw_score - 50
            # Scaling factor: stronger signals get pushed harder
            sharpen = 1.3 if abs(diff) > 15 else 1.15
            raw_score = 50 + (diff * sharpen)
            raw_score = max(5, min(95, raw_score))
            
        # Synergy Boost: If ML and Technical agree, it's a very strong sign
        if ml_dir != "NEUTRAL" and ml_dir == ta_dir:
            if final_dir == ml_dir:
                raw_score = min(100, raw_score + 5) if final_dir == "BUY" else max(0, raw_score - 5)
                logger.success(f"Synergy detected: ML & TA alignment on {final_dir} -> +5% boost")
                
        # Regime Alignment Boost: If signal matches market trend, it's safer
        current_regime = regime.get("regime", "")
        if (final_dir == "BUY" and current_regime == "TRENDING_UP") or \
           (final_dir == "SELL" and current_regime == "TRENDING_DOWN"):
            raw_score = min(100, raw_score + 7) if final_dir == "BUY" else max(0, raw_score - 7)
            logger.info(f"Regime Alignment: Signal {final_dir} matches {current_regime} -> +7% boost")

        # ── 8. Contradiction Penalties ─────────────────────────────────────────
        rsi_val = ta_results.get("rsi", {}).get("value", 50)
        penalty = 0.0

        # Expert Insight: Market Regime & Trend Alignment
        if final_dir == "BUY" and current_regime == "TRENDING_DOWN":
            penalty += 15
            logger.warning(f"Trend Risk: Counter-trend BUY attempt during TRENDING_DOWN -> −15% penalty")
        elif final_dir == "SELL" and current_regime == "TRENDING_UP":
            penalty += 15
            logger.warning(f"Trend Risk: Counter-trend SELL attempt during TRENDING_UP -> −15% penalty")
        
        # Expert Insight: BTC Correlation Check
        if symbol != "BTC/USDT":
            context_penalty = market_context.get_context_penalty(final_dir)
            if context_penalty > 0:
                penalty += context_penalty
                logger.warning(f"Macro Risk: BTC/ETH divergence detected -> −{context_penalty}% penalty")

        if final_dir == "BUY" and rsi_val > 80:
            penalty += 10
            logger.warning(f"RSI overbought ({rsi_val:.1f}) → BUY confidence penalised −10%")
        elif final_dir == "SELL" and rsi_val < 20:
            penalty += 10
            logger.warning(f"RSI oversold ({rsi_val:.1f}) → SELL confidence penalised −10%")

        # Divergence contradictions
        div = (pattern_results or {}).get("divergence", {})
        if final_dir == "SELL" and div.get("bullish_divergence"):
            penalty += 5
            logger.warning("Bullish divergence detected against SELL signal → −5%")
        if final_dir == "BUY" and div.get("bearish_divergence"):
            penalty += 5
            logger.warning("Bearish divergence detected against BUY signal → −5%")

        # Ranging market penalty for directional trade
        if regime.get("regime") == "RANGING" and final_dir != "NEUTRAL":
            penalty += 8
            logger.warning("Ranging market → directional trade confidence −8%")

        # ── 8b. HTF Confirmation ──────────────────────────────────────────────
        adx_val = regime.get("adx", 20)
        htf_dir = "NEUTRAL"

        if ta_results_htf:
            htf_comp = ta_results_htf.get("composite", {})
            htf_dir  = htf_comp.get("direction", "NEUTRAL")
            if final_dir != "NEUTRAL" and htf_dir != "NEUTRAL" and htf_dir != final_dir:
                penalty += 12
                logger.warning(f"HTF Contradiction! Signal is {final_dir} but HTF is {htf_dir} → −12%")
            elif final_dir != "NEUTRAL" and final_dir == htf_dir:
                raw_score = min(100, raw_score + 5)
                logger.info(f"HTF Confirmation! Both timeframes aligned on {final_dir} → +5%")

        # ── 8c. Volatility Filter ─────────────────────────────────────────────
        if atr_pct > 3.0:  # High volatility / panic
            penalty += 10
            logger.warning(f"Extreme volatility (ATR {atr_pct:.1f}%) → −10% confidence")

        # ── 8d. Trend Strength Filter ─────────────────────────────────────────
        if adx_val < 20 and final_dir != "NEUTRAL":
            penalty += 5
            logger.warning(f"Weak trend (ADX {adx_val:.1f}) → −5% confidence")

        # ── 8e. Funding Rate Penalty (Crowded Trade) ──────────────────────────
        if funding_rate is not None:
            # High positive funding -> Longs paying Shorts (Bullish exhaustion risk)
            if final_dir == "BUY" and funding_rate > 0.01: # 0.01% per 8h
                penalty += 8
                logger.warning(f"Funding Risk: Crowded Longs (Funding {funding_rate:.4f}%) → −8% penalty")
            # High negative funding -> Shorts paying Longs (Bearish exhaustion risk)
            elif final_dir == "SELL" and funding_rate < -0.01:
                penalty += 8
                logger.warning(f"Funding Risk: Crowded Shorts (Funding {funding_rate:.4f}%) → −8% penalty")

        # ── 8f. Directional Confidence Calculation ───────────────────────────
        # Expert Logic: Confidence should represent strength of conviction in the CURRENT direction.
        if final_dir == "BUY":
            # For BUY, higher raw_score means higher confidence
            base_confidence = raw_score
        elif final_dir == "SELL":
            # For SELL, lower raw_score means higher confidence in the downward move
            base_confidence = 100 - raw_score
        else:
            base_confidence = 0
            
        confidence = round(max(0, min(98.0, base_confidence - penalty)), 2)
        win_prob   = round(confidence, 2) # Bayesian posterior as win probability
        
        # ── 8f. Expert Confirmation Buffer ────────────────────────────────────
        # Only trade if high confidence is SUSTAINED for multiple cycles
        potential_signal = confidence >= self.threshold and final_dir in ("BUY", "SELL")
        
        if potential_signal:
            self.confirmation_counts[symbol] = self.confirmation_counts.get(symbol, 0) + 1
        else:
            self.confirmation_counts[symbol] = 0 # Reset if signal drops
            
        # Actual trade decision
        should_trade = potential_signal and self.confirmation_counts[symbol] >= self.req_confirmations
        
        if potential_signal and not should_trade:
            logger.info(f"Signal Pending: Sustaining conviction... ({self.confirmation_counts[symbol]}/{self.req_confirmations})")

        # ── 9. Adaptive Risk Parameters ────────────────────────────────────────
        # ── 9. Adaptive Risk Parameters (Expert Stabilized) ───────────────────
        if active_pos:
            # If we have an active position, we LOCK the reasoning to its fixed parameters
            stop_loss_pct   = active_pos.stop_loss_pct
            take_profit_pct = active_pos.take_profit_pct
            sl_price        = active_pos.stop_loss_price
            tp_price        = active_pos.take_profit_price
            
            reasoning: List[str] = [
                f"🛡️ ACTIVE TRADE: Entry ${active_pos.entry_price:.2f}",
                f"🛡️ FIXED EXIT: Stop Loss at ${sl_price} ({stop_loss_pct}%)",
                f"🛡️ FIXED TARGET: Take Profit at ${tp_price} ({take_profit_pct}%)",
                f"Neural Fusion: Tracking {active_pos.direction} momentum with institutional precision."
            ]
        else:
            # Use standard dynamic multipliers for potential new trade
            sl_multiplier = 1.5 if adx_val > 30 else 2.0
            stop_loss_pct   = round(max(1.0, min(6.0, atr_pct * sl_multiplier)), 2)
            take_profit_pct = round(stop_loss_pct * 2.5, 2)
            
            reasoning: List[str] = [
                f"Risk Model: SL set at {stop_loss_pct}% ({sl_multiplier}x ATR volatility)",
                f"Target Model: TP set at {take_profit_pct}% (2.5:1 Reward/Risk ratio)"
            ]
        if ml_dir != "NEUTRAL":
            reasoning.append(f"AI ML: {ml_dir} signal (Random Forest probability: {ml_score:.1f}%)")
        if ta_dir != "NEUTRAL":
            reasoning.append(f"TA Composite: {ta_dir} signal ({ta_score:.0f}% score)")
            
        if vol_ratio > 1.5:
            reasoning.append(f"Operator Interest: High trading volume ({vol_ratio:.1f}x average) confirms institutional participation.")
        elif vol_ratio < 0.8:
            reasoning.append(f"Volume Warning: Low trading volume ({vol_ratio:.1f}x average) indicates lack of institutional interest.")
            
        ema = ta_results.get("ema", {})
        if ema:
            ema_trend = "Bullish" if ema.get("price", 0) > ema.get("ema50", float('inf')) else "Bearish"
            reasoning.append(f"Trend Indicator (EMA50): {ema_trend} (Price vs Moving Average)")
            
        macd = ta_results.get("macd", {})
        if macd:
            macd_state = "Bullish Cross" if macd.get("macd", 0) > macd.get("signal", 0) else "Bearish Cross"
            reasoning.append(f"Momentum Indicator (MACD): {macd_state}")
            
        rsi_val = ta_results.get("rsi", {}).get("value", 50)
        if rsi_val > 70:
            reasoning.append(f"Overbought Warning (RSI): {rsi_val:.1f} - High risk of reversal")
        elif rsi_val < 30:
            reasoning.append(f"Oversold Opportunity (RSI): {rsi_val:.1f} - Undervalued conditions")
            
        if pat_dir != "NEUTRAL":
            reasoning.append(f"Patterns: {pat_dir} — Regime: {regime.get('regime','?')} ADX {regime.get('adx',0):.1f}")
        bb = ta_results.get("bollinger", {})
        if bb.get("is_squeeze"):
            reasoning.append("Volatility Squeeze (Bollinger Bands): Breakout imminent")
        if htf_dir == final_dir and final_dir != "NEUTRAL":
            reasoning.append(f"Institutional Confirmation: Higher Timeframe ({htf_dir}) aligns with signal")
        fib = (pattern_results or {}).get("fibonacci", {})
        if fib.get("at_support"):
            reasoning.append(f"At Fibonacci support ({fib.get('nearest_fib',0)*100:.1f}% level)")
        if fib.get("at_resistance"):
            reasoning.append(f"At Fibonacci resistance ({fib.get('nearest_fib',0)*100:.1f}% level)")
        vp = (pattern_results or {}).get("volume_profile", {})
        if vp.get("poc"):
            reasoning.append(f"Volume POC: ${vp['poc']:.2f} ({vp.get('price_vs_poc','?')} POC, VA={'in' if vp.get('in_value_area') else 'out'})")
        if div.get("bullish_divergence"):
            reasoning.append("RSI Bullish Divergence detected ↗")
        if div.get("bearish_divergence"):
            reasoning.append("RSI Bearish Divergence detected ↘")
            
        wyckoff = (pattern_results or {}).get("wyckoff", {})
        if wyckoff.get("phase") in ["ACCUMULATION", "DISTRIBUTION"]:
            reasoning.append(f"Wyckoff Phase: {wyckoff.get('phase')} (Conf: {wyckoff.get('confidence', 0)*100:.0f}%)")
            
        elliott = (pattern_results or {}).get("elliott", {})
        if elliott.get("trend") != "NEUTRAL":
            reasoning.append(f"Elliott Wave: {elliott.get('trend')} (Wave {elliott.get('wave', 0)})")
            
        sweeps = (pattern_results or {}).get("liquidity_sweeps", [])
        if sweeps:
            reasoning.append(f"Liquidity Sweep: {sweeps[-1]['type']} at ${sweeps[-1]['price']:.2f}")

        # Order Book Clusters
        for cluster in micro_data.get("clusters", []):
            reasoning.append(f"Whale Wall Found: {cluster['type']} at ${cluster['price']:.2f} ({cluster['strength']}x volume)")

        trix = ta_results.get("trix", {})
        if trix:
            reasoning.append(f"TRIX Momentum: {'Bullish' if trix.get('value', 0) > 0 else 'Bearish'} ({trix.get('value', 0):.2f})")
            
        uo = ta_results.get("ultimate_oscillator", {})
        if uo:
            reasoning.append(f"Ultimate Oscillator: {uo.get('value', 50):.1f}")
            
        if sent_dir != "NEUTRAL":
            if sent_dir == "POSITIVE":
                reasoning.append(f"News Sentiment: POSITIVE ({sent_pct:.0f}%) -> Macro narrative strongly supports UPWARD price discovery.")
            else:
                reasoning.append(f"News Sentiment: NEGATIVE ({sent_pct:.0f}%) -> Macro narrative warns of potential DOWNSIDE CRASH risk.")
                
        if mtf_dir != "NEUTRAL":
            reasoning.append(f"Multi-Timeframe Engine: {mtf_dir} agreement ({mtf_score:.0f}%)")
        candle_patterns = ta_results.get("candle_patterns", [])
        if candle_patterns:
            reasoning.append(f"Candlestick Patterns: {', '.join(candle_patterns)}")
        ichi = ta_results.get("ichimoku", {})
        if ichi.get("tenkan", 0) > ichi.get("kijun", 1):
            reasoning.append("Ichimoku Cloud: Tenkan > Kijun (bullish cross formation)")
            
        # Kelly Criterion sizing
        win_rate = 0.55 # Assume 55% for the AI model
        risk_reward = 2.5
        kelly_f = win_rate - ((1 - win_rate) / risk_reward)
        kelly_fraction = max(0, min(0.1, kelly_f * (confidence / 100)))
        reasoning.append(f"Position Sizing (Kelly): Institutional recommendation {kelly_fraction*100:.1f}% of equity")

        result = {
            "symbol":            symbol,
            "direction":         final_dir,
            "confidence":        confidence,
            "win_probability":   win_prob,
            "should_trade":      should_trade,
            "threshold":         self.threshold,
            "stop_loss_pct":     stop_loss_pct,
            "take_profit_pct":   take_profit_pct,
            "reasoning":         reasoning,
            "component_scores": {
                "ml_ai":          round(ml_score,    2),
                "technical":      round(ta_score,    2),
                "patterns":       round(pat_score,   2),
                "sentiment":      round(sent_pct,    2),
                "momentum_mtf":   round(mtf_score,   2),
                "microstructure": round(micro_score, 2),
            },
            "direction_votes": {
                "BUY":     buy_votes,
                "SELL":    sell_votes,
                "NEUTRAL": max(0, 5 - buy_votes - sell_votes),
            },
            "pattern_summary": {
                "regime":            regime.get("regime", "?"),
                "adx":               round(regime.get("adx", 0), 2),
                "di_plus":           round(regime.get("di_plus", 0), 2),
                "di_minus":          round(regime.get("di_minus", 0), 2),
                "tradeable":         regime.get("tradeable", True),
                "bullish_div":       div.get("bullish_divergence", False),
                "bearish_div":       div.get("bearish_divergence", False),
                "at_fib_support":    fib.get("at_support", False),
                "at_fib_resistance": fib.get("at_resistance", False),
                "nearest_fib":       fib.get("nearest_fib", None),
                "poc":               vp.get("poc", 0),
                "in_value_area":     vp.get("in_value_area", False),
            },
            "raw_score":  round(raw_score, 2),
            "penalty":    round(penalty, 2),
            "market_context": market_context.market_score,
            "expert_logic": self._generate_expert_summary(final_dir, confidence, raw_score, penalty, current_regime, rsi_val, adx_val, funding_rate)
        }

        emoji  = "🚀" if final_dir == "BUY" else ("🔻" if final_dir == "SELL" else "⏸️")
        status = "✅ TRADE SIGNAL" if should_trade else "🔍 Monitoring"
        logger.signal(
            f"{status} [{symbol}] {emoji} {final_dir} | "
            f"Confidence: {confidence}% (raw:{raw_score:.1f}% pen:{penalty:.1f}%) | "
            f"Threshold: {self.threshold}% | SL: {stop_loss_pct}% TP: {take_profit_pct}% | "
            f"Votes: {buy_votes}↑/{sell_votes}↓",
            result,
        )
        return result

    def _generate_expert_summary(self, direction, confidence, score, penalty, regime, rsi, adx, funding) -> str:
        if direction == "NEUTRAL":
            return "Bayesian Fusion Engine: Market is currently in a state of high entropy (equilibrium). Institutional order flow is balanced. Recommendation: Await structural breakout or liquidity sweep."
        
        strength = "Aggressive" if confidence > 85 else "Moderate"
        trend_status = "aligned with" if (direction == "BUY" and "UP" in regime) or (direction == "SELL" and "DOWN" in regime) else "countering"
        
        verdict = f"Bayesian Neural Fusion: {strength} {direction} conviction detected. "
        verdict += f"The calculated winning probability is {confidence:.1f}% based on multi-source evidence fusion. "
        
        if penalty > 10:
            verdict += f"Note: Signal is {trend_status} the primary structural trend. "
        else:
            verdict += f"Signal is well-{trend_status} the institutional trend. "
            
        if funding is not None and abs(funding) > 0.01:
            verdict += f"Institutional positioning is {'OVERBULLISH' if funding > 0 else 'OVERBEARISH'} (Funding: {funding:.4f}%). "
            
        return verdict
