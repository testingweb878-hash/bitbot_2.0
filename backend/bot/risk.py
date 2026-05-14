import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from bot import logger
from bot.exchange import exchange_client
from bot.database import SessionLocal, DBPosition
from bot.alerts import telegram_alerts


@dataclass
class Position:
    id:              str
    symbol:          str
    direction:       str   # BUY / SELL
    entry_price:     float
    amount:          float  # base currency amount
    amount_usdt:     float
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct:   float
    take_profit_pct: float
    confidence:      float
    opened_at:       str   = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    order_id:        str   = ""
    status:          str   = "open"  # open | closed_tp | closed_sl | closed_manual
    close_price:     float = 0.0
    pnl_usdt:        float = 0.0
    pnl_pct:         float = 0.0
    closed_at:       str   = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class RiskManager:
    def __init__(self):
        self.positions: Dict[str, Position] = {}   # id → Position
        self.trade_history: List[Dict] = []
        self.win_count: int = 0
        self.loss_count: int = 0
        self.total_pnl_usdt: float = 0.0
        self.daily_pnl_usdt: float = 0.0
        self.last_pnl_reset: str = datetime.utcnow().date().isoformat()
        self._load_state()

    def _save_pos(self, pos: Position):
        db = SessionLocal()
        try:
            existing = db.query(DBPosition).filter(DBPosition.id == pos.id).first()
            
            pos_data = pos.to_dict()
            # Convert ISO strings to datetime objects for SQLAlchemy
            if pos_data.get("opened_at"):
                pos_data["opened_at"] = datetime.fromisoformat(pos_data["opened_at"].replace("Z", ""))
            if pos_data.get("closed_at"):
                pos_data["closed_at"] = datetime.fromisoformat(pos_data["closed_at"].replace("Z", ""))
            else:
                pos_data["closed_at"] = None

            if existing:
                for key, value in pos_data.items():
                    setattr(existing, key, value)
            else:
                db_pos = DBPosition(**pos_data)
                db.add(db_pos)
            
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"DB Error saving position: {e}")
        finally:
            db.close()

    def _load_state(self):
        db = SessionLocal()
        try:
            db_positions = db.query(DBPosition).all()
            
            for p in db_positions:
                pos_dict = {
                    "id": p.id, "symbol": p.symbol, "direction": p.direction,
                    "entry_price": p.entry_price, "amount": p.amount, "amount_usdt": p.amount_usdt,
                    "stop_loss_price": p.stop_loss_price, "take_profit_price": p.take_profit_price,
                    "stop_loss_pct": p.stop_loss_pct, "take_profit_pct": p.take_profit_pct,
                    "confidence": p.confidence, 
                    "opened_at": p.opened_at.isoformat() + "Z" if p.opened_at else "",
                    "order_id": p.order_id,
                    "status": p.status, "close_price": p.close_price, 
                    "pnl_usdt": p.pnl_usdt, "pnl_pct": p.pnl_pct,
                    "closed_at": p.closed_at.isoformat() + "Z" if p.closed_at else ""
                }
                pos_obj = Position(**pos_dict)
                self.positions[p.id] = pos_obj
                
                if p.status != "open":
                    self.trade_history.append(pos_dict)
                    self.total_pnl_usdt += p.pnl_usdt
                    if p.pnl_usdt >= 0: self.win_count += 1
                    else: self.loss_count += 1
            
            logger.info(f"Loaded {len(self.positions)} positions from database.")
        except Exception as e:
            logger.error(f"DB Error loading state: {e}")
        finally:
            db.close()

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        amount_usdt: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        confidence: float,
        atr: Optional[float] = None,
        order_id: str = "",
    ) -> Optional[Position]:
        open_count = sum(1 for p in self.positions.values() if p.status == "open")
        from config import settings
        if open_count >= settings.max_open_trades:
            logger.warning(f"Max open trades ({settings.max_open_trades}) reached — skipping")
            return None

        amount = amount_usdt / entry_price

        if direction == "BUY":
            sl_price = entry_price * (1 - stop_loss_pct / 100)
            tp_price = entry_price * (1 + take_profit_pct / 100)
        else:  # SELL
            sl_price = entry_price * (1 + stop_loss_pct / 100)
            tp_price = entry_price * (1 - take_profit_pct / 100)

        if exchange_client.exchange and symbol in exchange_client.exchange.markets:
             sl_price = float(exchange_client.exchange.price_to_precision(symbol, sl_price))
             tp_price = float(exchange_client.exchange.price_to_precision(symbol, tp_price))
        else:
             sl_price = round(sl_price, 8)
             tp_price = round(tp_price, 8)

        import random
        suffix = random.randint(100, 999)
        pos_id = f"{symbol.replace('/', '')}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{suffix}"
        pos = Position(
            id=pos_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            amount=amount,
            amount_usdt=amount_usdt,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            confidence=confidence,
            order_id=order_id,
        )
        self.positions[pos_id] = pos
        self._save_pos(pos)
        
        telegram_alerts.notify_trade_open(pos.to_dict())
        
        logger.trade(
            f"Position opened [{symbol}] {direction} @ {entry_price} "
            f"| SL: {sl_price} | TP: {tp_price} | Amount: ${amount_usdt:.2f}",
            pos.to_dict()
        )
        return pos

    def check_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        closed = []
        for pos in list(self.positions.values()):
            if pos.status != "open":
                continue
            price = current_prices.get(pos.symbol)
            if not price:
                continue

            hit_tp = hit_sl = False
            if pos.direction == "BUY":
                hit_tp = price >= pos.take_profit_price
                hit_sl = price <= pos.stop_loss_price
            else:
                hit_tp = price <= pos.take_profit_price
                hit_sl = price >= pos.stop_loss_price

            if hit_tp or hit_sl:
                reason = "closed_tp" if hit_tp else "closed_sl"
                closed_pos = self._close_position(pos, price, reason)
                closed.append(closed_pos)

        return closed

    def _check_pnl_reset(self):
        """Reset daily PnL if a new day has started."""
        today = datetime.utcnow().date().isoformat()
        if today != self.last_pnl_reset:
            logger.info(f"New trading day! Resetting daily PnL (was ${self.daily_pnl_usdt:+.2f})")
            self.daily_pnl_usdt = 0.0
            self.last_pnl_reset = today

    def is_circuit_broken(self) -> Tuple[bool, str]:
        """
        Institutional Safety Check:
        Returns (True, reason) if trading should be halted.
        """
        self._check_pnl_reset()
        from config import settings
        
        # 1. Daily Loss Limit
        # Assume starting equity is roughly sum of trade amounts * 10 (arbitrary for now, or use balance)
        # Better: check against settings.max_daily_loss_pct of total balance
        balance = exchange_client.fetch_balance()
        total_equity = balance.get("total", {}).get("USDT", 1000.0) # Fallback to 1000 if fetch fails
        
        daily_loss_limit = total_equity * (settings.max_daily_loss_pct / 100)
        if self.daily_pnl_usdt < -daily_loss_limit:
            return True, f"Daily Loss Limit Exceeded (${self.daily_pnl_usdt:.2f} < -${daily_loss_limit:.2f})"
            
        # 2. Consecutive Losses Guard
        if len(self.trade_history) >= 5:
            last_5 = self.trade_history[-5:]
            if all(t.get("pnl_usdt", 0) < 0 for t in last_5):
                return True, "Consecutive Loss Limit (5) reached. Halting for strategy review."
                
        return False, ""

    def check_signal_decay(self, symbol: str, current_prediction: Dict[str, Any]) -> Optional[Dict]:
        """
        Exits a position if the AI signal has significantly decayed or flipped.
        This protects profits and cuts losses before SL is hit if the thesis changes.
        """
        for pos in list(self.positions.values()):
            if pos.status == "open" and pos.symbol == symbol:
                new_dir = current_prediction.get("direction", "NEUTRAL")
                conf = current_prediction.get("confidence", 0)
                
                # Exit if direction flips or confidence drops below 40% (high uncertainty)
                should_exit = False
                reason = ""
                
                if new_dir != "NEUTRAL" and new_dir != pos.direction:
                    should_exit = True
                    reason = f"Signal Flip ({pos.direction} -> {new_dir})"
                elif conf < 40:
                    should_exit = True
                    reason = f"Signal Decay (Confidence {conf}% < 40%)"
                
                if should_exit:
                    logger.warning(f"🚀 AI Signal Decay/Flip detected for {symbol}: {reason}. Closing position early.")
                    ticker = exchange_client.fetch_ticker(symbol)
                    price = ticker.get("last", 0)
                    if price:
                        return self._close_position(pos, price, f"closed_decay_{new_dir.lower()}")
        return None

    def _close_position(self, pos: Position, close_price: float, reason: str) -> Dict:
        logger.info(f"Executing exchange order to close {pos.symbol} position...")
        if pos.direction == "BUY":
            exchange_client.create_market_sell(pos.symbol, amount=pos.amount)
        else:
            # Corrected: Use base amount to close short position, not amount_usdt
            exchange_client.create_market_buy(pos.symbol, amount=pos.amount)

        if pos.direction == "BUY":
            pnl_pct  = ((close_price - pos.entry_price) / pos.entry_price) * 100
        else:
            pnl_pct  = ((pos.entry_price - close_price) / pos.entry_price) * 100
            
        # Expert logic: Deduct 0.2% for estimated fees (taker + maker)
        pnl_pct -= 0.2
        pnl_usdt = pos.amount_usdt * (pnl_pct / 100)

        pos.status      = reason
        pos.close_price = close_price
        pos.pnl_pct     = round(pnl_pct, 4)
        pos.pnl_usdt    = round(pnl_usdt, 4)
        pos.closed_at   = datetime.utcnow().isoformat() + "Z"

        self.total_pnl_usdt += pnl_usdt
        self.daily_pnl_usdt += pnl_usdt
        if pnl_usdt >= 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        self.trade_history.append(pos.to_dict())
        self._save_pos(pos)
        
        # Notify Telegram
        telegram_alerts.notify_trade_close(pos.to_dict())

        emoji = "💚" if pnl_usdt >= 0 else "🔴"
        logger.trade(
            f"{emoji} Position {reason.upper()} [{pos.symbol}] @ {close_price} "
            f"| PnL: {pnl_pct:+.2f}% (${pnl_usdt:+.2f}) "
            f"| Total PnL: ${self.total_pnl_usdt:+.2f}",
            pos.to_dict()
        )
        return pos.to_dict()

    def close_position_manual(self, pos_id: str, current_price: float) -> Optional[Dict]:
        pos = self.positions.get(pos_id)
        if not pos or pos.status != "open":
            return None
        return self._close_position(pos, current_price, "closed_manual")

    def get_open_positions(self) -> List[Dict]:
        return [p.to_dict() for p in self.positions.values() if p.status == "open"]

    def get_stats(self) -> Dict:
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        return {
            "total_trades": total_trades,
            "wins":         self.win_count,
            "losses":       self.loss_count,
            "win_rate_pct": round(win_rate, 2),
            "total_pnl_usdt": round(self.total_pnl_usdt, 4),
            "daily_pnl_usdt": round(self.daily_pnl_usdt, 4),
            "open_positions": len(self.get_open_positions()),
            "trade_history": self.trade_history[-20:],
        }


# Singleton
risk_manager = RiskManager()
