#!/usr/bin/env python3
"""
High Probability Trading Bot – Kraken Spot (Paper Trading)
- Fixed risk: $20 per trade
- Fixed profit target: $40 (2:1 reward)
- Uses ATR-based stop-loss (dynamic, widens in volatile markets)
- Balance is $100 simulated (can be changed via SIMULATED_BALANCE env var)
- Position size is automatically capped by available cash
"""

import time
import logging
import json
import os
import numpy as np
import random
from datetime import datetime, timezone
from kraken.spot import Market, User

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("high_prob_bot")

# ---------- Kraken API ----------
class KrakenClient:
    def __init__(self, api_key=None, api_secret=None):
        self.market = Market()
        self.user = None
        self.pair = "XBTUSD"
        if api_key and api_secret:
            self.user = User(key=api_key, secret=api_secret)

    def get_current_price(self):
        try:
            ticker = self.market.get_ticker(pair=self.pair)
            if ticker and self.pair in ticker:
                return float(ticker[self.pair]['c'][0])
        except Exception as e:
            log.warning(f"Price fetch error: {e}")
        return None

    def get_historical_candles(self, interval: int, limit: int = 50):
        try:
            ohlc = self.market.get_ohlc(pair=self.pair, interval=interval)
            if ohlc and 'result' in ohlc and self.pair in ohlc['result']:
                data = ohlc['result'][self.pair]
                candles = []
                for item in data[:limit]:
                    candles.append({
                        "timestamp": item[0],
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                        "trades": item[6]
                    })
                return candles
        except Exception as e:
            log.warning(f"Candles error: {e}")
        return []

    def place_order(self, side: str, size: float, price=None):
        log.error("place_order called but bot is in PAPER MODE – no real order placed")
        return None


# ---------- Market Data Wrapper ----------
class RealMarketData:
    def __init__(self, api_key=None, api_secret=None):
        self.kraken = KrakenClient(api_key, api_secret)
        self.cached_price = 50000.0
        self.last_price_fetch = 0
        self.price_cache_ttl = 5
        self.interval_map = {1: 1, 5: 5, 15: 15, 60: 60, 240: 240, 1440: 1440}

    def get_current_price(self) -> float:
        now = time.time()
        if now - self.last_price_fetch < self.price_cache_ttl:
            return self.cached_price
        price = self.kraken.get_current_price()
        if price is None:
            price = self.cached_price * (1 + random.uniform(-0.001, 0.001))
        self.cached_price = price
        self.last_price_fetch = now
        return price

    def get_historical_candles(self, interval: int, limit: int = 50) -> list:
        candles = self.kraken.get_historical_candles(interval, limit)
        if not candles:
            candles = self._generate_simulated_candles(limit)
        return candles

    def _generate_simulated_candles(self, count: int) -> list:
        current_price = self.get_current_price()
        candles = []
        price = current_price
        for _ in range(count):
            change = np.random.normal(0, 0.002) * price
            close = price + change
            high = max(price, close) + abs(change) * random.uniform(0.2, 0.8)
            low = min(price, close) - abs(change) * random.uniform(0.2, 0.8)
            candles.append({
                "timestamp": int(time.time() * 1000),
                "open": round(price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": random.randint(100, 10000),
                "trades": random.randint(1, 500)
            })
            price = close
        return candles

    def get_previous_session_close(self) -> float:
        candles = self.get_historical_candles(60, 2)
        if len(candles) >= 2:
            return candles[-2]["close"]
        return self.get_current_price() * 0.98

    def generate_candles(self, timeframe_min: int, count: int) -> list:
        if timeframe_min not in self.interval_map:
            raise ValueError(f"Unsupported timeframe {timeframe_min}")
        interval = self.interval_map[timeframe_min]
        return self.get_historical_candles(interval, count)

    def place_order(self, side: str, size: float, price=None):
        return self.kraken.place_order(side, size, price)


# ---------- Adaptive Parameters (unchanged) ----------
class AdaptiveParams:
    def __init__(self, state_file="learning_state.json"):
        self.state_file = state_file
        self.pattern_weights = {'engulfing': 1.0, 'divergence': 1.0, 'double': 1.0}
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.volume_spike_factor = 2.0
        self.double_pattern_tolerance = 0.003
        self.win_counts = {p: 0 for p in self.pattern_weights}
        self.total_counts = {p: 0 for p in self.pattern_weights}
        self.recent_winrate = {p: 0.5 for p in self.pattern_weights}
        self.last_update_time = time.time()
        self.update_interval = 3600
        self.load_state()

    def save_state(self):
        state = {
            "pattern_weights": self.pattern_weights,
            "win_counts": self.win_counts,
            "total_counts": self.total_counts,
            "recent_winrate": self.recent_winrate,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "volume_spike_factor": self.volume_spike_factor,
            "double_pattern_tolerance": self.double_pattern_tolerance,
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            log.error(f"Failed to save learning state: {e}")

    def load_state(self):
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.pattern_weights = state["pattern_weights"]
            self.win_counts = state["win_counts"]
            self.total_counts = state["total_counts"]
            self.recent_winrate = state["recent_winrate"]
            self.rsi_oversold = state["rsi_oversold"]
            self.rsi_overbought = state["rsi_overbought"]
            self.volume_spike_factor = state["volume_spike_factor"]
            self.double_pattern_tolerance = state["double_pattern_tolerance"]
            log.info("Learning state loaded")
        except FileNotFoundError:
            log.info("No previous learning state found, starting fresh")
        except Exception as e:
            log.error(f"Error loading learning state: {e}")

    def record_trade_result(self, pattern_name: str, pnl: float):
        if pattern_name not in self.total_counts:
            return
        self.total_counts[pattern_name] += 1
        if pnl > 0:
            self.win_counts[pattern_name] += 1
        if self.total_counts[pattern_name] >= 5:
            self.recent_winrate[pattern_name] = self.win_counts[pattern_name] / self.total_counts[pattern_name]
        self.save_state()

    def adapt_parameters(self):
        now = time.time()
        if now - self.last_update_time < self.update_interval:
            return
        self.last_update_time = now
        total = sum(max(0.2, self.recent_winrate[p]) for p in self.pattern_weights)
        for p in self.pattern_weights:
            raw = max(0.2, self.recent_winrate[p])
            self.pattern_weights[p] = raw / total

        div_win = self.recent_winrate.get('divergence', 0.5)
        if div_win > 0.6:
            self.rsi_oversold = max(20, self.rsi_oversold - 2)
            self.rsi_overbought = min(80, self.rsi_overbought + 2)
        elif div_win < 0.4:
            self.rsi_oversold = min(40, self.rsi_oversold + 2)
            self.rsi_overbought = max(60, self.rsi_overbought - 2)

        eng_win = self.recent_winrate.get('engulfing', 0.5)
        if eng_win > 0.6:
            self.volume_spike_factor = max(1.2, self.volume_spike_factor - 0.1)
        elif eng_win < 0.4:
            self.volume_spike_factor = min(3.0, self.volume_spike_factor + 0.2)

        dbl_win = self.recent_winrate.get('double', 0.5)
        if dbl_win > 0.6:
            self.double_pattern_tolerance = max(0.002, self.double_pattern_tolerance - 0.0005)
        elif dbl_win < 0.4:
            self.double_pattern_tolerance = min(0.01, self.double_pattern_tolerance + 0.001)

        log.info(f"Adaptation: weights={self.pattern_weights}, RSI({self.rsi_oversold}/{self.rsi_overbought}), vol_spike={self.volume_spike_factor:.2f}, double_tol={self.double_pattern_tolerance:.4f}")
        self.save_state()

    def choose_pattern(self) -> str:
        patterns = list(self.pattern_weights.keys())
        probs = [self.pattern_weights[p] for p in patterns]
        total = sum(probs)
        if total <= 0:
            probs = [1.0 / len(patterns)] * len(patterns)
        else:
            probs = [p / total for p in probs]
        if np.random.random() < 0.1:
            return np.random.choice(patterns)
        return np.random.choice(patterns, p=probs)


# ---------- Strategy (unchanged from previous version with strong filters) ----------
class AdaptiveHighProbStrategy:
    def __init__(self, data_client, adaptive_params):
        self.data = data_client
        self.adaptive = adaptive_params
        self.last_signal = None
        self.last_pattern_used = None

    def get_trend_direction(self) -> str:
        candles_1h = self.data.generate_candles(60, 210)
        if len(candles_1h) < 200:
            return "neutral"
        closes_1h = [c['close'] for c in candles_1h[-200:]]
        sma200_1h = sum(closes_1h) / len(closes_1h)
        price_1h = closes_1h[-1]

        candles_4h = self.data.generate_candles(240, 110)
        if len(candles_4h) < 100:
            return "neutral"
        closes_4h = [c['close'] for c in candles_4h[-100:]]
        sma100_4h = sum(closes_4h) / len(closes_4h)
        price_4h = closes_4h[-1]

        if price_1h > sma200_1h and price_4h > sma100_4h:
            return "bullish"
        elif price_1h < sma200_1h and price_4h < sma100_4h:
            return "bearish"
        return "neutral"

    def check_engulfing_with_volume(self, current_price):
        candles = self.data.generate_candles(1, 5)
        if len(candles) < 2:
            return None
        prev_close = self.data.get_previous_session_close()
        last = candles[-1]
        prev = candles[-2]
        avg_vol = np.mean([c['volume'] for c in candles[-5:]])
        vol_spike = last['volume'] > avg_vol * self.adaptive.volume_spike_factor
        if (prev['close'] < prev['open'] and last['close'] > last['open'] and
            last['close'] > prev['open'] and last['open'] < prev['close'] and
            last['low'] <= prev_close <= last['high'] and last['close'] > prev_close and vol_spike):
            return "BUY"
        elif (prev['close'] > prev['open'] and last['close'] < last['open'] and
              last['close'] < prev['open'] and last['open'] > prev['close'] and
              last['high'] >= prev_close >= last['low'] and last['close'] < prev_close and vol_spike):
            return "SELL"
        return None

    def calculate_rsi(self, prices: list, period=14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) else 0
        avg_loss = np.mean(losses) if len(losses) else 0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def check_divergence(self, current_price):
        candles = self.data.generate_candles(15, 30)
        if len(candles) < 10:
            return None
        prices_15m = [c['close'] for c in candles]
        if len(prices_15m) < 6:
            return None
        rsi = self.calculate_rsi(prices_15m)
        if len(prices_15m) >= 6:
            if prices_15m[-1] < prices_15m[-5] and rsi > self.calculate_rsi(prices_15m[:-5]):
                if rsi < self.adaptive.rsi_oversold:
                    return "BUY"
            if prices_15m[-1] > prices_15m[-5] and rsi < self.calculate_rsi(prices_15m[:-5]):
                if rsi > self.adaptive.rsi_overbought:
                    return "SELL"
        return None

    def check_double_pattern(self, current_price):
        candles = self.data.generate_candles(5, 50)
        if len(candles) < 30:
            return None
        lows = [c['low'] for c in candles[-30:]]
        highs = [c['high'] for c in candles[-30:]]
        if not lows or not highs:
            return None
        min1 = min(lows)
        idx1 = lows.index(min1)
        lows2 = lows[:idx1] + lows[idx1+1:]
        if not lows2:
            return None
        min2 = min(lows2)
        if abs(min1 - min2) / min1 < self.adaptive.double_pattern_tolerance and candles[-1]['close'] > min1 * 1.003:
            return "BUY"
        max1 = max(highs)
        idx1 = highs.index(max1)
        highs2 = highs[:idx1] + highs[idx1+1:]
        if not highs2:
            return None
        max2 = max(highs2)
        if abs(max1 - max2) / max1 < self.adaptive.double_pattern_tolerance and candles[-1]['close'] < max1 * 0.997:
            return "SELL"
        return None

    def get_signal(self, current_price):
        self.adaptive.adapt_parameters()
        pattern = self.adaptive.choose_pattern()
        self.last_pattern_used = pattern
        try:
            if pattern == 'engulfing':
                sig = self.check_engulfing_with_volume(current_price)
            elif pattern == 'divergence':
                sig = self.check_divergence(current_price)
            elif pattern == 'double':
                sig = self.check_double_pattern(current_price)
            else:
                sig = None
            if sig:
                trend = self.get_trend_direction()
                if sig == "BUY" and trend != "bullish":
                    return None
                if sig == "SELL" and trend != "bearish":
                    return None
                self.last_signal = f"{pattern.upper()} {sig}"
                return sig
        except Exception as e:
            log.error(f"Pattern error: {e}")
        return None


# ---------- Order Manager with Fixed Risk ($20) and Fixed Profit Target ($40) ----------
class AdaptiveOrderManager:
    def __init__(self, data_client, adaptive_params, live_mode=False):
        self.data = data_client
        self.adaptive = adaptive_params
        self.live_mode = live_mode

        # Trade state
        self.position = 0.0
        self.remaining_position = 0.0
        self.entry_price = 0.0
        self.entry_pattern = None
        self.trades = []
        self.last_trade_time = 0
        self.last_price_at_trade = 0.0

        # Fixed risk and reward (in USD)
        self.risk_usd = 20.0
        self.profit_usd = 40.0      # 2:1 reward-to-risk

        # Stop logic: dynamic ATR-based stop (widens in volatile markets)
        self.atr_stop_mult = 2.5     # initial stop = entry - (ATR * mult)
        self.atr_target_mult = 5.0   # target = entry + (ATR * mult) – gives ~2:1 reward

        # Cooldown and price filter
        self.cooldown_seconds = 60
        self.min_price_change_pct = 0.001

        # Daily loss limit (20% of balance, but we use fixed balance of $100)
        self.balance = 100.0
        self.daily_loss_limit_pct = 0.20
        self.daily_pnl = 0.0
        self.last_reset_day = datetime.now(timezone.utc).date()

        # Will be recalculated on each trade
        self.position_size_btc = 0.0
        self.stop_loss_price = 0.0
        self.profit_target_price = 0.0
        self.partial_taken = False
        self.highest_price = 0.0

    def get_atr(self, period=14) -> float:
        candles = self.data.generate_candles(5, period+1)
        if len(candles) < period+1:
            return 200.0
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        return sum(true_ranges[-period:]) / period

    def _reset_daily_pnl_if_needed(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_day:
            self.daily_pnl = 0.0
            self.last_reset_day = today
            log.info("Daily PnL reset to 0 (new trading day)")

    def _compute_position_size(self, entry_price, stop_distance):
        """
        Calculate position size (BTC) such that:
        position_size * stop_distance = risk_usd
        Also cap by available cash (balance / entry_price).
        """
        if stop_distance <= 0:
            return 0.0
        size_by_risk = self.risk_usd / stop_distance
        max_size_by_cash = self.balance / entry_price
        size = min(size_by_risk, max_size_by_cash)
        if size < 0.0001:
            log.warning(f"Calculated position size {size:.6f} BTC is below minimum; skipping trade.")
            return 0.0
        return size

    def can_enter(self, current_price) -> bool:
        self._reset_daily_pnl_if_needed()
        if self.balance <= 0.0:
            log.critical("Account balance is ZERO or NEGATIVE – trading permanently halted.")
            return False
        now = time.time()
        if self.position != 0.0:
            return False
        if now - self.last_trade_time < self.cooldown_seconds:
            return False
        if self.last_price_at_trade > 0:
            pct_change = abs((current_price - self.last_price_at_trade) / self.last_price_at_trade)
            if pct_change < self.min_price_change_pct:
                return False
        daily_limit_abs = self.balance * self.daily_loss_limit_pct
        if self.daily_pnl <= -daily_limit_abs:
            log.warning(f"Daily loss limit reached (lost {self.daily_pnl:.2f}, limit {daily_limit_abs:.2f})")
            return False
        return True

    def execute_buy(self, price: float, pattern: str):
        if not self.can_enter(price):
            return

        # Calculate ATR and dynamic stop distance (in USD per BTC)
        atr = self.get_atr()
        stop_distance = atr * self.atr_stop_mult
        target_distance = atr * self.atr_target_mult

        # Compute position size based on fixed risk ($20)
        size = self._compute_position_size(price, stop_distance)
        if size <= 0:
            log.warning("Could not compute valid position size – trade skipped.")
            return

        self.position_size_btc = size
        self.stop_loss_price = price - stop_distance
        self.profit_target_price = price + target_distance

        # Ensure stop and target are not too close (safety)
        if self.stop_loss_price <= 0 or self.profit_target_price <= price:
            log.warning("Invalid stop/target levels – skipping trade.")
            return

        actual_risk_usd = size * stop_distance
        actual_profit_usd = size * target_distance
        log.info(f"Trade parameters: price=${price:,.2f}, ATR={atr:.2f}")
        log.info(f"Stop distance = ${stop_distance:.2f} BTC, target distance = ${target_distance:.2f} BTC")
        log.info(f"Position size: {size:.6f} BTC (≈ ${size*price:.2f})")
        log.info(f"Actual risk: ${actual_risk_usd:.2f} (target ${self.risk_usd:.2f}), profit target: ${actual_profit_usd:.2f}")

        if self.live_mode:
            result = self.data.place_order("buy", size)
            if result:
                self.position = size
                self.remaining_position = size
                self.entry_price = price
                self.entry_pattern = pattern
                self.last_trade_time = time.time()
                self.last_price_at_trade = price
                self.highest_price = price
                self.partial_taken = False
                log.info(f"🔵 LIVE BUY {size:.6f} BTC @ ${price:,.2f} (Pattern: {pattern})")
            else:
                log.error("Live buy failed")
        else:
            self.position = size
            self.remaining_position = size
            self.entry_price = price
            self.entry_pattern = pattern
            self.last_trade_time = time.time()
            self.last_price_at_trade = price
            self.highest_price = price
            self.partial_taken = False
            log.info(f"🔵 SIM BUY {size:.6f} BTC @ ${price:,.2f} (Pattern: {pattern})")

    def update_exit(self, current_price):
        if self.position == 0.0:
            return False

        # Hard stop-loss
        if current_price <= self.stop_loss_price:
            log.info(f"🛑 Stop-loss hit (loss: ${self.risk_usd:.2f})")
            return True

        # Profit target
        if current_price >= self.profit_target_price:
            log.info(f"🎯 Profit target hit (gain: ${self.profit_usd:.2f})")
            return True

        # Update highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price

        # Trailing stop (optional, but keep for safety)
        atr = self.get_atr()
        trail_stop = self.highest_price - (atr * 1.5)
        if current_price <= trail_stop:
            log.info(f"🔻 Trailing stop hit (high: {self.highest_price:.2f}, stop: {trail_stop:.2f})")
            return True

        return False

    def execute_sell(self, price: float):
        if self.position == 0.0:
            return
        size = self.remaining_position
        pnl = (price - self.entry_price) * size
        self.balance += pnl
        self.daily_pnl += pnl
        self.trades.append({'price': price, 'size': size, 'pnl': pnl, 'pattern': self.entry_pattern})
        self.adaptive.record_trade_result(self.entry_pattern, pnl)
        log.info(f"🔴 SIM SELL {size:.6f} BTC @ ${price:,.2f} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f} | Daily PnL: ${self.daily_pnl:.2f}")
        self.position = 0.0
        self.remaining_position = 0.0
        self.entry_price = 0.0
        self.entry_pattern = None
        self.highest_price = 0.0
        self.partial_taken = False

    def force_exit(self, price):
        if self.position != 0.0:
            self.execute_sell(price)


# ---------- Main ----------
def main():
    log.info("===== High Probability Bot – Kraken PAPER TRADING (Fixed $20 risk / $40 profit) =====")
    log.info("Risk per trade: $20 | Profit target: $40 | ATR‑based stop (2.5x ATR)")
    log.info("Balance: $100 simulated (fixed)")

    # API keys optional
    api_key = os.environ.get("KRAKEN_API_KEY")
    api_secret = os.environ.get("KRAKEN_API_SECRET")
    if api_key and api_secret:
        log.info("🔑 API keys present (but live trading disabled)")
    else:
        log.info("🔑 No API keys – bot will run in paper mode only")

    LIVE_TRADING = False          # PAPER MODE
    SCAN_SECONDS = 10

    adaptive = AdaptiveParams()
    data = RealMarketData()
    orders = AdaptiveOrderManager(data, adaptive, live_mode=False)
    strategy = AdaptiveHighProbStrategy(data, adaptive)

    log.info(f"Cooldown: {orders.cooldown_seconds}s | Scan interval: {SCAN_SECONDS}s")

    try:
        while True:
            price = data.get_current_price()
            if orders.position == 0.0:
                if orders.can_enter(price):
                    signal = strategy.get_signal(price)
                    if signal == "BUY":
                        orders.execute_buy(price, strategy.last_pattern_used)
                else:
                    if int(time.time()) % 60 < SCAN_SECONDS:
                        log.info(f"⏳ Cooldown or price stale – no entry. BTC: ${price:,.2f}")
            else:
                if orders.update_exit(price):
                    orders.execute_sell(price)
                else:
                    if int(time.time()) % 30 < SCAN_SECONDS:
                        atr = orders.get_atr()
                        log.info(f"📈 Holding BTC @ ${price:,.2f} | Entry: ${orders.entry_price:,.2f} | Stop: ${orders.stop_loss_price:.2f} | Target: ${orders.profit_target_price:.2f} | ATR: {atr:.2f}")
            time.sleep(SCAN_SECONDS)
    except KeyboardInterrupt:
        log.info("Shutting down – closing position")
        if orders.position != 0.0:
            orders.force_exit(data.get_current_price())
        adaptive.save_state()


if __name__ == "__main__":
    main()
