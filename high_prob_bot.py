#!/usr/bin/env python3
"""
High Probability Trading Bot – Adaptive + Cooldown + Trailing Stop
- Real Bitcoin price (Binance, free API)
- Real 1m, 5m, 15m candles (no more fake patterns)
- Learns which patterns work in current market
- 5‑minute cooldown after each trade
- Minimum price movement filter (0.3%)
- Trailing stop (0.2% from highest price since entry)
- Profit target (1% from entry price)
- Simulated trading only – no real orders
- Starting balance: $400
"""

import time
import logging
import numpy as np
import requests
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("high_prob_bot")

# ---------- Real Market Data from Binance ----------
class RealMarketData:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.symbol = "BTCUSDT"
        self.cached_price = 0.0
        self.last_price_fetch = 0
        self.price_cache_ttl = 5  # seconds

    def _fetch_binance(self, endpoint, params=None):
        """Make a request to Binance public API (no API key needed)."""
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"Binance API error: {e}")
            return None

    def get_current_price(self) -> float:
        """Get real-time BTC price from Binance (cached 5 sec)."""
        now = time.time()
        if now - self.last_price_fetch < self.price_cache_ttl:
            return self.cached_price

        data = self._fetch_binance("ticker/price", {"symbol": self.symbol})
        if data and "price" in data:
            self.cached_price = float(data["price"])
            self.last_price_fetch = now
            return self.cached_price
        return self.cached_price if self.cached_price else 50000.0

    def get_historical_candles(self, interval: str, limit: int = 50) -> list:
        """Fetch real OHLCV candles from Binance.
        Interval: '1m', '5m', '15m', '1h', '4h', '1d' etc.
        Returns list of candles with keys: timestamp, open, high, low, close, volume.
        """
        data = self._fetch_binance("klines", {"symbol": self.symbol, "interval": interval, "limit": limit})
        if not data:
            return []
        candles = []
        for candle in data:
            candles.append({
                "timestamp": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })
        return candles

    def get_previous_session_close(self) -> float:
        """Return 1 hour close as previous session close."""
        candles = self.get_historical_candles("1h", 2)
        if len(candles) >= 2:
            return candles[-2]["close"]
        return self.get_current_price() * 0.98

    def generate_candles(self, timeframe_min: int, count: int, current_price=None) -> list:
        """
        Fetch REAL candles from Binance based on timeframe.
        This replaces the old random-walk simulation.
        """
        interval_map = {
            1: "1m",
            5: "5m",
            15: "15m",
            60: "1h",
            240: "4h",
            1440: "1d"
        }
        if timeframe_min not in interval_map:
            raise ValueError(f"Timeframe {timeframe_min} not supported. Use 1,5,15,60,240,1440")
        interval = interval_map[timeframe_min]
        return self.get_historical_candles(interval, count)

# ---------- Adaptive Strategy (same as before, but uses real candles) ----------
class AdaptiveParams:
    def __init__(self):
        self.pattern_weights = {'engulfing': 1.0, 'divergence': 1.0, 'double': 1.0}
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.volume_spike_factor = 1.5
        self.double_pattern_tolerance = 0.005
        self.win_counts = {p: 0 for p in self.pattern_weights}
        self.total_counts = {p: 0 for p in self.pattern_weights}
        self.recent_winrate = {p: 0.5 for p in self.pattern_weights}
        self.last_update_time = time.time()
        self.update_interval = 3600

    def record_trade_result(self, pattern_name: str, pnl: float):
        if pattern_name not in self.total_counts:
            return
        self.total_counts[pattern_name] += 1
        if pnl > 0:
            self.win_counts[pattern_name] += 1
        if self.total_counts[pattern_name] >= 5:
            self.recent_winrate[pattern_name] = self.win_counts[pattern_name] / self.total_counts[pattern_name]

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

class AdaptiveHighProbStrategy:
    def __init__(self, data_client, adaptive_params):
        self.data = data_client
        self.adaptive = adaptive_params
        self.last_signal = None
        self.last_pattern_used = None

    def check_engulfing_with_volume(self, current_price):
        candles = self.data.generate_candles(1, 5)  # 1-minute candles
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
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def check_divergence(self, current_price):
        prices_15m = [c['close'] for c in self.data.generate_candles(15, 30)]
        rsi = self.calculate_rsi(prices_15m)
        if prices_15m[-1] < prices_15m[-5] and rsi > self.calculate_rsi(prices_15m[:-5]):
            if rsi < self.adaptive.rsi_oversold:
                return "BUY"
        if prices_15m[-1] > prices_15m[-5] and rsi < self.calculate_rsi(prices_15m[:-5]):
            if rsi > self.adaptive.rsi_overbought:
                return "SELL"
        return None

    def check_double_pattern(self, current_price):
        candles = self.data.generate_candles(5, 50)
        lows = [c['low'] for c in candles[-30:]]
        highs = [c['high'] for c in candles[-30:]]
        min1 = min(lows)
        idx1 = lows.index(min1)
        lows2 = lows[:idx1] + lows[idx1+1:]
        min2 = min(lows2) if lows2 else min1
        if abs(min1 - min2) / min1 < self.adaptive.double_pattern_tolerance and candles[-1]['close'] > min1 * 1.003:
            return "BUY"
        max1 = max(highs)
        idx1 = highs.index(max1)
        highs2 = highs[:idx1] + highs[idx1+1:]
        max2 = max(highs2) if highs2 else max1
        if abs(max1 - max2) / max1 < self.adaptive.double_pattern_tolerance and candles[-1]['close'] < max1 * 0.997:
            return "SELL"
        return None

    def get_signal(self, current_price):
        self.adaptive.adapt_parameters()
        pattern = self.adaptive.choose_pattern()
        self.last_pattern_used = pattern
        if pattern == 'engulfing':
            sig = self.check_engulfing_with_volume(current_price)
        elif pattern == 'divergence':
            sig = self.check_divergence(current_price)
        elif pattern == 'double':
            sig = self.check_double_pattern(current_price)
        else:
            sig = None
        if sig:
            self.last_signal = f"{pattern.upper()} {sig}"
            return sig
        return None

# ---------- Simulated Order Manager (no real orders) ----------
class AdaptiveOrderManager:
    def __init__(self, data_client, adaptive_params):
        self.data = data_client
        self.adaptive = adaptive_params
        self.position = 0
        self.entry_price = 0.0
        self.entry_pattern = None
        self.balance = 400.0
        self.trades = []
        self.last_trade_time = 0
        self.cooldown_seconds = 300
        self.last_price_at_trade = 0.0
        self.min_price_change_pct = 0.003
        self.trailing_stop_pct = 0.002
        self.profit_target_pct = 0.01
        self.highest_price = 0.0

    def can_enter(self, current_price) -> bool:
        now = time.time()
        if self.position != 0:
            return False
        if now - self.last_trade_time < self.cooldown_seconds:
            return False
        if self.last_price_at_trade > 0:
            pct_change = abs((current_price - self.last_price_at_trade) / self.last_price_at_trade)
            if pct_change < self.min_price_change_pct:
                return False
        return True

    def execute_buy(self, price: float, pattern: str, contracts: int = 1):
        if not self.can_enter(price):
            return
        self.position = contracts
        self.entry_price = price
        self.entry_pattern = pattern
        self.last_trade_time = time.time()
        self.last_price_at_trade = price
        self.highest_price = price
        log.info(f"🔵 BUY {contracts} BTC (simulated) @ ${price:,.2f} (Pattern: {pattern})")

    def update_exit(self, current_price):
        """Return True if trailing stop or profit target is hit."""
        if self.position == 0:
            return False
        if current_price > self.highest_price:
            self.highest_price = current_price
        profit_target = self.entry_price * (1 + self.profit_target_pct)
        if current_price >= profit_target:
            log.info(f"Profit target hit {self.profit_target_pct*100}%")
            return True
        trail_stop = self.highest_price * (1 - self.trailing_stop_pct)
        if current_price <= trail_stop:
            log.info(f"Trailing stop hit (high: {self.highest_price:.2f}, stop: {trail_stop:.2f})")
            return True
        return False

    def execute_sell(self, price: float, contracts: int = 1):
        if self.position == 0:
            return
        pnl = (price - self.entry_price) * contracts
        self.balance += pnl
        self.trades.append({'price': price, 'pnl': pnl, 'pattern': self.entry_pattern})
        self.adaptive.record_trade_result(self.entry_pattern, pnl)
        log.info(f"🔴 SELL {contracts} BTC @ ${price:,.2f} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f} | Pattern: {self.entry_pattern}")
        self.position = 0
        self.entry_price = 0.0
        self.entry_pattern = None
        self.highest_price = 0.0

    def force_exit(self, price):
        if self.position != 0:
            self.execute_sell(price)

def main():
    log.info("Adaptive High Probability Bot – Real Binance Data")
    log.info("Trailing stop: 0.2% | Profit target: 1% | Cooldown: 5 min")
    data = RealMarketData()
    adaptive = AdaptiveParams()
    strategy = AdaptiveHighProbStrategy(data, adaptive)
    orders = AdaptiveOrderManager(data, adaptive)
    CONTRACTS = 1
    SCAN_SECONDS = 15

    try:
        while True:
            price = data.get_current_price()
            if orders.position == 0:
                if orders.can_enter(price):
                    signal = strategy.get_signal(price)
                    if signal == "BUY":
                        orders.execute_buy(price, strategy.last_pattern_used, CONTRACTS)
                else:
                    if int(time.time()) % 60 < SCAN_SECONDS:
                        log.info(f"Cooldown or price stale – no entry. BTC: ${price:,.2f}")
            else:
                if orders.update_exit(price):
                    orders.execute_sell(price, CONTRACTS)
                else:
                    if int(time.time()) % 30 < SCAN_SECONDS:
                        log.info(f"Holding BTC @ ${price:,.2f} | Entry: ${orders.entry_price:,.2f} | High: ${orders.highest_price:,.2f} | Pattern: {orders.entry_pattern}")
            time.sleep(SCAN_SECONDS)
    except KeyboardInterrupt:
        log.info("Shutting down – force closing any open position")
        if orders.position != 0:
            orders.force_exit(data.get_current_price())

if __name__ == "__main__":
    main()
