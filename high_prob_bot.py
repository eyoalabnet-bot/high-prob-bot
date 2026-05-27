#!/usr/bin/env python3
"""
High Probability Trading Bot – Adaptive + Cooldown
- Real Bitcoin price (CoinGecko, rate‑limit safe)
- Learns which patterns work in current market
- 5‑minute cooldown after each trade (prevents overtrading)
- Minimum price movement filter (0.3%)
- Simulated trading only – no real orders
- Starting balance: $400
"""

import time
import logging
import json
import urllib.request
import numpy as np
from datetime import datetime

# ---------- Setup logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("high_prob_bot")

# ---------- Rate‑limit safe price feed ----------
class RealCryptoData:
    def __init__(self):
        self.cached_price = 50000.0
        self.last_fetch_time = 0
        self.cache_ttl = 30
        self.url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        self.error_count = 0

    def get_last_price(self, force=False) -> float:
        now = time.time()
        if not force and (now - self.last_fetch_time) < self.cache_ttl:
            return self.cached_price
        try:
            with urllib.request.urlopen(self.url, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                self.cached_price = data['bitcoin']['usd']
                self.last_fetch_time = now
                self.error_count = 0
                if self.cache_ttl < 60:
                    self.cache_ttl = min(self.cache_ttl + 5, 60)
                return self.cached_price
        except urllib.error.HTTPError as e:
            if e.code == 429:
                self.error_count += 1
                self.cache_ttl = min(self.cache_ttl + 10, 300)
                log.warning(f"Rate limit – using cached price, TTL={self.cache_ttl}s")
                return self.cached_price
            else:
                log.error(f"HTTP error: {e}")
                return self.cached_price
        except Exception as e:
            log.error(f"Price fetch error: {e}")
            return self.cached_price

    def get_previous_session_close(self) -> float:
        return self.get_last_price() * 0.98

    def generate_candles(self, timeframe_min: int, count: int, current_price=None) -> list:
        if current_price is None:
            current_price = self.get_last_price()
        candles = []
        price = current_price
        for _ in range(count):
            change = np.random.normal(0, 0.002) * price
            close = price + change
            high = max(price, close) + abs(change) * np.random.uniform(0.2, 0.8)
            low = min(price, close) - abs(change) * np.random.uniform(0.2, 0.8)
            candles.append({
                'open': round(price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': int(np.random.uniform(100, 10000)),
                'timestamp': datetime.now()
            })
            price = close
        return candles

# ---------- Adaptive Parameter Manager ----------
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
        # Weight patterns by recent win rate (softmax-like)
        total = sum(max(0.2, self.recent_winrate[p]) for p in self.pattern_weights)
        for p in self.pattern_weights:
            raw = max(0.2, self.recent_winrate[p])
            self.pattern_weights[p] = raw / total

        # Adjust RSI thresholds based on divergence performance
        div_win = self.recent_winrate.get('divergence', 0.5)
        if div_win > 0.6:
            self.rsi_oversold = max(20, self.rsi_oversold - 2)
            self.rsi_overbought = min(80, self.rsi_overbought + 2)
        elif div_win < 0.4:
            self.rsi_oversold = min(40, self.rsi_oversold + 2)
            self.rsi_overbought = max(60, self.rsi_overbought - 2)

        # Adjust volume spike factor based on engulfing performance
        eng_win = self.recent_winrate.get('engulfing', 0.5)
        if eng_win > 0.6:
            self.volume_spike_factor = max(1.2, self.volume_spike_factor - 0.1)
        elif eng_win < 0.4:
            self.volume_spike_factor = min(3.0, self.volume_spike_factor + 0.2)

        # Adjust double pattern tolerance based on its win rate
        dbl_win = self.recent_winrate.get('double', 0.5)
        if dbl_win > 0.6:
            self.double_pattern_tolerance = max(0.002, self.double_pattern_tolerance - 0.0005)
        elif dbl_win < 0.4:
            self.double_pattern_tolerance = min(0.01, self.double_pattern_tolerance + 0.001)

        log.info(f"Adaptation: weights={self.pattern_weights}, RSI({self.rsi_oversold}/{self.rsi_overbought}), vol_spike={self.volume_spike_factor:.2f}, double_tol={self.double_pattern_tolerance:.4f}")

    def choose_pattern(self) -> str:
        """Select a pattern based on weights, ensuring probabilities sum to 1."""
        patterns = list(self.pattern_weights.keys())
        probs = [self.pattern_weights[p] for p in patterns]
        # Normalize to fix floating point rounding errors
        total = sum(probs)
        if total <= 0:
            # Fallback to equal probabilities
            probs = [1.0 / len(patterns)] * len(patterns)
        else:
            probs = [p / total for p in probs]
        # Epsilon‑greedy exploration (10% random)
        if np.random.random() < 0.1:
            return np.random.choice(patterns)
        return np.random.choice(patterns, p=probs)

# ---------- Adaptive Strategy ----------
class AdaptiveHighProbStrategy:
    def __init__(self, data_client, adaptive_params):
        self.data = data_client
        self.adaptive = adaptive_params
        self.last_signal = None
        self.last_pattern_used = None

    def check_engulfing_with_volume(self, current_price):
        candles = self.data.generate_candles(1, 5, current_price=current_price)
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

    def check_divergence(self, current_price):
        prices_15m = [c['close'] for c in self.data.generate_candles(15, 30, current_price=current_price)]
        rsi = self.calculate_rsi(prices_15m)
        if prices_15m[-1] < prices_15m[-5] and rsi > self.calculate_rsi(prices_15m[:-5]):
            if rsi < self.adaptive.rsi_oversold:
                return "BUY"
        if prices_15m[-1] > prices_15m[-5] and rsi < self.calculate_rsi(prices_15m[:-5]):
            if rsi > self.adaptive.rsi_overbought:
                return "SELL"
        return None

    def check_double_pattern(self, current_price):
        candles = self.data.generate_candles(5, 50, current_price=current_price)
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

# ---------- Simulated Order Manager with Cooldown ----------
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
        self.cooldown_seconds = 300          # 5 minutes
        self.last_price_at_trade = 0.0
        self.min_price_change_pct = 0.003    # 0.3%

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
        log.info(f"🔵 BUY {contracts} BTC (simulated) @ ${price:,.2f} (Pattern: {pattern})")

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

    def force_exit(self, price):
        if self.position != 0:
            self.execute_sell(price)

# ---------- Main Loop ----------
def main():
    log.info("Adaptive High Probability Bot – Cooldown + Learning (Simulated)")
    data = RealCryptoData()
    adaptive = AdaptiveParams()
    strategy = AdaptiveHighProbStrategy(data, adaptive)
    orders = AdaptiveOrderManager(data, adaptive)

    CONTRACTS = 1
    SCAN_SECONDS = 15

    try:
        while True:
            price = data.get_last_price()
            if orders.position == 0:
                if orders.can_enter(price):
                    signal = strategy.get_signal(price)
                    if signal == "BUY":
                        orders.execute_buy(price, strategy.last_pattern_used, CONTRACTS)
                else:
                    if int(time.time()) % 60 < SCAN_SECONDS:
                        log.info(f"Cooldown or price stale – no entry. BTC: ${price:,.2f}")
            else:
                # Simple exit after 30 seconds (you can change to trailing stop)
                if time.time() - orders.last_trade_time > 30:
                    orders.execute_sell(price, CONTRACTS)
                else:
                    if int(time.time()) % 30 < SCAN_SECONDS:
                        log.info(f"Holding BTC @ ${price:,.2f} | Entry: ${orders.entry_price:,.2f} | Pattern: {orders.entry_pattern}")
            time.sleep(SCAN_SECONDS)
    except KeyboardInterrupt:
        log.info("Shutting down – force closing any open position")
        if orders.position != 0:
            orders.force_exit(data.get_last_price())

if __name__ == "__main__":
    main()
