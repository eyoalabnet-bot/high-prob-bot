#!/usr/bin/env python3
"""
High Probability Trading Bot – 24/7
Real Bitcoin price feed from CoinGecko (free)
Rate‑limit safe – fetches price once per cycle, caches for 30s.
All trading decisions are simulated – no real orders.
Starting balance: $400
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

# ---------- Real Bitcoin price feed with aggressive caching ----------
class RealCryptoData:
    def __init__(self):
        self.cached_price = 50000.0
        self.last_fetch_time = 0
        self.cache_ttl = 30  # seconds – never fetch more than once every 30s
        self.url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        self.error_count = 0

    def get_last_price(self, force=False) -> float:
        now = time.time()
        # Return cached price if still fresh and not forced
        if not force and (now - self.last_fetch_time) < self.cache_ttl:
            return self.cached_price

        try:
            with urllib.request.urlopen(self.url, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                self.cached_price = data['bitcoin']['usd']
                self.last_fetch_time = now
                self.error_count = 0
                # Increase TTL if we're getting close to rate limit
                if self.cache_ttl < 60:
                    self.cache_ttl = min(self.cache_ttl + 5, 60)
                return self.cached_price
        except urllib.error.HTTPError as e:
            if e.code == 429:
                self.error_count += 1
                # Exponential backoff: longer TTL after errors
                self.cache_ttl = min(self.cache_ttl + 10, 300)
                log.warning(f"Rate limit (429) – using cached price, new TTL={self.cache_ttl}s")
                return self.cached_price
            else:
                log.error(f"HTTP error: {e}")
                return self.cached_price
        except Exception as e:
            log.error(f"Price fetch error: {e}")
            return self.cached_price

    def get_previous_session_close(self) -> float:
        # Approximate previous close as current price * 0.98
        return self.get_last_price() * 0.98

    def generate_candles(self, timeframe_min: int, count: int, current_price=None) -> list:
        """
        Generate candles using a single price (no extra API calls).
        If current_price is provided, use it; otherwise fetch once.
        """
        if current_price is None:
            current_price = self.get_last_price()
        candles = []
        price = current_price
        for _ in range(count):
            # Random walk – realistic for pattern detection
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

# ---------- Strategy (unchanged but adapted to use passed price) ----------
class HighProbStrategy:
    def __init__(self, data_client):
        self.data = data_client
        self.last_signal = None

    def check_engulfing_with_volume(self, current_price) -> str | None:
        candles = self.data.generate_candles(1, 5, current_price=current_price)
        if len(candles) < 2:
            return None
        prev_close = self.data.get_previous_session_close()
        last = candles[-1]
        prev = candles[-2]
        avg_vol = np.mean([c['volume'] for c in candles[-5:]])
        vol_spike = last['volume'] > avg_vol * 1.5
        # Bullish engulfing
        if (prev['close'] < prev['open'] and
            last['close'] > last['open'] and
            last['close'] > prev['open'] and
            last['open'] < prev['close'] and
            last['low'] <= prev_close <= last['high'] and
            last['close'] > prev_close and
            vol_spike):
            return "BUY"
        # Bearish engulfing
        elif (prev['close'] > prev['open'] and
              last['close'] < last['open'] and
              last['close'] < prev['open'] and
              last['open'] > prev['close'] and
              last['high'] >= prev_close >= last['low'] and
              last['close'] < prev_close and
              vol_spike):
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

    def check_divergence(self, current_price) -> str | None:
        prices_15m = [c['close'] for c in self.data.generate_candles(15, 30, current_price=current_price)]
        rsi_15m = self.calculate_rsi(prices_15m)
        if prices_15m[-1] < prices_15m[-5] and rsi_15m > self.calculate_rsi(prices_15m[:-5]):
            if rsi_15m < 35:
                return "BUY"
        if prices_15m[-1] > prices_15m[-5] and rsi_15m < self.calculate_rsi(prices_15m[:-5]):
            if rsi_15m > 65:
                return "SELL"
        return None

    def check_double_pattern(self, current_price) -> str | None:
        candles = self.data.generate_candles(5, 50, current_price=current_price)
        lows = [c['low'] for c in candles[-30:]]
        highs = [c['high'] for c in candles[-30:]]
        min1 = min(lows)
        idx1 = lows.index(min1)
        lows2 = lows[:idx1] + lows[idx1+1:]
        min2 = min(lows2)
        if abs(min1 - min2) / min1 < 0.005 and candles[-1]['close'] > min1 * 1.003:
            return "BUY"
        max1 = max(highs)
        idx1 = highs.index(max1)
        highs2 = highs[:idx1] + highs[idx1+1:]
        max2 = max(highs2)
        if abs(max1 - max2) / max1 < 0.005 and candles[-1]['close'] < max1 * 0.997:
            return "SELL"
        return None

    def get_signal(self, current_price) -> str | None:
        sig = self.check_engulfing_with_volume(current_price)
        if sig:
            self.last_signal = f"ENGULFING {sig}"
            return sig
        sig = self.check_divergence(current_price)
        if sig:
            self.last_signal = f"DIVERGENCE {sig}"
            return sig
        sig = self.check_double_pattern(current_price)
        if sig:
            self.last_signal = f"DOUBLE_PATTERN {sig}"
            return sig
        return None

# ---------- Simulated Order Manager ----------
class OrderManager:
    def __init__(self, data_client):
        self.data = data_client
        self.position = 0
        self.entry_price = 0.0
        self.balance = 400.0
        self.trades = []

    def execute_buy(self, price: float, contracts: int = 1):
        if self.position != 0:
            log.warning("Already in position, cannot buy")
            return
        self.position = contracts
        self.entry_price = price
        log.info(f"🔵 BUY {contracts} BTC (simulated) @ ${price:,.2f}")

    def execute_sell(self, price: float, contracts: int = 1):
        if self.position == 0:
            log.warning("No position to sell")
            return
        pnl = (price - self.entry_price) * contracts
        self.balance += pnl
        self.trades.append({'price': price, 'pnl': pnl})
        log.info(f"🔴 SELL {contracts} BTC (simulated) @ ${price:,.2f} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f}")
        self.position = 0
        self.entry_price = 0.0

# ---------- Main Loop ----------
def main():
    log.info("High Probability Trading Bot – 24/7 Mode (Real Bitcoin Price, Rate‑Limit Safe)")
    data = RealCryptoData()
    strategy = HighProbStrategy(data)
    orders = OrderManager(data)

    CONTRACTS = 1
    SCAN_SECONDS = 15   # longer interval to be gentle on CoinGecko

    try:
        while True:
            # Fetch price once per cycle (cached internally)
            price = data.get_last_price()
            signal = strategy.get_signal(current_price=price)
            if signal == "BUY" and orders.position == 0:
                orders.execute_buy(price, CONTRACTS)
            elif signal == "SELL" and orders.position != 0:
                orders.execute_sell(price, CONTRACTS)
            else:
                # Heartbeat every 30 seconds
                if int(time.time()) % 30 < SCAN_SECONDS:
                    log.info(f"BTC: ${price:,.2f} | Position: {orders.position} | Last signal: {strategy.last_signal}")
            time.sleep(SCAN_SECONDS)
    except KeyboardInterrupt:
        log.info("Bot stopped by user")
        if orders.position != 0:
            orders.execute_sell(data.get_last_price(), CONTRACTS)

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
High Probability Trading Bot – 24/7
Real Bitcoin price feed from CoinGecko (free, with rate limit handling)
All trading decisions are simulated – no real orders.
Starting balance: $400
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

# ---------- Real Bitcoin price feed with caching & rate limit handling ----------
class RealCryptoData:
    def __init__(self):
        self.cached_price = 50000.0
        self.last_fetch_time = 0
        self.cache_ttl = 15  # seconds – avoid hitting rate limit
        self.url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

    def get_last_price(self) -> float:
        now = time.time()
        # Return cached price if still fresh
        if now - self.last_fetch_time < self.cache_ttl:
            return self.cached_price

        # Fetch new price
        try:
            with urllib.request.urlopen(self.url, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                self.cached_price = data['bitcoin']['usd']
                self.last_fetch_time = now
                return self.cached_price
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Too Many Requests
                log.warning("Rate limit hit – using cached price and slowing down")
                # Increase TTL temporarily to reduce requests
                self.cache_ttl = min(self.cache_ttl + 5, 60)
                return self.cached_price
            else:
                log.error(f"HTTP error: {e}")
                return self.cached_price
        except Exception as e:
            log.error(f"Price fetch error: {e}")
            return self.cached_price

    def get_previous_session_close(self) -> float:
        # Approximate previous close as current price * 0.98
        return self.get_last_price() * 0.98

    def generate_candles(self, timeframe_min: int, count: int) -> list:
        """Generate candles using cached price – no extra API calls."""
        current = self.get_last_price()  # uses cache
        candles = []
        for i in range(count):
            # Small random walk – realistic enough for pattern detection
            change = np.random.normal(0, 0.002) * current
            close = current + change
            high = max(current, close) + abs(change) * np.random.uniform(0.2, 0.8)
            low = min(current, close) - abs(change) * np.random.uniform(0.2, 0.8)
            candles.append({
                'open': round(current, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': int(np.random.uniform(100, 10000)),
                'timestamp': datetime.now()
            })
            current = close
        return candles

# ---------- Strategy: High Probability Patterns (unchanged) ----------
class HighProbStrategy:
    def __init__(self, data_client):
        self.data = data_client
        self.last_signal = None

    def check_engulfing_with_volume(self) -> str | None:
        candles = self.data.generate_candles(1, 5)
        if len(candles) < 2:
            return None
        prev_close = self.data.get_previous_session_close()
        last = candles[-1]
        prev = candles[-2]
        avg_vol = np.mean([c['volume'] for c in candles[-5:]])
        vol_spike = last['volume'] > avg_vol * 1.5
        # Bullish engulfing
        if (prev['close'] < prev['open'] and
            last['close'] > last['open'] and
            last['close'] > prev['open'] and
            last['open'] < prev['close'] and
            last['low'] <= prev_close <= last['high'] and
            last['close'] > prev_close and
            vol_spike):
            return "BUY"
        # Bearish engulfing
        elif (prev['close'] > prev['open'] and
              last['close'] < last['open'] and
              last['close'] < prev['open'] and
              last['open'] > prev['close'] and
              last['high'] >= prev_close >= last['low'] and
              last['close'] < prev_close and
              vol_spike):
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

    def check_divergence(self) -> str | None:
        prices_15m = [c['close'] for c in self.data.generate_candles(15, 30)]
        rsi_15m = self.calculate_rsi(prices_15m)
        if prices_15m[-1] < prices_15m[-5] and rsi_15m > self.calculate_rsi(prices_15m[:-5]):
            if rsi_15m < 35:
                return "BUY"
        if prices_15m[-1] > prices_15m[-5] and rsi_15m < self.calculate_rsi(prices_15m[:-5]):
            if rsi_15m > 65:
                return "SELL"
        return None

    def check_double_pattern(self) -> str | None:
        candles = self.data.generate_candles(5, 50)
        lows = [c['low'] for c in candles[-30:]]
        highs = [c['high'] for c in candles[-30:]]
        min1 = min(lows)
        idx1 = lows.index(min1)
        lows2 = lows[:idx1] + lows[idx1+1:]
        min2 = min(lows2)
        if abs(min1 - min2) / min1 < 0.005 and candles[-1]['close'] > min1 * 1.003:
            return "BUY"
        max1 = max(highs)
        idx1 = highs.index(max1)
        highs2 = highs[:idx1] + highs[idx1+1:]
        max2 = max(highs2)
        if abs(max1 - max2) / max1 < 0.005 and candles[-1]['close'] < max1 * 0.997:
            return "SELL"
        return None

    def get_signal(self) -> str | None:
        sig = self.check_engulfing_with_volume()
        if sig:
            self.last_signal = f"ENGULFING {sig}"
            return sig
        sig = self.check_divergence()
        if sig:
            self.last_signal = f"DIVERGENCE {sig}"
            return sig
        sig = self.check_double_pattern()
        if sig:
            self.last_signal = f"DOUBLE_PATTERN {sig}"
            return sig
        return None

# ---------- Simulated Order Manager ----------
class OrderManager:
    def __init__(self, data_client):
        self.data = data_client
        self.position = 0
        self.entry_price = 0.0
        self.balance = 400.0
        self.trades = []

    def execute_buy(self, price: float, contracts: int = 1):
        if self.position != 0:
            log.warning("Already in position, cannot buy")
            return
        self.position = contracts
        self.entry_price = price
        log.info(f"🔵 BUY {contracts} BTC (simulated) @ ${price:,.2f}")

    def execute_sell(self, price: float, contracts: int = 1):
        if self.position == 0:
            log.warning("No position to sell")
            return
        pnl = (price - self.entry_price) * contracts
        self.balance += pnl
        self.trades.append({'price': price, 'pnl': pnl})
        log.info(f"🔴 SELL {contracts} BTC (simulated) @ ${price:,.2f} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f}")
        self.position = 0
        self.entry_price = 0.0

# ---------- Main Loop ----------
def main():
    log.info("High Probability Trading Bot – 24/7 Mode (Real Bitcoin Price)")
    data = RealCryptoData()
    strategy = HighProbStrategy(data)
    orders = OrderManager(data)

    CONTRACTS = 1
    SCAN_SECONDS = 10

    try:
        while True:
            price = data.get_last_price()
            signal = strategy.get_signal()
            if signal == "BUY" and orders.position == 0:
                orders.execute_buy(price, CONTRACTS)
            elif signal == "SELL" and orders.position != 0:
                orders.execute_sell(price, CONTRACTS)
            else:
                if int(time.time()) % 30 < SCAN_SECONDS:
                    log.info(f"BTC: ${price:,.2f} | Position: {orders.position} | Last signal: {strategy.last_signal}")
            time.sleep(SCAN_SECONDS)
    except KeyboardInterrupt:
        log.info("Bot stopped by user")
        if orders.position != 0:
            orders.execute_sell(data.get_last_price(), CONTRACTS)

if __name__ == "__main__":
    main()
