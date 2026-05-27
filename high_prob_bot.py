#!/usr/bin/env python3
"""
High Probability Trading Bot – 24/7
Real Bitcoin price feed from CoinGecko (free, no API key)
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

# ---------- Real Bitcoin price feed (CoinGecko) ----------
class RealCryptoData:
    def __init__(self):
        self.last_price = 0.0
        self.url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

    def get_last_price(self) -> float:
        try:
            with urllib.request.urlopen(self.url, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                self.last_price = data['bitcoin']['usd']
                return self.last_price
        except Exception as e:
            log.error(f"Price fetch error: {e}")
            return self.last_price if self.last_price else 50000.0

    def get_previous_session_close(self) -> float:
        # For crypto, return 24h ago price (simplified – uses current price * 0.98)
        return self.get_last_price() * 0.98

    def generate_candles(self, timeframe_min: int, count: int) -> list:
        """
        Generate dummy candles for pattern detection.
        Uses current real price + random noise.
        For a real strategy, you would fetch historical candles from an API.
        """
        current = self.get_last_price()
        candles = []
        for i in range(count):
            # Add small random walk
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

# ---------- Strategy: High Probability Patterns ----------
class HighProbStrategy:
    def __init__(self, data_client):
        self.data = data_client
        self.last_signal = None

    def check_engulfing_with_volume(self) -> str | None:
        """Bullish/bearish engulfing at previous session close + volume spike."""
        candles = self.data.generate_candles(1, 5)   # 1‑minute candles for quick detection
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
        """Simplified RSI divergence (regular)."""
        # Generate two timeframes of price data
        prices_15m = [c['close'] for c in self.data.generate_candles(15, 30)]
        rsi_15m = self.calculate_rsi(prices_15m)
        # Bullish divergence: price lower low, RSI higher low
        if prices_15m[-1] < prices_15m[-5] and rsi_15m > self.calculate_rsi(prices_15m[:-5]):
            if rsi_15m < 35:
                return "BUY"
        # Bearish divergence: price higher high, RSI lower high
        if prices_15m[-1] > prices_15m[-5] and rsi_15m < self.calculate_rsi(prices_15m[:-5]):
            if rsi_15m > 65:
                return "SELL"
        return None

    def check_double_pattern(self) -> str | None:
        """Double bottom / double top on 5‑minute candles."""
        candles = self.data.generate_candles(5, 50)
        lows = [c['low'] for c in candles[-30:]]
        highs = [c['high'] for c in candles[-30:]]
        # Double bottom
        min1 = min(lows)
        idx1 = lows.index(min1)
        lows2 = lows[:idx1] + lows[idx1+1:]
        min2 = min(lows2)
        if abs(min1 - min2) / min1 < 0.005 and candles[-1]['close'] > min1 * 1.003:
            return "BUY"
        # Double top
        max1 = max(highs)
        idx1 = highs.index(max1)
        highs2 = highs[:idx1] + highs[idx1+1:]
        max2 = max(highs2)
        if abs(max1 - max2) / max1 < 0.005 and candles[-1]['close'] < max1 * 0.997:
            return "SELL"
        return None

    def get_signal(self) -> str | None:
        """Combine all patterns – return first triggered signal."""
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

# ---------- Simulated Order Manager (no real trades) ----------
class OrderManager:
    def __init__(self, data_client):
        self.data = data_client
        self.position = 0
        self.entry_price = 0.0
        self.balance = 400.0   # <--- STARTING BALANCE SET TO $400
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

# ---------- Main Loop (24/7) ----------
def main():
    log.info("High Probability Trading Bot – 24/7 Mode (Real Bitcoin Price)")
    data = RealCryptoData()
    strategy = HighProbStrategy(data)
    orders = OrderManager(data)

    CONTRACTS = 1      # 1 BTC (simulated)
    SCAN_SECONDS = 10  # check every 10 seconds

    try:
        while True:
            price = data.get_last_price()
            signal = strategy.get_signal()
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
Markets: NQ (simulated) – swap client for real broker later
Strategy: 3 high‑probability patterns (engulfing + divergence + double pattern)
"""

import time
import logging
import random
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict

# ---------- Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("high_prob_bot")

# ---------- Simulated Market Data (replace with real broker) ----------
class SimulatedData:
    def __init__(self):
        self.price = 18500.0
        self.trend = 0  # -1 down, 0 sideways, 1 up
        self.volatility = 0.005  # 0.5% per minute
        self.bars = []

    def generate_candles(self, timeframe_min: int, count: int) -> List[Dict]:
        """Generate realistic candles for a given timeframe."""
        candles = []
        for _ in range(count):
            # Random walk with momentum
            change = np.random.normal(0, self.volatility) * self.price
            if self.trend == 1:
                change += abs(change) * 0.5
            elif self.trend == -1:
                change -= abs(change) * 0.5
            open_p = self.price
            close_p = self.price + change
            high = max(open_p, close_p) + abs(change) * np.random.uniform(0.2, 0.8)
            low = min(open_p, close_p) - abs(change) * np.random.uniform(0.2, 0.8)
            volume = int(np.random.uniform(500, 20000))
            candles.append({
                'open': round(open_p, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_p, 2),
                'volume': volume,
                'timestamp': datetime.now()
            })
            self.price = close_p
        return candles

    def get_last_price(self) -> float:
        return self.price

    def get_previous_session_close(self) -> float:
        # Simulate a realistic level (last 4H close)
        return self.price - np.random.uniform(10, 50)

    def get_balance(self) -> float:
        return 50000.0  # dummy

# ---------- Strategy: High Probability Patterns ----------
class HighProbStrategy:
    def __init__(self, data_client):
        self.data = data_client
        self.last_signal = None

    def check_engulfing_with_volume(self, timeframe: str) -> Optional[str]:
        """Bullish/bearish engulfing at previous session close + volume spike."""
        candles = self.data.generate_candles(1, 5)  # 1‑minute for quick detection
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

    def calculate_rsi(self, prices: List[float], period=14) -> float:
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

    def check_divergence(self) -> Optional[str]:
        """Regular divergence on 15m and 1h (simulated)."""
        # Generate two timeframes of price data
        prices_15m = [c['close'] for c in self.data.generate_candles(15, 30)]
        prices_1h = [c['close'] for c in self.data.generate_candles(60, 10)]
        rsi_15m = self.calculate_rsi(prices_15m)
        rsi_1h = self.calculate_rsi(prices_1h)
        # Simplified divergence check:
        # Bullish: price lower low but RSI higher low (oversold)
        if prices_15m[-1] < prices_15m[-5] and rsi_15m > self.calculate_rsi(prices_15m[:-5]):
            if rsi_15m < 35:
                return "BUY"
        # Bearish: price higher high but RSI lower high (overbought)
        if prices_15m[-1] > prices_15m[-5] and rsi_15m < self.calculate_rsi(prices_15m[:-5]):
            if rsi_15m > 65:
                return "SELL"
        return None

    def check_double_pattern(self) -> Optional[str]:
        """Double bottom or double top on 5m."""
        candles = self.data.generate_candles(5, 50)
        lows = [c['low'] for c in candles[-30:]]
        highs = [c['high'] for c in candles[-30:]]
        # Double bottom: two distinct troughs within 20 candles, similar level
        # Very simplified: find min and second min
        min1 = min(lows)
        idx1 = lows.index(min1)
        lows2 = lows[:idx1] + lows[idx1+1:]
        min2 = min(lows2)
        if abs(min1 - min2) / min1 < 0.005:  # within 0.5%
            # Check recent price is above both (breakout)
            if candles[-1]['close'] > min1 * 1.003:
                return "BUY"
        # Double top
        max1 = max(highs)
        idx1 = highs.index(max1)
        highs2 = highs[:idx1] + highs[idx1+1:]
        max2 = max(highs2)
        if abs(max1 - max2) / max1 < 0.005 and candles[-1]['close'] < max1 * 0.997:
            return "SELL"
        return None

    def get_signal(self) -> Optional[str]:
        """Combine all patterns – if any triggers, return signal."""
        # 1. Engulfing + volume (highest probability)
        sig = self.check_engulfing_with_volume("1m")
        if sig:
            self.last_signal = f"ENGULFING {sig}"
            return sig
        # 2. RSI divergence
        sig = self.check_divergence()
        if sig:
            self.last_signal = f"DIVERGENCE {sig}"
            return sig
        # 3. Double pattern
        sig = self.check_double_pattern()
        if sig:
            self.last_signal = f"DOUBLE_PATTERN {sig}"
            return sig
        return None

# ---------- Order Manager (simulated) ----------
class OrderManager:
    def __init__(self, data_client):
        self.data = data_client
        self.position = 0
        self.entry_price = 0.0
        self.balance = 50000.0
        self.trades = []

    def execute_buy(self, price: float, contracts: int = 1):
        if self.position != 0:
            log.warning("Already in position, cannot buy")
            return
        self.position = contracts
        self.entry_price = price
        log.info(f"🔵 BUY {contracts} @ {price}")

    def execute_sell(self, price: float, contracts: int = 1):
        if self.position == 0:
            log.warning("No position to sell")
            return
        pnl = (price - self.entry_price) * contracts * 2.0  # $2 per point for MNQ
        self.balance += pnl
        self.trades.append({'price': price, 'pnl': pnl})
        log.info(f"🔴 SELL {contracts} @ {price} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f}")
        self.position = 0
        self.entry_price = 0.0

# ---------- Main Loop (24/7) ----------
def main():
    log.info("High Probability Trading Bot – 24/7 Mode")
    data = SimulatedData()
    strategy = HighProbStrategy(data)
    orders = OrderManager(data)

    # Trading parameters
    CONTRACTS = 1
    SCAN_SECONDS = 10  # check every 10 seconds (fast for demo)

    try:
        while True:
            price = data.get_last_price()
            signal = strategy.get_signal()
            if signal == "BUY" and orders.position == 0:
                orders.execute_buy(price, CONTRACTS)
            elif signal == "SELL" and orders.position != 0:
                orders.execute_sell(price, CONTRACTS)
            else:
                # Optional: print a heartbeat every 30 seconds
                if int(time.time()) % 30 == 0:
                    log.info(f"Price: {price:.2f} | Position: {orders.position} | Last signal: {strategy.last_signal}")
            time.sleep(SCAN_SECONDS)
    except KeyboardInterrupt:
        log.info("Bot stopped by user")
        # Close any open position
        if orders.position != 0:
            orders.execute_sell(data.get_last_price(), CONTRACTS)

if __name__ == "__main__":
    main()
