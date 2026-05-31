#!/usr/bin/env python3
"""
High Probability Trading Bot – Coinbase Advanced Trade
- Real Bitcoin data from Coinbase (public endpoints for candles/price)
- Optional live trading on Coinbase Advanced Trade
- Adaptive pattern selection, trailing stop, profit target
- Paper trading by default (set LIVE_TRADING = True and add keys)
- Starting simulated balance: $100
"""

import time
import logging
import json
import os
import hmac
import hashlib
import base64
import numpy as np
import requests
import random
from datetime import datetime, timezone
from urllib.parse import urlencode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("high_prob_bot")


# ---------- Real Market Data from Coinbase (Public) + Fallback ----------
class RealMarketData:
    def __init__(self, api_key=None, api_secret=None, passphrase=None):
        self.public_url = "https://api.exchange.coinbase.com"
        self.auth_url = "https://api.exchange.coinbase.com"  # same for authenticated endpoints
        self.product = "BTC-USD"
        self.cached_price = 50000.0
        self.last_price_fetch = 0
        self.price_cache_ttl = 5
        # For live trading (Coinbase Advanced Trade)
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

    # ---------- Public endpoints (no key) ----------
    def get_current_price(self) -> float:
        now = time.time()
        if now - self.last_price_fetch < self.price_cache_ttl:
            return self.cached_price
        try:
            url = f"{self.public_url}/products/{self.product}/ticker"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.cached_price = float(data["price"])
                self.last_price_fetch = now
                return self.cached_price
        except Exception as e:
            log.warning(f"Coinbase price error: {e}, using cached")
        return self.cached_price

    def get_historical_candles(self, granularity: int, limit: int = 50) -> list:
        """
        Coinbase granularity in seconds: 60 (1m), 300 (5m), 900 (15m), 3600 (1h), 21600 (6h), 86400 (1d)
        """
        try:
            url = f"{self.public_url}/products/{self.product}/candles"
            params = {"granularity": granularity, "limit": limit}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                candles = []
                # Coinbase returns: [time, low, high, open, close, volume]
                for item in data:
                    candles.append({
                        "timestamp": item[0],
                        "open": float(item[3]),
                        "high": float(item[2]),
                        "low": float(item[1]),
                        "close": float(item[4]),
                        "volume": float(item[5])
                    })
                return candles
        except Exception as e:
            log.warning(f"Coinbase candles error: {e}, using fallback")
        return self._generate_simulated_candles(limit)

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
                "volume": random.randint(100, 10000)
            })
            price = close
        return candles

    def get_previous_session_close(self) -> float:
        candles = self.get_historical_candles(3600, 2)  # 1h
        if len(candles) >= 2:
            return candles[-2]["close"]
        return self.get_current_price() * 0.98

    def generate_candles(self, timeframe_min: int, count: int) -> list:
        # Map minutes to Coinbase granularity (seconds)
        granularity_map = {
            1: 60, 5: 300, 15: 900, 60: 3600, 240: 14400, 1440: 86400
        }
        if timeframe_min not in granularity_map:
            raise ValueError(f"Unsupported timeframe {timeframe_min}")
        granularity = granularity_map[timeframe_min]
        return self.get_historical_candles(granularity, count)

    # ---------- Live order placement (Coinbase Advanced Trade) ----------
    def _generate_signature(self, method, path, body=""):
        timestamp = str(int(time.time()))
        message = timestamp + method + path + body
        signature = base64.b64encode(
            hmac.new(self.api_secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()
        ).decode('utf-8')
        return timestamp, signature

    def _request(self, method, path, body=None):
        if not self.api_key or not self.api_secret:
            log.error("API keys missing – cannot place real order")
            return None
        url = self.auth_url + path
        body_str = json.dumps(body) if body else ""
        timestamp, signature = self._generate_signature(method, path, body_str)
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
        try:
            resp = requests.request(method, url, headers=headers, data=body_str, timeout=10)
            if resp.status_code in [200, 201]:
                return resp.json()
            else:
                log.error(f"Coinbase API error {resp.status_code}: {resp.text}")
        except Exception as e:
            log.error(f"Order request error: {e}")
        return None

    def place_order(self, side: str, size: float, price=None):
        """
        side: 'buy' or 'sell'
        size: amount of BTC (e.g., 0.001)
        price: optional limit price (if None, market order)
        """
        order_type = "market" if price is None else "limit"
        body = {
            "product_id": self.product,
            "side": side,
            "size": str(size),
            "type": order_type
        }
        if price:
            body["price"] = str(price)
        path = "/orders"
        result = self._request("POST", path, body)
        if result and "id" in result:
            log.info(f"✅ Live {side.upper()} order placed: {size} BTC")
            return result
        else:
            log.error(f"Live order failed: {result}")
            return None


# ---------- Adaptive Parameters with Persistence (unchanged) ----------
class AdaptiveParams:
    def __init__(self, state_file="learning_state.json"):
        self.state_file = state_file
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


# ---------- Strategy (unchanged from previous version) ----------
class AdaptiveHighProbStrategy:
    def __init__(self, data_client, adaptive_params):
        self.data = data_client
        self.adaptive = adaptive_params
        self.last_signal = None
        self.last_pattern_used = None

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
                self.last_signal = f"{pattern.upper()} {sig}"
                return sig
        except Exception as e:
            log.error(f"Pattern error: {e}")
        return None


# ---------- Order Manager (Simulated + Live) – Starting balance $100 ----------
class AdaptiveOrderManager:
    def __init__(self, data_client, adaptive_params, live_mode=False):
        self.data = data_client
        self.adaptive = adaptive_params
        self.live_mode = live_mode
        self.position = 0
        self.entry_price = 0.0
        self.entry_pattern = None
        self.balance = 100.0               # simulated starting balance
        self.trades = []
        self.last_trade_time = 0
        self.cooldown_seconds = 300
        self.last_price_at_trade = 0.0
        self.min_price_change_pct = 0.003
        self.trailing_stop_pct = 0.002
        self.profit_target_pct = 0.01
        self.highest_price = 0.0
        self.max_btc_size = 0.001          # ~$70 – adjust as needed

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
        if self.live_mode:
            result = self.data.place_order("buy", self.max_btc_size)
            if result and "id" in result:
                self.position = 1
                self.entry_price = price
                self.entry_pattern = pattern
                self.last_trade_time = time.time()
                self.last_price_at_trade = price
                self.highest_price = price
                log.info(f"🔵 LIVE BUY {self.max_btc_size} BTC @ ${price:,.2f} (Pattern: {pattern})")
            else:
                log.error("Live buy failed, not entering simulated position")
        else:
            self.position = contracts
            self.entry_price = price
            self.entry_pattern = pattern
            self.last_trade_time = time.time()
            self.last_price_at_trade = price
            self.highest_price = price
            log.info(f"🔵 SIM BUY {contracts} BTC @ ${price:,.2f} (Pattern: {pattern})")

    def update_exit(self, current_price):
        if self.position == 0:
            return False
        if current_price > self.highest_price:
            self.highest_price = current_price
        profit_target = self.entry_price * (1 + self.profit_target_pct)
        if current_price >= profit_target:
            log.info(f"✅ Profit target hit ({self.profit_target_pct*100}%)")
            return True
        trail_stop = self.highest_price * (1 - self.trailing_stop_pct)
        if current_price <= trail_stop:
            log.info(f"🔻 Trailing stop hit (high: {self.highest_price:.2f}, stop: {trail_stop:.2f})")
            return True
        return False

    def execute_sell(self, price: float, contracts: int = 1):
        if self.position == 0:
            return
        if self.live_mode:
            result = self.data.place_order("sell", self.max_btc_size)
            if result and "id" in result:
                pnl = (price - self.entry_price) * self.max_btc_size
                self.balance += pnl   # local tracking only
                log.info(f"🔴 LIVE SELL {self.max_btc_size} BTC @ ${price:,.2f} | PnL: ${pnl:.2f}")
            else:
                log.error("Live sell failed")
        else:
            pnl = (price - self.entry_price) * contracts
            self.balance += pnl
            self.trades.append({'price': price, 'pnl': pnl, 'pattern': self.entry_pattern})
            self.adaptive.record_trade_result(self.entry_pattern, pnl)
            log.info(f"🔴 SIM SELL {contracts} BTC @ ${price:,.2f} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f} | Pattern: {self.entry_pattern}")
        self.position = 0
        self.entry_price = 0.0
        self.entry_pattern = None
        self.highest_price = 0.0

    def force_exit(self, price):
        if self.position != 0:
            self.execute_sell(price)


# ---------- Main ----------
def main():
    log.info("===== High Probability Bot – Coinbase Advanced Trade =====")

    # --- CONFIGURATION ---
    LIVE_TRADING = False          # change to True to enable real orders
    SCAN_SECONDS = 30

    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    api_passphrase = os.environ.get("COINBASE_API_PASSPHRASE")

    adaptive = AdaptiveParams()

    if LIVE_TRADING and all([api_key, api_secret, api_passphrase]):
        log.warning("⚠️ LIVE TRADING ENABLED – real orders will be placed on Coinbase")
        data = RealMarketData(api_key, api_secret, api_passphrase)
        orders = AdaptiveOrderManager(data, adaptive, live_mode=True)
    else:
        if LIVE_TRADING and not all([api_key, api_secret, api_passphrase]):
            log.error("LIVE_TRADING is True but API keys missing. Falling back to paper trading.")
        log.info("🔒 PAPER TRADING MODE – no real orders")
        data = RealMarketData()
        orders = AdaptiveOrderManager(data, adaptive, live_mode=False)

    strategy = AdaptiveHighProbStrategy(data, adaptive)
    CONTRACTS = 1

    log.info(f"Trailing stop: {orders.trailing_stop_pct*100}% | Profit target: {orders.profit_target_pct*100}% | Cooldown: {orders.cooldown_seconds}s")
    log.info(f"Scan interval: {SCAN_SECONDS}s")

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
                        log.info(f"⏳ Cooldown or price stale – no entry. BTC: ${price:,.2f}")
            else:
                if orders.update_exit(price):
                    orders.execute_sell(price, CONTRACTS)
                else:
                    if int(time.time()) % 30 < SCAN_SECONDS:
                        log.info(f"📈 Holding BTC @ ${price:,.2f} | Entry: ${orders.entry_price:,.2f} | High: ${orders.highest_price:,.2f} | Pattern: {orders.entry_pattern}")
            time.sleep(SCAN_SECONDS)
    except KeyboardInterrupt:
        log.info("Shutting down – closing position")
        if orders.position != 0:
            orders.force_exit(data.get_current_price())
        adaptive.save_state()


if __name__ == "__main__":
    main()
