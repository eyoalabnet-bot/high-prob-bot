#!/usr/bin/env python3
"""
High Probability Trading Bot – Coinbase Advanced Trade (JWT, no passphrase)
- Real Bitcoin data from Coinbase
- Optional live trading (set LIVE_TRADING = True and add API keys)
- Adaptive pattern selection, trend filter (1h EMA), ATR trailing stop, partial profit taking
- Hard stop-loss 1%, cooldown 10 min, dynamic daily loss limit (20% of balance)
- Balance floor – stops trading if balance <= $0
- Paper trading by default, simulated balance $100
- Real position size: 0.0005 BTC (~$35) – safe for $100 account
"""

import time
import logging
import json
import os
import jwt
import requests
import numpy as np
import random
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("high_prob_bot")

# ---------- Coinbase API with JWT (no passphrase) ----------
class CoinbaseClient:
    def __init__(self, api_key, api_secret, use_sandbox=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coinbase.com" if not use_sandbox else "https://api-public.sandbox.exchange.coinbase.com"
        self.product = "BTC-USD"

    def _generate_jwt(self, method, request_path, body=""):
        uri = f"{method} {request_path}"
        if body:
            uri += body
        current_time = int(time.time())
        payload = {
            "sub": self.api_key,
            "iss": "coinbase-cloud",
            "nbf": current_time,
            "exp": current_time + 120,
            "uri": uri
        }
        token = jwt.encode(payload, self.api_secret, algorithm='ES256')
        return token

    def _request(self, method, path, body=None):
        url = self.base_url + path
        body_str = json.dumps(body) if body else ""
        token = self._generate_jwt(method, path, body_str)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        try:
            resp = requests.request(method, url, headers=headers, data=body_str, timeout=10)
            if resp.status_code in [200, 201]:
                return resp.json()
            else:
                log.error(f"Coinbase API error {resp.status_code}: {resp.text}")
        except Exception as e:
            log.error(f"Request error: {e}")
        return None

    def get_current_price(self):
        try:
            url = f"https://api.exchange.coinbase.com/products/{self.product}/ticker"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return float(resp.json()["price"])
        except Exception as e:
            log.warning(f"Price fetch error: {e}")
        return None

    def get_historical_candles(self, granularity: int, limit: int = 50):
        try:
            url = f"https://api.exchange.coinbase.com/products/{self.product}/candles"
            params = {"granularity": granularity, "limit": limit}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                candles = []
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
            log.warning(f"Candles error: {e}")
        return []

    def place_order(self, side: str, size: float, price=None):
        order_type = "MARKET" if price is None else "LIMIT"
        body = {
            "product_id": self.product,
            "side": side.upper(),
            "size": str(size),
            "type": order_type
        }
        if price:
            body["price"] = str(price)
            body["time_in_force"] = "GTC"
        return self._request("POST", "/api/v3/brokerage/orders", body)


# ---------- Market Data Wrapper ----------
class RealMarketData:
    def __init__(self, api_key=None, api_secret=None):
        self.coinbase = CoinbaseClient(api_key, api_secret) if api_key and api_secret else None
        self.cached_price = 50000.0
        self.last_price_fetch = 0
        self.price_cache_ttl = 5
        self.use_simulated_fallback = False

    def get_current_price(self) -> float:
        now = time.time()
        if now - self.last_price_fetch < self.price_cache_ttl:
            return self.cached_price
        price = None
        if self.coinbase:
            price = self.coinbase.get_current_price()
        if price is None:
            price = self.cached_price * (1 + random.uniform(-0.001, 0.001))
        self.cached_price = price
        self.last_price_fetch = now
        return price

    def get_historical_candles(self, granularity: int, limit: int = 50) -> list:
        candles = []
        if self.coinbase:
            candles = self.coinbase.get_historical_candles(granularity, limit)
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
                "volume": random.randint(100, 10000)
            })
            price = close
        return candles

    def get_previous_session_close(self) -> float:
        candles = self.get_historical_candles(3600, 2)
        if len(candles) >= 2:
            return candles[-2]["close"]
        return self.get_current_price() * 0.98

    def generate_candles(self, timeframe_min: int, count: int) -> list:
        granularity_map = {1: 60, 5: 300, 15: 900, 60: 3600, 240: 14400, 1440: 86400}
        if timeframe_min not in granularity_map:
            raise ValueError(f"Unsupported timeframe {timeframe_min}")
        granularity = granularity_map[timeframe_min]
        return self.get_historical_candles(granularity, count)

    def place_order(self, side: str, size: float, price=None):
        if not self.coinbase:
            log.error("No Coinbase client – cannot place real order")
            return None
        return self.coinbase.place_order(side, size, price)


# ---------- Adaptive Parameters (with persistence) ----------
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


# ---------- Strategy with Trend Filter, ATR, Partial Profit ----------
class AdaptiveHighProbStrategy:
    def __init__(self, data_client, adaptive_params):
        self.data = data_client
        self.adaptive = adaptive_params
        self.last_signal = None
        self.last_pattern_used = None

    # ----- Trend Filter: 1h EMA(50) -----
    def get_trend_direction(self) -> str:
        candles = self.data.generate_candles(60, 60)  # 60 = 1 hour
        if len(candles) < 50:
            return "neutral"
        closes = [c['close'] for c in candles[-50:]]
        ema = np.mean(closes)  # simple moving average for simplicity (or use EMA)
        current_price = closes[-1]
        if current_price > ema:
            return "bullish"
        elif current_price < ema:
            return "bearish"
        return "neutral"

    # ----- Pattern detection (unchanged) -----
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
                # Apply trend filter
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


# ---------- Order Manager: ATR trailing stop, partial profit taking ----------
class AdaptiveOrderManager:
    def __init__(self, data_client, adaptive_params, live_mode=False):
        self.data = data_client
        self.adaptive = adaptive_params
        self.live_mode = live_mode
        self.position = 0
        self.remaining_position = 0   # for partial profit
        self.entry_price = 0.0
        self.entry_pattern = None
        self.balance = 100.0
        self.trades = []
        self.last_trade_time = 0
        self.cooldown_seconds = 600        # 10 minutes
        self.last_price_at_trade = 0.0
        self.min_price_change_pct = 0.003
        self.stop_loss_pct = 0.01          # 1% hard stop
        self.partial_profit_pct = 0.015    # 1.5% take half
        self.highest_price = 0.0
        self.max_btc_size = 0.0005
        self.full_position_size = 1        # 1 BTC for sim; live uses max_btc_size
        self.partial_taken = False

        # Dynamic daily loss limit (20% of balance)
        self.daily_loss_limit_pct = 0.20
        self.daily_pnl = 0.0
        self.last_reset_day = datetime.now(timezone.utc).date()

    # ATR calculation (5m candles)
    def get_atr(self, period=14) -> float:
        candles = self.data.generate_candles(5, period+1)
        if len(candles) < period+1:
            return 200.0   # fallback $200
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

    def can_enter(self, current_price) -> bool:
        self._reset_daily_pnl_if_needed()
        if self.balance <= 0.0:
            log.critical("Account balance is ZERO or NEGATIVE – trading permanently halted.")
            return False
        now = time.time()
        if self.position != 0:
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

    def execute_buy(self, price: float, pattern: str, contracts: int = 1):
        if not self.can_enter(price):
            return
        if self.live_mode:
            result = self.data.place_order("buy", self.max_btc_size)
            if result and "order_id" in result:
                self.position = 1
                self.remaining_position = 1
                self.entry_price = price
                self.entry_pattern = pattern
                self.last_trade_time = time.time()
                self.last_price_at_trade = price
                self.highest_price = price
                self.partial_taken = False
                self.full_position_size = self.max_btc_size
                log.info(f"🔵 LIVE BUY {self.max_btc_size} BTC @ ${price:,.2f} (Pattern: {pattern})")
            else:
                log.error("Live buy failed")
        else:
            self.position = contracts
            self.remaining_position = contracts
            self.entry_price = price
            self.entry_pattern = pattern
            self.last_trade_time = time.time()
            self.last_price_at_trade = price
            self.highest_price = price
            self.partial_taken = False
            self.full_position_size = contracts
            log.info(f"🔵 SIM BUY {contracts} BTC @ ${price:,.2f} (Pattern: {pattern})")

    def update_exit(self, current_price):
        if self.position == 0:
            return False

        # Hard stop-loss (1% from entry)
        stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
        if current_price <= stop_loss_price:
            log.info(f"🛑 Stop-loss hit (1% loss from entry ${self.entry_price:,.2f})")
            return True

        # Update highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price

        # Partial profit taking (1.5% profit) – close half of remaining position
        partial_target = self.entry_price * (1 + self.partial_profit_pct)
        if not self.partial_taken and current_price >= partial_target and self.remaining_position > 0:
            # Close half of the remaining position
            half_size = self.remaining_position / 2
            if half_size < 0.0001:  # too small, ignore
                pass
            else:
                self.remaining_position -= half_size
                self.position = self.remaining_position
                log.info(f"💰 Partial profit taken at +{self.partial_profit_pct*100}% – closed {half_size} BTC, remaining {self.remaining_position} BTC")
                # For logging, we will record the partial PnL when the order is actually closed
                # We'll handle it as a separate sell event in execute_sell
                # For simplicity, we treat this as a signal to close half, but actual sell will happen in execute_sell
                # We'll set a flag and let the main loop handle it (or call execute_sell with size)
                # To keep it simple, we'll do the partial sell inside this method.
                # Note: we must call execute_sell with a specific size.
                self._partial_sell(current_price, half_size)
                self.partial_taken = True
                # Continue checking trailing stop for remaining position

        # Dynamic trailing stop using ATR (1.5×ATR from highest)
        atr = self.get_atr()
        trail_stop = self.highest_price - (atr * 1.5)
        if current_price <= trail_stop:
            log.info(f"🔻 Trailing stop hit (ATR={atr:.2f}, high: {self.highest_price:.2f}, stop: {trail_stop:.2f})")
            return True

        return False

    def _partial_sell(self, price: float, size: float):
        """Execute a partial sell (only for paper mode; live mode would need separate order)."""
        if self.live_mode:
            # For live, we would place a separate sell order for the partial amount
            result = self.data.place_order("sell", size)
            if result and "order_id" in result:
                pnl = (price - self.entry_price) * size
                self.balance += pnl
                self.daily_pnl += pnl
                log.info(f"🔴 PARTIAL LIVE SELL {size} BTC @ ${price:,.2f} | PnL: ${pnl:.2f} | Daily PnL: ${self.daily_pnl:.2f}")
            else:
                log.error("Partial live sell failed")
        else:
            pnl = (price - self.entry_price) * size
            self.balance += pnl
            self.daily_pnl += pnl
            self.trades.append({'price': price, 'pnl': pnl, 'pattern': self.entry_pattern, 'partial': True})
            self.adaptive.record_trade_result(self.entry_pattern, pnl)  # record partial profit as a win
            log.info(f"🔴 PARTIAL SIM SELL {size} BTC @ ${price:,.2f} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f} | Daily PnL: ${self.daily_pnl:.2f}")

    def execute_sell(self, price: float, contracts: int = 1):
        """Full sell of remaining position."""
        if self.position == 0:
            return
        size = self.remaining_position if not self.live_mode else self.max_btc_size
        if self.live_mode:
            result = self.data.place_order("sell", size)
            if result and "order_id" in result:
                pnl = (price - self.entry_price) * size
                self.balance += pnl
                self.daily_pnl += pnl
                log.info(f"🔴 LIVE SELL {size} BTC @ ${price:,.2f} | PnL: ${pnl:.2f} | Daily PnL: ${self.daily_pnl:.2f}")
            else:
                log.error("Live sell failed")
        else:
            pnl = (price - self.entry_price) * self.remaining_position
            self.balance += pnl
            self.daily_pnl += pnl
            self.trades.append({'price': price, 'pnl': pnl, 'pattern': self.entry_pattern, 'full': True})
            self.adaptive.record_trade_result(self.entry_pattern, pnl)
            log.info(f"🔴 SIM SELL {self.remaining_position} BTC @ ${price:,.2f} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f} | Daily PnL: ${self.daily_pnl:.2f} | Pattern: {self.entry_pattern}")
        self.position = 0
        self.remaining_position = 0
        self.entry_price = 0.0
        self.entry_pattern = None
        self.highest_price = 0.0

    def force_exit(self, price):
        if self.position != 0:
            self.execute_sell(price)


# ---------- Main ----------
def main():
    log.info("===== High Probability Bot – Upgraded with Trend Filter, ATR Trailing Stop, Partial Profit =====")
    log.info("Trend filter: 1h EMA(50) – trades only in direction of trend")
    log.info("Trailing stop: 1.5× ATR (dynamic)")
    log.info("Partial profit: take 50% at +1.5% profit, then trail the rest")
    log.info("Position size: 0.0005 BTC per trade (≈ $35 at current prices)")
    log.info("Dynamic daily loss limit: 20% of current balance")
    log.info("Balance floor: bot stops trading if simulated balance <= $0")

    LIVE_TRADING = False          # set to True to enable real orders
    SCAN_SECONDS = 30

    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")

    adaptive = AdaptiveParams()

    if LIVE_TRADING and api_key and api_secret:
        log.warning("⚠️ LIVE TRADING ENABLED – real orders will be placed on Coinbase")
        data = RealMarketData(api_key, api_secret)
        orders = AdaptiveOrderManager(data, adaptive, live_mode=True)
    else:
        if LIVE_TRADING and (not api_key or not api_secret):
            log.error("LIVE_TRADING is True but API keys missing. Falling back to paper trading.")
        log.info("🔒 PAPER TRADING MODE – no real orders")
        data = RealMarketData()
        orders = AdaptiveOrderManager(data, adaptive, live_mode=False)

    strategy = AdaptiveHighProbStrategy(data, adaptive)
    CONTRACTS = 1

    log.info(f"Hard stop-loss: {orders.stop_loss_pct*100}% | Partial profit: {orders.partial_profit_pct*100}% | ATR multiplier: 1.5")
    log.info(f"Cooldown: {orders.cooldown_seconds}s | Scan interval: {SCAN_SECONDS}s")

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
                        atr = orders.get_atr()
                        log.info(f"📈 Holding BTC @ ${price:,.2f} | Entry: ${orders.entry_price:,.2f} | High: ${orders.highest_price:,.2f} | Stop-loss: ${orders.entry_price * (1-orders.stop_loss_pct):,.2f} | ATR: {atr:.2f}")
            time.sleep(SCAN_SECONDS)
    except KeyboardInterrupt:
        log.info("Shutting down – closing position")
        if orders.position != 0:
            orders.force_exit(data.get_current_price())
        adaptive.save_state()


if __name__ == "__main__":
    main()
