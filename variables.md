## 1. Volume History (Most Important Predictors)

Volume is **strongly autocorrelated**, so lagged volume features appear in nearly every model.

### Common Variables

| Variable | Definition |
|---|---|
| lag_volume_1 | volume in previous bin |
| lag_volume_2 | volume in previous two bins |
| daily_volume | total volume previous day |
| rolling_volume_mean | average volume last N days |
| rolling_volume_std | volatility of volume |
| cumulative_volume | volume since market open |

**Example features**

$V_{t-1},\; V_{t-2},\; \sum_{i=1}^{8} V_{t-i}$

---

## 2. Time-of-Day Variables (Capturing U-Shape)

Volume has a **very strong intraday seasonal pattern**.

### Typical Predictors

| Variable | Description |
|---|---|
| time_of_day | minute or bin index |
| opening_dummy | early session |
| midday_dummy | midday trading |
| close_dummy | closing period |
| sin_time | sinusoidal seasonality |
| cos_time | sinusoidal seasonality |

**Example encoding**

$
\sin\left(\frac{2\pi t}{T}\right), \quad
\cos\left(\frac{2\pi t}{T}\right)
$

---

## 3. Trade Activity Features

These capture **how active the market is**.

### Common Variables

| Variable | Definition |
|---|---|
| num_trades | number of trades in interval |
| avg_trade_size | mean trade size |
| median_trade_size | robust trade size |
| trade_notional | dollar value traded |
| trade_rate | trades per second |

**Example**

$
Trades_t = \sum_i 1
$

---

## 4. Buy/Sell Flow Features

Used to capture **directional pressure and liquidity demand**.

### Common Predictors

| Variable | Definition |
|---|---|
| buy_volume | shares bought |
| sell_volume | shares sold |
| buy_notional | buy volume × price |
| sell_notional | sell volume × price |
| trade_imbalance | buy − sell volume |

**Example imbalance**

$Imbalance_t = \frac{BuyVol_t - SellVol_t}{BuyVol_t + SellVol_t}$

---

## 5. Order Book Liquidity Features

These require **order book data (MBP-10 / LOBSTER)**.

### Common Predictors

| Variable | Definition |
|---|---|
| bid_depth | total size at bid |
| ask_depth | total size at ask |
| order_book_imbalance | depth imbalance |
| spread | ask − bid |
| midprice | (bid + ask)/2 |

**Example**

$OBI = \frac{BidDepth - AskDepth}{BidDepth + AskDepth}$

---

## 6. Price and Volatility Features

Volume and volatility are **strongly correlated**.

### Common Predictors

| Variable | Definition |
|---|---|
| return | price return |
| abs_return | magnitude of return |
| realized_volatility | high-frequency volatility |
| price_range | high − low |
| momentum | cumulative return |

**Example**

$\sigma_t = \sqrt{\sum r_i^2}$