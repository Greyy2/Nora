# Execute Module - The Final Gatekeeper

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING FLOW                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  REGIME OFF (Fast Path - Static Mode):                      │
│  ┌───────┐    ┌────────┐    ┌────────┐                     │
│  │Signal │ ─> │ Sizing │ ─> │ Broker │                     │
│  └───────┘    └────────┘    └────────┘                     │
│                                                              │
│  Characteristics:                                           │
│  • Speed: ~10ms latency                                     │
│  • Simple: Direct market orders                             │
│  • Use case: High-frequency, simple signals                 │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  REGIME ON (Smart Path - Dynamic Mode):                     │
│  ┌───────┐   ┌────────┐   ┌────────┐   ┌─────────┐   ┌────┐│
│  │Signal │─>│ Regime │─>│ Sizing │─>│ EXECUTE │─>│Brkr││
│  └───────┘   └────────┘   └────────┘   └─────────┘   └────┘│
│                                                              │
│  Characteristics:                                           │
│  • Intelligence: Market condition checks                    │
│  • Complexity: Multi-tranche sequencing                     │
│  • Safety: Spread/slippage validation                       │
│  • Use case: Regime-based tactical execution                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔑 Core Responsibilities

### 1. Live Market Validation (Physical Filters)

Execute is the **only** component that sees real-time market conditions through ticker data:

```python
ticker = {
    'bid': 99.975,      # Best buy price
    'ask': 100.025,     # Best sell price
    'quoteVolume': 5B   # 24h liquidity
}
```

**Checks Performed:**

- **Spread Check**: Reject if `(ask - bid) / mid_price > max_spread_pct`
  - Normal: 0.05% (5 basis points)
  - Alert: 0.2% (20 basis points) → Market Maker manipulation risk
  - Reject: 0.5%+ → Flash crash / liquidity crisis

- **Liquidity Check**: Ensure 24h volume > minimum threshold
  - Crypto: $100M+ daily volume (avoid pump & dump tokens)
  - Forex: Major pairs only during active sessions

### 2. Order Translation (API Formatter)

Regime and Sizer speak in **abstract concepts** (Pyramid_4_3_3, STOP orders). Brokers speak **API JSON**.

Execute translates:

```python
# Input (from Sizer):
OrderTranche(
    entry_price=100.0,
    sl_price=98.0,
    tp_price=104.0,
    trigger_type='MARKET',
    volume_lots=10.0
)

# Output (to Binance):
{
    'symbol': 'BTC/USDT',
    'type': 'market',
    'side': 'buy',
    'amount': 10.0,
    'params': {
        'stopLoss': {'triggerPrice': 98.0},
        'takeProfit': {'triggerPrice': 104.0}
    }
}

# Output (to MT5):
{
    'action': mt5.TRADE_ACTION_DEAL,
    'symbol': 'BTCUSD',
    'volume': 10.0,
    'type': mt5.ORDER_TYPE_BUY,
    'price': mt5.symbol_info_tick('BTCUSD').ask,
    'sl': 98.0,
    'tp': 104.0
}
```

### 3. Order Sequencing (Command Logic)

**Critical Principle**: Cannot fire all orders simultaneously. Must respect causality.

**Execution Order:**
1. **MARKET orders first** → Instant fill, establish position
2. **Wait for confirmation** → Check order status (filled/rejected)
3. **STOP orders second** → Breakout triggers, layered above market
4. **LIMIT orders last** → Pullback targets, layered below market

**Example - Pyramid_4_3_3 (LONG Strong):**
```
Time T+0:   Fire MARKET order (40% capital) → Entry: 100.0
Time T+1:   Confirm fill → Filled @ 100.02 (2 basis points slippage)
Time T+2:   Place STOP order (30% capital) → Trigger: 102.0
Time T+3:   Place STOP order (30% capital) → Trigger: 104.0
```

**Why This Matters:**
- If STOP orders go first, they trigger before position is established → Loss
- If network fails mid-execution, Emergency Cancel prevents half-filled positions

## 📊 Error Handling

### Retry Strategy

```python
config = ExecutionConfig(
    order_retry_attempts=3,
    retry_delay_ms=500  # Exponential backoff: 500ms, 1000ms, 1500ms
)
```

**Handled Errors:**
- **Network timeout**: Retry with ping to check if order already exists (prevent double entry)
- **Rate limit (429)**: Increase delay, backoff exponentially
- **Insufficient margin**: Reject immediately (don't retry, margin won't magically appear)

### Emergency Cancel Protocol

**Trigger Conditions:**
- Exception during Tranche 2/3 execution
- Network failure mid-sequence
- Partial fill with remaining orders stuck

**Action:**
```python
if config.enable_emergency_cancel:
    _emergency_cancel_all([order_id for order_id in executed_orders])
```

**Result:** All pending orders canceled, only filled orders remain (minimize loss)

## 🔧 Configuration

```python
ExecutionConfig(
    max_spread_pct=0.002,           # 20 basis points (0.2%)
    max_slippage_pct=0.001,         # 10 basis points (0.1%)
    
    order_retry_attempts=3,
    retry_delay_ms=500,
    
    market_order_timeout_sec=5,     # Wait 5s for market fill
    limit_order_timeout_sec=2,      # Wait 2s to confirm limit placed
    
    enable_emergency_cancel=True,
    min_liquidity_lots=10.0,        # Minimum order book depth
    
    rate_limit_delay_ms=100         # 100ms between orders (10 orders/sec max)
)
```

## 🚀 Usage Examples

### Example 1: Regime OFF (Direct to Broker)

```python
from position_sizer import PositionSizer

# Regime disabled → Execute not needed
sizer = PositionSizer(enable_regime_mode=False)

payload = sizer.process_signal(
    signal_state='LONG',
    current_price=100.0,
    atr=2.0
)

# Send directly to broker
for tranche in payload.tranches:
    broker_api.create_order(
        symbol='BTC/USDT',
        type='market',
        side='buy',
        amount=tranche.volume_lots
    )
```

### Example 2: Regime ON (Full Pipeline)

```python
from position_sizer import PositionSizer
from regime import analyze_regime_v3, format_regime_output
from regime.execute.execute import ExecutionCommander, ExecutionConfig

# Step 1: Analyze regime
regime_output = analyze_regime_v3(df, ema, atr, upper_band, lower_band)
regime_data = format_regime_output(regime_output)

# Step 2: Size position (Dynamic mode)
sizer = PositionSizer(enable_regime_mode=True)
payload = sizer.process_signal(
    signal_state=regime_data['current_state'],
    current_price=100.0,
    atr=2.0,
    regime_data=regime_data
)

# Step 3: Execute via Commander
config = ExecutionConfig(max_spread_pct=0.002)
commander = ExecutionCommander(
    exchange_api=ccxt_client,
    config=config,
    enable_live_checks=True
)

ticker = ccxt_client.fetch_ticker('BTC/USDT')

report = commander.execute_payload(
    payload=payload,
    symbol='BTC/USDT',
    current_ticker=ticker
)

# Step 4: Check results
if report.status == ExecutionStatus.SUCCESS:
    print(f"✅ All {len(report.executed_orders)} orders filled")
    print(f"Avg slippage: {report.average_slippage_pct:.4%}")
else:
    print(f"❌ Execution failed: {report.rejection_reason}")
```

### Example 3: Dry Run Mode (Testing)

```python
# Test strategy without risking capital
commander = ExecutionCommander(
    exchange_api=broker_api,
    dry_run=True  # Simulates execution, no real orders
)

report = commander.execute_payload(payload, 'BTC/USDT', ticker)

print(f"Simulated {len(report.executed_orders)} orders")
print(f"Estimated slippage: {report.average_slippage_pct:.4%}")
```

## 🔌 Broker Integration (Plug-and-Play)

### Binance (CCXT)

```python
import ccxt

binance = ccxt.binance({
    'apiKey': 'your_api_key',
    'secret': 'your_secret',
    'enableRateLimit': True
})

commander = ExecutionCommander(exchange_api=binance)
```

### MT5 (MetaTrader 5)

```python
import MetaTrader5 as mt5

class MT5Adapter:
    def create_order(self, **params):
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': params['symbol'].replace('/', ''),
            'volume': params['amount'],
            'type': mt5.ORDER_TYPE_BUY if params['side'] == 'buy' else mt5.ORDER_TYPE_SELL,
            'price': mt5.symbol_info_tick(params['symbol']).ask,
            'sl': params['params']['stopLoss']['triggerPrice'],
            'tp': params['params']['takeProfit']['triggerPrice']
        }
        result = mt5.order_send(request)
        return {'id': result.order, 'status': 'filled'}
    
    def fetch_order(self, order_id):
        return {'status': 'filled'}
    
    def cancel_order(self, order_id):
        return mt5.order_cancel(order_id)

commander = ExecutionCommander(exchange_api=MT5Adapter())
```

### Bybit (CCXT)

```python
import ccxt

bybit = ccxt.bybit({
    'apiKey': 'your_api_key',
    'secret': 'your_secret',
    'enableRateLimit': True
})

commander = ExecutionCommander(exchange_api=bybit)
```

**Key Insight**: Only `_format_broker_order()` needs to change. Core logic (validation, sequencing, retry) remains identical.

## 📈 Performance Metrics

**Typical Execution Timeline:**
```
0ms:    Receive payload from Sizer
1ms:    Validate market conditions (spread check)
2ms:    Sequence tranches (MARKET → STOP → LIMIT)
---
10ms:   Fire Tranche 1 (MARKET order)
500ms:  Wait for fill confirmation
510ms:  Fire Tranche 2 (STOP order)
610ms:  Fire Tranche 3 (STOP order)
---
710ms:  TOTAL EXECUTION TIME
```

**Slippage Tracking:**
```python
report.average_slippage_pct = 0.08%  # 8 basis points

# Compare to expected:
# Market impact < 0.1% → Excellent
# Market impact 0.1-0.3% → Normal
# Market impact > 0.5% → Poor execution (wide spread)
```

## 🛡️ Safety Guarantees

1. **No Double Entry**: Network timeout → Ping broker to check order status before retry
2. **No Orphan Orders**: Emergency cancel ensures all-or-nothing execution
3. **No Spread Traps**: Reject execution if spread > max threshold
4. **No Slippage Bombs**: Track actual fill price vs expected, fail if slippage > limit
5. **No Rate Limit Bans**: Configurable delay between orders (100ms default)

## 🎯 Design Philosophy

**Single Responsibility Principle:**
- **Regime**: Answers "What is the market doing?"
- **Sizer**: Answers "How much should I bet?"
- **Execute**: Answers "Can I safely place this bet right now?"

**Execute does NOT:**
- ❌ Recalculate position sizes (Sizer's job)
- ❌ Analyze market regime (Regime's job)
- ❌ Generate signals (Signal's job)

**Execute DOES:**
- ✅ Validate live market conditions
- ✅ Translate abstract orders to broker API
- ✅ Sequence orders correctly
- ✅ Handle network failures gracefully

## 📝 Execution Report

```python
@dataclass
class ExecutionReport:
    status: ExecutionStatus              # SUCCESS / PARTIAL / FAILED / REJECTED
    executed_orders: List[Dict]          # List of confirmed order IDs
    failed_tranches: List[int]           # Tranche IDs that failed
    
    total_filled_lots: float             # Sum of successfully filled volume
    total_rejected_lots: float           # Sum of rejected volume
    
    rejection_reason: Optional[str]      # Why execution was blocked
    warnings: List[str]                  # Non-critical issues
    
    execution_time_ms: float             # Total time from start to finish
    average_slippage_pct: float          # Average (fill_price - expected) / expected
```

## 🚦 Status Codes

| Status | Meaning | Action |
|--------|---------|--------|
| `SUCCESS` | All tranches filled | Continue to next signal |
| `PARTIAL` | Some tranches filled, some failed | Evaluate risk, may cancel remaining |
| `FAILED` | No tranches filled | Retry with adjusted parameters |
| `REJECTED` | Pre-trade checks failed (spread/liquidity) | Wait for better market conditions |

## 🔍 Debugging

```python
import logging

# Enable debug logs
logger = logging.getLogger("ExecutionCommander")
logger.setLevel(logging.DEBUG)

# Output:
# [DEBUG] Attempt 1/3: {'symbol': 'BTC/USDT', 'type': 'market', ...}
# [INFO] ✅ Tranche 1 executed: Order ID ORDER_12345
# [WARNING] Attempt 2 failed: Rate limit exceeded
# [ERROR] ❌ Tranche 3 failed: Insufficient margin
```

## 🎓 When to Use Execute vs Direct Broker

### Use Execute (Regime ON):
- ✅ Complex multi-tranche strategies (Pyramid_4_3_3, Accumulate_3_3_4)
- ✅ Need spread/slippage validation
- ✅ Want emergency cancel safety
- ✅ Trading during volatile hours (news events)

### Direct to Broker (Regime OFF):
- ✅ Simple single-order strategies
- ✅ High-frequency signals (< 1 second latency required)
- ✅ Known liquid markets (BTC/USDT on Binance)
- ✅ Backtesting mode (no live checks needed)

---

**Execute is the bridge between theoretical strategy and real-world execution. It's the last line of defense against market manipulation, technical failures, and execution risk.**
