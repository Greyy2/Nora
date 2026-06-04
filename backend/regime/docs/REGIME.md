# REGIME V3 - Quy Trình Thực Thi

## 🎯 Triết Lý Cốt Lõi

**Regime = Máy Dự Đoán Xu Thế** + **Máy Xác Thực Signal** + **Máy Tính Xác Suất**

- **KHÔNG** tự quyết định vào lệnh
- **LUÔN** phân loại & dự đoán thị trường (quá khứ/hiện tại/tương lai)
- **CHỈ** xác thực signal khi có tín hiệu đến
- **ĐƯA RA** danh sách xác suất chiến lược cho Execute quyết định

---

## 📊 Quy Trình 3 Bước
REGIME = "Máy Dự Đoán Xu Thế" + "Máy Xác Thực Signal" + "Máy Tính Xác Suất"

┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 0: Phân Loại & Dự Đoán (Luôn Chạy)                     │
│  - Quá khứ: Long/Short/Sideway zones                        │
│  - Tương lai: Xu thế dài hạn (không theo nến ngắn)          │
│  - Hiển thị: Chart background colors                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
  ┌───────────┐         ┌──────────────┐
  │ Sideway?  │         │ Signal đến?  │
  └─────┬─────┘         └──────┬───────┘
        │                      │
       YES                    YES
        │                      │
        ▼                      ▼
┌─────────────────┐   ┌────────────────────────────────────────┐
│ Tính Khả Thi    │   │ BƯỚC 1: Xác Thực Signal                │
│ Sideway:        │   │  - So sánh: Regime vs Signal → Khớp?   │
│  - Mean Revert  │   │  - Confidence đủ cao? (>0.70)          │
│    xác suất     │   │  - Volume confirmation                 │
│  - Breakout     │   └────────────────┬───────────────────────┘
│    potential    │                    │
│  - Zone bound   │          ┌─────────▼─────────┐
│    strength     │          │  Signal khớp?     │
│                 │          │  (matches_regime) │
│ Output:         │          └──┬────────────┬───┘
│ - feasible?     │             │           YES 
│ - mean_revert   │            NO            │
│   _prob         │             │            │
│ - breakout_     │             ▼            ▼
│   prob          │       ┌─────────┐  ┌────────────┐
└────────┬────────┘       │ REJECT  │  │  is_strong?│
         │                │ (allow= │  │(Trend mạnh │
  ┌──────▼──────┐         │  False) │  │  hay yếu?) │
  │  Feasible?  │         └─────────┘  └──┬─────────┘
  └──┬──────┬───┘                         │   
     │     YES                            │
    NO      │                             │
     │      │                             │
     ▼      │                             │
┌─────────┐ │                             │
│  WAIT   │ │                             │
│ (Chờ TT)│ │                             │
└─────────┘ │                             │
            │                             │
            └─────────────────┬───────────┘
                              │
                              ▼
            ┌─────────────────────────────────────────┐
            │ BƯỚC 2: Tính Xác Suất Chiến Lược        │
            │  Input: Sideway feasible HOẶC Signal OK │
            │  - LONG/SHORT mạnh → Trend strategies   │
            │  - SIDEWAY → Mean Revert/Scalp          │
            │  - Risk điều chỉnh (không vượt max)     │
            │  Output: List[{strategy, probability,   │
            │          risk, winrate}]                │
            └─────────────────┬───────────────────────┘
                              │
                              ▼
                     [Execute quyết định]
### **BƯỚC 0: Phân Loại & Dự Đoán** (Luôn Chạy)

**Input**: 
- OHLCV data
- EMA, ATR, Keltner Bands
- RSI, Volume (optional)

**Xử Lý**:
1. **Phân loại quá khứ** → Chia data thành zones (Long/Short/Sideway)
2. **Phát hiện hiện tại** → Current state từ price vs EMA
3. **Dự đoán tương lai** → Next state từ momentum + zone pattern
4. **Tính độ tin cậy** → Confidence score dựa vào zone stability
5. **Phân tích sức mạnh** → Trend strength (Strong/Weak/Neutral)

**Output**:
```python
RegimePrediction {
    current_state: LONG/SHORT/SIDEWAY
    next_state: LONG/SHORT/SIDEWAY
    confidence: 0.0-1.0
    zones: [MarketZone, ...]
    trend_strength: STRONG/WEAK/NEUTRAL
    is_strong: bool
    strength_score: 0.0-1.0
}
```

**Hiển thị**: Chart background color (bull=green, bear=red, side=gray)

---

### **BƯỚC 1: Xác Thực Signal** (Khi Signal Đến)

**Trigger**: `signal_long=True` hoặc `signal_short=True`

**Xử Lý**:
1. **So sánh** → Signal state vs Regime state
2. **Kiểm tra** → Confidence đủ cao? (>0.70)
3. **Xác nhận** → Trend đủ mạnh? (is_strong)
4. **Phát hiện rủi ro** → Volatility, Volume confirmation
5. **Tính trust score** → Trung bình các thành phần tin cậy

**Output**:
```python
SignalValidation {
    is_valid: bool              # True nếu trust >= 0.65 AND matches
    matches_regime: bool         # Regime khớp với Signal?
    trust_score: 0.0-1.0
    regime_says: LONG/SHORT/SIDEWAY
    signal_says: LONG/SHORT/SIDEWAY
    warnings: [str, ...]
}
```

**Logic Quyết Định**:
```
IF not is_valid OR not matches_regime:
    → REJECT signal (allow_long=False, allow_short=False)
ELSE:
    → ACCEPT signal → Tiếp tục Bước 2
```

---

### **BƯỚC 2: Tính Xác Suất Chiến Lược** (Sau Khi Xác Thực OK)

**Input**: 
- Prediction (từ Bước 0)
- Validation (từ Bước 1)
- Trust score

**Xử Lý**:
1. **Chọn strategies** → Dựa vào current_state + is_strong
2. **Tính probability** → Base prob × trust_score × strength
3. **Xác định risk** → Risk multiplier theo từng strategy
4. **Tính edge score** → Strength × strategy weight
5. **Sắp xếp** → Theo prob × edge (cao → thấp)

**Output**:
```python
List[StrategyProbability] {
    strategy: TREND_FOLLOW/BREAKOUT/MEAN_REVERT/SCALP
    probability: 0.0-1.0
    expected_winrate: 0.0-1.0
    risk_multiplier: 0.0-2.0    # Điều chỉnh risk (không vượt max)
    sl_multiplier: ATR multiplier
    tp_multiplier: ATR multiplier
    edge_score: 0.0-1.0
}
```

**Chiến Lược Theo State**:

| State | Strong? | Strategies |
|-------|---------|-----------|
| **LONG** | Yes | 1. Trend Follow (P=0.70, WR=0.65)<br>2. Breakout (P=0.50, WR=0.55) |
| **LONG** | No | 1. Breakout (P=0.40, WR=0.55) |
| **SHORT** | Yes | 1. Trend Follow (P=0.70, WR=0.65)<br>2. Breakout (P=0.50, WR=0.55) |
| **SHORT** | No | 1. Breakout (P=0.40, WR=0.55) |
| **SIDEWAY** | - | 1. Mean Revert (P=0.60, WR=0.60)<br>2. Scalp (P=0.55, WR=0.58) |

---

## 🔄 Flow Chi Tiết

```
┌─────────────────────────────────────────────────────────┐
│ INPUT: df, ema, atr, bands, rsi, volume                 │
│        signal_long=?, signal_short=?                    │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│ BƯỚC 0: _step0_classify_and_predict()                   │
│                                                          │
│ 1. _identify_historical_zones()                         │
│    → Close vs EMA → LONG/SHORT/SIDEWAY zones            │
│    → Calculate zone strength & confidence               │
│                                                          │
│ 2. _detect_current_state()                              │
│    → Lookback 20 bars                                   │
│    → % bars above EMA > 65% → LONG                      │
│    → % bars below EMA > 65% → SHORT                     │
│    → Else → SIDEWAY                                     │
│                                                          │
│ 3. _predict_next_state()                                │
│    → Recent zone pattern                                │
│    → Momentum + EMA slope                               │
│    → If momentum > 0.02 AND slope > 0.01 → LONG         │
│                                                          │
│ 4. _calculate_prediction_confidence()                   │
│    → Zone confidence (40%)                              │
│    → State alignment (40%)                              │
│    → Volatility factor (20%)                            │
│                                                          │
│ 5. _analyze_trend_strength()                            │
│    → Price momentum (25%)                               │
│    → EMA slope (20%)                                    │
│    → Distance from EMA (15%)                            │
│    → Directional consistency (20%)                      │
│    → RSI alignment (10%)                                │
│    → Volume strength (10%)                              │
│    → If score >= 0.65 → STRONG                          │
│                                                          │
│ OUTPUT: RegimePrediction                                │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ Signal arrived?
                 ├─── NO ───┐
                 │          │
                 │          ▼
                 │    [ Skip to Step 2, validation=None ]
                 │
                 └─── YES ──┐
                            │
┌───────────────────────────▼─────────────────────────────┐
│ BƯỚC 1: _step1_validate_signal()                        │
│                                                          │
│ 1. Compare:                                             │
│    signal_state (LONG/SHORT) vs regime_state            │
│    matches = (signal_state == regime_state)             │
│                                                          │
│ 2. Trust Components:                                    │
│    a) Confidence check                                  │
│       IF confidence < 0.70 → 0.3, warning               │
│       ELSE → 0.8                                        │
│                                                          │
│    b) Strength check                                    │
│       IF is_strong → 0.9                                │
│       ELSE → 0.5, warning                               │
│                                                          │
│    c) Match check                                       │
│       IF matches → 1.0                                  │
│       ELSE → 0.2, warning                               │
│                                                          │
│    d) Volatility check                                  │
│       IF volatility > 0.05 → 0.4, warning               │
│       ELSE → 0.8                                        │
│                                                          │
│    e) Volume confirmation                               │
│       vol_spike = current / avg_20                      │
│       IF spike > 1.5 → 0.9                              │
│       ELSE → 0.6, warning                               │
│                                                          │
│ 3. Calculate:                                           │
│    trust_score = mean(components)                       │
│    is_valid = (trust >= 0.65 AND matches)               │
│                                                          │
│ OUTPUT: SignalValidation                                │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│ BƯỚC 2: _step2_calculate_probabilities()                │
│                                                          │
│ base_trust = validation.trust_score (nếu có)            │
│              hoặc 0.5 (nếu không có signal)             │
│                                                          │
│ FOR EACH state:                                         │
│                                                          │
│ IF LONG + STRONG:                                       │
│   1. Trend Follow                                       │
│      prob = 0.70 × base_trust                           │
│      winrate = 0.65                                     │
│      risk = 1.0, SL=2×ATR, TP=4×ATR                     │
│      edge = strength × 1.2                              │
│                                                          │
│   2. Breakout                                           │
│      prob = 0.50 × base_trust × strength                │
│      winrate = 0.55                                     │
│      risk = 0.8, SL=3×ATR, TP=5×ATR                     │
│      edge = strength × 0.9                              │
│                                                          │
│ IF SIDEWAY:                                             │
│   1. Mean Revert                                        │
│      prob = 0.60 × base_trust                           │
│      winrate = 0.60                                     │
│      risk = 0.5, SL=1.5×ATR, TP=2×ATR                   │
│      edge = 0.7                                         │
│                                                          │
│   2. Scalp                                              │
│      prob = 0.55 × base_trust                           │
│      winrate = 0.58                                     │
│      risk = 0.3, SL=1×ATR, TP=1.5×ATR                   │
│      edge = 0.6                                         │
│                                                          │
│ OUTPUT: List[StrategyProbability]                       │
│         Sorted by (probability × edge_score) DESC       │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│ FINAL OUTPUT: RegimeOutput                              │
│                                                          │
│ prediction: RegimePrediction                            │
│ validation: SignalValidation (or None)                  │
│ strategies: List[StrategyProbability]                   │
│                                                          │
│ allow_long:    current_state==LONG AND valid            │
│ allow_short:   current_state==SHORT AND valid           │
│ allow_sideway: current_state==SIDEWAY                   │
│                                                          │
│ recommended_strategy: max(prob × edge)                  │
│                                                          │
│ metadata: {lookback, thresholds, ...}                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🧮 Công Thức Toán Học Chính

### **Zone Strength**
```
distance = |close - ema|
normalized_distance = distance / atr
zone_strength = mean(normalized_distance)
```

### **Zone Confidence**
```
touches = count(low <= ema <= high)
consistency = 1.0 - (touches / total_bars)
zone_confidence = consistency
```

### **Trend Strength Score**
```
price_momentum = (close[-1] - close[0]) / close[0]
ema_slope = (ema[-1] - ema[0]) / ema[0]
distance_from_ema = mean(|close - ema| / atr)
consistency = directional_moves / total_moves
rsi_alignment = (mean(rsi) - 50) / 50      # for LONG
vol_strength = current_vol / avg_vol_20

strength_score = 
    price_momentum × 0.25 +
    ema_slope × 0.20 +
    distance_from_ema × 0.15 +
    consistency × 0.20 +
    rsi_alignment × 0.10 +
    vol_strength × 0.10

IF strength_score >= 0.65: STRONG
ELIF strength_score >= 0.40: WEAK
ELSE: NEUTRAL
```

### **Trust Score**
```
components = [
    confidence_score,    # 0.3 or 0.8
    strength_score,      # 0.9 or 0.5
    match_score,         # 1.0 or 0.2
    volatility_score,    # 0.4 or 0.8
    volume_score         # 0.9 or 0.6
]

trust_score = mean(components)
is_valid = (trust >= 0.65) AND matches_regime
```

### **Strategy Probability**
```
base_prob = {
    Trend Follow: 0.70,
    Breakout: 0.50,
    Mean Revert: 0.60,
    Scalp: 0.55
}

probability = base_prob × trust_score × strength_modifier
edge_score = strength_score × strategy_weight
```

---

## 🎛️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_short` | 20 | Cửa sổ ngắn (current state, strength) |
| `lookback_long` | 100 | Cửa sổ dài (zones, prediction) |
| `strength_threshold` | 0.65 | Ngưỡng STRONG (>=0.65) |
| `confidence_threshold` | 0.70 | Ngưỡng tin cậy tối thiểu |
| `multi_tf_levels` | 3 | Số mức timeframe (future) |

---

## 📤 Output Structure

```python
{
    'current_state': 'LONG',
    'next_state': 'LONG',
    'confidence': 0.8234,
    'trend_strength': 'strong',
    'is_strong': True,
    'strength_score': 0.7245,
    
    'zones': [
        {'start': 0, 'end': 50, 'state': 'SIDEWAY', 'strength': 0.3, 'confidence': 0.6},
        {'start': 51, 'end': 120, 'state': 'LONG', 'strength': 0.8, 'confidence': 0.9},
        ...
    ],
    
    'validation': {  # Nếu có signal
        'is_valid': True,
        'matches_regime': True,
        'trust_score': 0.8567,
        'regime_says': 'LONG',
        'signal_says': 'LONG',
        'warnings': []
    },
    
    'strategies': [
        {
            'strategy': 'trend_follow',
            'probability': 0.6234,
            'expected_winrate': 0.65,
            'risk_multiplier': 1.0,
            'sl_multiplier': 2.0,
            'tp_multiplier': 4.0,
            'edge_score': 0.8694
        },
        {
            'strategy': 'breakout',
            'probability': 0.4123,
            'expected_winrate': 0.55,
            'risk_multiplier': 0.8,
            'sl_multiplier': 3.0,
            'tp_multiplier': 5.0,
            'edge_score': 0.6521
        }
    ],
    
    'recommended': {
        'strategy': 'trend_follow',
        'probability': 0.6234,
        'edge_score': 0.8694
    },
    
    'allow_long': True,
    'allow_short': False,
    'allow_sideway': False
}
```

---

## 🚀 Usage

```python
from regime_v3 import analyze_regime_v3, format_regime_output

# Bước 0: Luôn phân loại & dự đoán
output = analyze_regime_v3(
    df=df,
    ema=ema,
    atr=atr,
    upper_band=upper_band,
    lower_band=lower_band,
    rsi=rsi,           # optional
    volume=volume      # optional
)

# Hiển thị chart background
zones = output.prediction.zones
for zone in zones:
    color = 'green' if zone.state == MarketState.LONG else 'red' if zone.state == MarketState.SHORT else 'gray'
    chart.draw_zone(zone.start_idx, zone.end_idx, color)

# Bước 1 + 2: Khi signal đến
output_with_signal = analyze_regime_v3(
    df=df,
    ema=ema,
    atr=atr,
    upper_band=upper_band,
    lower_band=lower_band,
    rsi=rsi,
    volume=volume,
    signal_long=True   # Signal báo LONG
)

# Kiểm tra validation
if output_with_signal.validation.is_valid:
    # Lấy chiến lược recommend
    rec = output_with_signal.recommended_strategy
    print(f"Strategy: {rec.strategy.value}")
    print(f"Probability: {rec.probability:.2%}")
    print(f"WinRate: {rec.expected_winrate:.2%}")
    print(f"Risk: {rec.risk_multiplier}x")
    
    # Execute quyết định dựa vào list strategies
    for strategy in output_with_signal.strategies:
        if strategy.probability > 0.6:
            execute_strategy(strategy)
else:
    print("Signal rejected:", output_with_signal.validation.warnings)

# Format cho API
formatted = format_regime_output(output)
```

---

## ✅ Ưu Điểm

1. **Single-file**: Toàn bộ logic trong 1 file, dễ maintain
2. **Clear flow**: 3 bước tuyến tính, không rẽ nhánh phức tạp
3. **Vectorized**: Dùng numpy array, tính toán nhanh
4. **Flexible**: Parameters điều chỉnh được
5. **Transparent**: Output chi tiết, dễ debug
6. **No comments**: Code tự giải thích, documentation riêng

---

## 🔄 So Sánh V2 vs V3

| Aspect | V2 (4 Engines) | V3 (Single File) |
|--------|----------------|------------------|
| **Files** | 5 files (regime_v2.py + 4 engines) | 1 file (regime_v3.py) |
| **Flow** | Parallel → Aggregate | Linear: Step 0 → 1 → 2 |
| **Hypothesis** | H1/H2/H3 scores | State-based strategies |
| **Signal** | Always analyze | Only when signal arrives |
| **Output** | Blueprint (Regime + Model) | Prediction + Validation + Strategies |
| **Philosophy** | "Assessor" | "Oracle + Validator + Calculator" |
| **Complexity** | Higher (4 engines) | Lower (3 steps) |
| **Maintenance** | 5 files to track | 1 file |

---

## 📝 Notes

- **Bước 0** chạy mọi lúc → Chart background luôn được tô màu
- **Bước 1** chỉ chạy khi có signal → Xác thực trước khi vào lệnh
- **Bước 2** cung cấp xác suất → Execute tự quyết định chiến lược
- **Risk multiplier** điều chỉnh linh động, không vượt max từ Position Sizing
- **Thresholds** có thể optimize theo từng asset/timeframe

---

**Version**: 3.0.0  
**Date**: 2026-03-02  
**Philosophy**: "Know the future, validate the present, calculate the edge"
