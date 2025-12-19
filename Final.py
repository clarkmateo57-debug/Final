import streamlit as st
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

# =========================
# CSS (merged)
# =========================
st.markdown("""
<style>

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #fff;
    min-height: 100vh;
}

.container {
    padding: 20px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    max-width: 1600px;
    margin: 0 auto;
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.header h1 {
    font-size: 2.5em;
    margin-bottom: 10px;
    color: #60a5fa;
}

.status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(30, 41, 59, 0.8);
    padding: 15px 25px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid rgba(96, 165, 250, 0.2);
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
}

.live-dot {
    width: 12px;
    height: 12px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.card {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(96, 165, 250, 0.2);
}

.card h3 {
    font-size: 1.2em;
    margin-bottom: 15px;
    color: #60a5fa;
}

.market-item {
    background: rgba(15, 23, 42, 0.6);
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 4px solid #60a5fa;
}

.market-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.market-name {
    font-weight: bold;
    font-size: 1.1em;
}

.market-price {
    font-size: 1.3em;
    font-weight: bold;
}

.price-up {
    color: #10b981;
}

.price-down {
    color: #ef4444;
}

.market-details {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 10px;
    font-size: 0.9em;
}

.detail-item {
    display: flex;
    justify-content: space-between;
}

.label {
    color: #94a3b8;
}

.order-feed {
    height: 400px;
    overflow-y: auto;
    background: rgba(15, 23, 42, 0.6);
    padding: 15px;
    border-radius: 8px;
}

.order-item {
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 6px;
    border-left: 3px solid;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(-10px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.order-long {
    background: rgba(16, 185, 129, 0.1);
    border-color: #10b981;
}

.order-short {
    background: rgba(239, 68, 68, 0.1);
    border-color: #ef4444;
}

.order-close {
    background: rgba(96, 165, 250, 0.1);
    border-color: #60a5fa;
}

.order-header {
    display: flex;
    justify-content: space-between;
    font-weight: bold;
    margin-bottom: 5px;
}

.order-details {
    font-size: 0.85em;
    color: #94a3b8;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}

.stat-box {
    background: rgba(15, 23, 42, 0.6);
    padding: 15px;
    border-radius: 8px;
    text-align: center;
}

.stat-label {
    color: #94a3b8;
    font-size: 0.9em;
    margin-bottom: 5px;
}

.stat-value {
    font-size: 1.8em;
    font-weight: bold;
}

.prediction-box {
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}

.prediction-label {
    color: #a78bfa;
    font-size: 0.9em;
    margin-bottom: 5px;
}

.prediction-value {
    font-size: 1.3em;
    font-weight: bold;
    color: #c4b5fd;
}

canvas {
    max-height: 300px;
}

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.4);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(96, 165, 250, 0.3);
    border-radius: 4px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Engine data structures
# =========================

@dataclass
class Market:
    name: str
    symbol: str
    price: float
    volatility: float
    trend: float
    prediction: Optional[float] = None
    momentum: Optional[float] = None

@dataclass
class Position:
    contracts: int = 0
    side: Optional[str] = None  # "LONG", "SHORT", or None
    entry: float = 0.0

@dataclass
class Order:
    timestamp: str
    market: str
    symbol: str
    side: str
    contracts: int
    price: float
    reason: str

@dataclass
class Trade:
    symbol: str
    side: str
    entry: float
    exit: float
    contracts: int
    pnl: float
    timestamp_open: str
    timestamp_close: str

INITIAL_MARKETS: List[Market] = [
    Market("Crude Oil (CL)", "CL", 72.50, 0.025, 0.001),
    Market("Gold (GC)", "GC", 2050.00, 0.015, 0.0005),
    Market("S&P 500 (ES)", "ES", 4780.00, 0.018, 0.0008),
    Market("EUR/USD (EC)", "EC", 1.0950, 0.012, -0.0003),
    Market("Treasury (ZN)", "ZN", 110.50, 0.01, -0.0002),
    Market("Natural Gas (NG)", "NG", 2.85, 0.035, 0.002),
]

# =========================
# Engine initialization
# =========================

def init_engine_state(state):
    if "markets" not in state:
        state.markets = [Market(**m.__dict__) for m in INITIAL_MARKETS]

    if "price_history" not in state:
        state.price_history: Dict[str, List[float]] = {}
        for m in state.markets:
            prices = []
            for _ in range(100):
                shock = (random.random() - 0.5) * m.volatility * 0.5
                prices.append(m.price * (1 + shock))
            state.price_history[m.symbol] = prices

    if "positions" not in state:
        state.positions: Dict[str, Position] = {m.symbol: Position() for m in state.markets}

    if "orders" not in state:
        state.orders: List[Order] = []

    if "equity" not in state:
        state.equity: List[float] = [100_000.0]

    if "total_orders" not in state:
        state.total_orders: int = 0

    if "trades" not in state:
        state.trades: List[Trade] = []

# =========================
# Indicators
# =========================

def calculate_ma(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    slice_ = prices[-period:]
    return sum(slice_) / len(slice_)

def calculate_atr(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    ranges = []
    for i in range(len(prices) - period, len(prices)):
        ranges.append(abs(prices[i] - prices[i - 1]))
    return sum(ranges) / len(ranges)

def predict_next_price(prices: List[float]) -> float:
    n = len(prices)
    if n < 3:
        return prices[-1]
    velocity = prices[n - 1] - prices[n - 2]
    acceleration = prices[n - 1] - 2 * prices[n - 2] + prices[n - 3]
    return prices[n - 1] + velocity + 0.5 * acceleration

def calculate_momentum(prices: List[float]) -> float:
    n = len(prices)
    if n < 10:
        return 0.0
    return (prices[n - 1] - prices[n - 10]) / prices[n - 10]

# =========================
# Orders, trades, equity
# =========================

def execute_order(state, market: Market, side: str, contracts: int, reason: str):
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")

    order = Order(
        timestamp=timestamp,
        market=market.name,
        symbol=market.symbol,
        side=side,
        contracts=contracts,
        price=market.price,
        reason=reason
    )

    state.orders.insert(0, order)
    state.orders = state.orders[:50]
    state.total_orders += 1

    pos = state.positions[market.symbol]

    # trade logging on CLOSE
    if side == "CLOSE" and pos.side is not None and pos.contracts > 0:
        direction = 1 if pos.side == "LONG" else -1
        pnl = direction * (market.price - pos.entry) * pos.contracts * 100.0
        trade = Trade(
            symbol=market.symbol,
            side=pos.side,
            entry=pos.entry,
            exit=market.price,
            contracts=pos.contracts,
            pnl=pnl,
            timestamp_open=pos.entry_time if hasattr(pos, "entry_time") else "",
            timestamp_close=timestamp
        )
        state.trades.append(trade)
        state.positions[market.symbol] = Position()
    elif side in ("LONG", "SHORT"):
        # opening / flipping position
        new_pos = Position(
            contracts=contracts,
            side=side,
            entry=market.price,
        )
        new_pos.entry_time = timestamp
        state.positions[market.symbol] = new_pos

def update_equity_from_positions(state):
    last_equity = state.equity[-1]
    pnl_open = 0.0
    for m in state.markets:
        pos = state.positions[m.symbol]
        if pos.contracts == 0 or pos.side is None:
            continue
        direction = 1 if pos.side == "LONG" else -1
        pnl_open += direction * (m.price - pos.entry) * pos.contracts * 100.0
    state.equity.append(last_equity + pnl_open)

# =========================
# One simulation step
# =========================

def simulate_step(state):
    # price evolution (synthetic)
    for i, m in enumerate(state.markets):
        prices = state.price_history[m.symbol]
        last_price = prices[-1]
        random_shock = (random.random() - 0.5) * m.volatility
        noise = random.uniform(-1, 1) * m.volatility * 0.3
        new_price = last_price * (1 + random_shock + m.trend + noise)
        state.markets[i].price = new_price
        state.price_history[m.symbol].append(new_price)

    # strategy per market
    for m in state.markets:
        prices = state.price_history[m.symbol]
        fast = calculate_ma(prices, st.session_state.fast_ma)
        slow = calculate_ma(prices, st.session_state.slow_ma)
        atr = calculate_atr(prices, st.session_state.atr_period)
        prediction = predict_next_price(prices)
        momentum = calculate_momentum(prices)

        if fast is None or slow is None or atr is None:
            continue

        m.prediction = prediction
        m.momentum = momentum

        pos = state.positions[m.symbol]
        equity_now = state.equity[-1]
        risk_amount = equity_now * (st.session_state.risk_per_trade / 100.0)
        contracts = max(1, int(risk_amount / max(atr * 100.0, 1e-6)))

        current_price = prices[-1]
        trend_signal = 1 if fast > slow else -1
        prediction_signal = 1 if prediction > current_price else -1
        if momentum > 0.001:
            momentum_signal = 1
        elif momentum < -0.001:
            momentum_signal = -1
        else:
            momentum_signal = 0

        combined = trend_signal + prediction_signal + momentum_signal
        side = None
        reason = None

        if pos.contracts == 0:
            if combined >= 2:
                side = "LONG"
                reason = f"MA+Prediction({prediction:.2f})+Momentum"
            elif combined <= -2:
                side = "SHORT"
                reason = f"MA+Prediction({prediction:.2f})+Momentum"
        else:
            if pos.side == "LONG" and combined <= -1:
                side = "CLOSE"
                reason = "Exit Long - Signal Reversal"
            elif pos.side == "SHORT" and combined >= 1:
                side = "CLOSE"
                reason = "Exit Short - Signal Reversal"

        if side is not None:
            execute_order(state, m, side, contracts if side != "CLOSE" else pos.contracts, reason)

    update_equity_from_positions(state)

# =========================
# Performance analytics
# =========================

def compute_performance(state):
    trades: List[Trade] = state.trades
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "max_dd": 0.0
        }

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100.0
    avg_pnl = sum(pnls) / total_trades
    total_pnl = sum(pnls)

    # max drawdown from equity curve
    equity = state.equity
    max_equity = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > max_equity:
            max_equity = v
        dd = (v - max_equity) / max_equity
        if dd < max_dd:
            max_dd = dd

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "max_dd": max_dd * 100.0
    }

# =========================
# Sidebar controls
# =========================

st.sidebar.header("⚙️ Engine & Strategy")

# strategy params
st.session_state.fast_ma = st.sidebar.slider("Fast MA", 5, 50, 10, 1)
st.session_state.slow_ma = st.sidebar.slider("Slow MA", 10, 200, 30, 5)
st.session_state.atr_period = st.sidebar.slider("ATR Period", 5, 50, 14, 1)
st.session_state.risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)

# auto-run
auto = st.sidebar.checkbox("Auto‑Run Engine", value=False)
st.session_state.auto_run = auto

delay = st.sidebar.slider(
    "Auto‑Run Speed (seconds per step)",
    min_value=0.05,
    max_value=1.0,
    value=0.5,
    step=0.05
)
st.session_state.auto_run_delay = delay

batch_steps = st.sidebar.number_input(
    "Batch Steps",
    min_value=1,
    max_value=1000,
    value=100,
    step=50
)

if st.sidebar.button("Run Batch"):
    init_engine_state(st.session_state)
    for _ in range(batch_steps):
        simulate_step(st.session_state)
    st.experimental_rerun()

if st.sidebar.button("Reset Engine"):
    for key in ["markets", "price_history", "positions", "orders", "equity", "total_orders", "trades"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

# =========================
# Init engine
# =========================

init_engine_state(st.session_state)

# =========================
# Auto-run loop
# =========================

def autorun_engine():
    if st.session_state.get("auto_run", False):
        simulate_step(st.session_state)
        time.sleep(st.session_state.get("auto_run_delay", 0.5))
        st.experimental_rerun()

# =========================
# Main UI
# =========================

st.markdown('<div class="container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>⚡ Live CTA Trading System</h1>
    <p>Real-time futures trading with predictive analytics (simulated feed)</p>
</div>
""", unsafe_allow_html=True)

# Status bar
status_left, status_right = st.columns([3, 1])
with status_left:
    st.markdown("""
    <div class="status-bar">
        <div class="status-indicator">
            <div class="live-dot"></div>
            <span><strong>LIVE SIMULATION</strong> - Engine running in Python</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with status_right:
    st.markdown(
        f"<div id='timestamp'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        unsafe_allow_html=True
    )

# Manual controls
c1, c2 = st.columns(2)
with c1:
    if st.button("Run 1 step"):
        simulate_step(st.session_state)
with c2:
    if st.button("Run 20 steps"):
        for _ in range(20):
            simulate_step(st.session_state)

# Grid 1: markets + orders
left1, right1 = st.columns([2, 1])

with left1:
    st.markdown('<div class="card"><h3>📊 Active Markets</h3>', unsafe_allow_html=True)
    for m in st.session_state.markets:
        prices = st.session_state.price_history[m.symbol]
        last = prices[-1]
        prev = prices[-2] if len(prices) > 1 else last
        change = last - prev
        change_pct = (change / prev * 100.0) if prev != 0 else 0.0
        direction_class = "price-up" if change >= 0 else "price-down"

        pos = st.session_state.positions[m.symbol]
        pos_color = "#10b981" if pos.side == "LONG" else "#ef4444" if pos.side == "SHORT" else "#94a3b8"
        pos_text = f"{pos.side} {pos.contracts}" if pos.contracts > 0 else "FLAT"

        st.markdown(f"""
        <div class="market-item">
            <div class="market-header">
                <div class="market-name">{m.name}</div>
                <div class="market-price {direction_class}">
                    ${last:.2f} {'▲' if change >= 0 else '▼'} {abs(change_pct):.2f}%
                </div>
            </div>
            <div class="market-details">
                <div class="detail-item">
                    <span class="label">Fast MA:</span><span>{calculate_ma(prices, st.session_state.fast_ma) or '-':}</span>
                </div>
                <div class="detail-item">
                    <span class="label">Slow MA:</span><span>{calculate_ma(prices, st.session_state.slow_ma) or '-':}</span>
                </div>
                <div class="detail-item">
                    <span class="label">Position:</span><span style="color:{pos_color}">{pos_text}</span>
                </div>
                <div class="detail-item">
                    <span class="label">Momentum:</span><span>{(m.momentum * 100):.2f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if m.prediction is not None:
            pred_change = (m.prediction - last) / last * 100.0 if last != 0 else 0.0
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">🔮 Predicted Next Price (Calculus)</div>
                <div class="prediction-value">
                    ${m.prediction:.2f} {'↗' if m.prediction > last else '↘'} {pred_change:.2f}%
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right1:
    st.markdown('<div class="card"><h3>📈 Order Flow</h3><div class="order-feed">', unsafe_allow_html=True)
    for order in st.session_state.orders:
        if order.side == "LONG":
            cls = "order-item order-long"
            icon = "📈"
        elif order.side == "SHORT":
            cls = "order-item order-short"
            icon = "📉"
        else:
            cls = "order-item order-close"
            icon = "✖"
        st.markdown(f"""
        <div class="{cls}">
            <div class="order-header">
                <span>{icon} {order.side}</span>
                <span>{order.contracts} contracts</span>
            </div>
            <div class="order-details">
                {order.market} @ ${order.price:.2f}<br>
                {order.reason}<br>
                <span style="color:#64748b">{order.timestamp}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# Grid 2: stats + equity + performance
left2, right2 = st.columns([2, 1])

with left2:
    st.markdown('<div class="card"><h3>💼 Portfolio Statistics</h3>', unsafe_allow_html=True)

    positions: Dict[str, Position] = st.session_state.positions
    total_orders = st.session_state.total_orders
    long_positions = sum(1 for p in positions.values() if p.side == "LONG")
    short_positions = sum(1 for p in positions.values() if p.side == "SHORT")
    total_contracts = sum(p.contracts for p in positions.values())

    perf = compute_performance(st.session_state)

    st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">Total Orders</div>
        <div class="stat-value">{total_orders}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Open Longs</div>
        <div class="stat-value price-up">{long_positions}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Open Shorts</div>
        <div class="stat-value price-down">{short_positions}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Total Contracts</div>
        <div class="stat-value">{total_contracts}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Trades</div>
        <div class="stat-value">{perf['total_trades']}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Win Rate</div>
        <div class="stat-value">{perf['win_rate']:.1f}%</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Avg PnL / Trade</div>
        <div class="stat-value">{perf['avg_pnl']:.0f}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Max Drawdown</div>
        <div class="stat-value">{perf['max_dd']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

with right2:
    st.markdown('<div class="card"><h3>📉 Portfolio Equity</h3>', unsafe_allow_html=True)
    st.line_chart(st.session_state.equity)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Auto-run at the very end
autorun_engine()
