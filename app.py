import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# 1. KONFIGURASI HALAMAN WEB
st.set_page_config(page_title="Forex Master Auto-Analyzer", layout="wide")
st.title("📈 Forex Master Technical & Pattern Analyzer")

# 2. SIDEBAR / INPUT USER
st.sidebar.header("⚙️ Pengaturan Analisis")
pair = st.sidebar.selectbox("Pilih Currency Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "GBPJPY=X"])
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "15m"])
period = st.sidebar.selectbox("Periode Data", ["1mo", "3mo", "6mo", "1y"])

st.sidebar.header("🛡️ Manajemen Risiko")
account_balance = st.sidebar.number_input("Saldo Akun ($)", value=1000, step=100)
risk_percent = st.sidebar.slider("Risiko per Trade (%)", 0.5, 5.0, 1.0)

# 3. AMBIL DATA HARGA
@st.cache_data(ttl=300)
def load_data(ticker, p, i):
    data = yf.download(tickers=ticker, period=p, interval=i)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.dropna(inplace=True)
    return data

df = load_data(pair, period, timeframe)

if not df.empty:
    # 4. INDIKATOR TEKNIKAL, VOLATILITAS (ATR) & STOCHASTIC
    # Moving Averages (Trend)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator (14, 3, 3)
    n_stoch = 14
    low_min = df['Low'].rolling(window=n_stoch).min()
    high_max = df['High'].rolling(window=n_stoch).max()
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=3).mean()

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    last_atr = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else 0.0010
    last_close = df['Close'].iloc[-1]

    # 5. ALGORITMA DETEKSI POLA CHART
    order = 5
    maxima_idx = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    minima_idx = argrelextrema(df['Close'].values, np.less, order=order)[0]

    # Double Bottom (Reversal Naik)
    double_bottoms = []
    if len(minima_idx) >= 2:
        for i in range(len(minima_idx) - 1):
            i1, i2 = minima_idx[i], minima_idx[i+1]
            p1, p2 = df['Close'].iloc[i1], df['Close'].iloc[i2]
            if abs(p1 - p2) <= (1.5 * last_atr):
                double_bottoms.append((df.index[i1], df.index[i2]))

    # Head & Shoulders (Reversal Turun)
    head_shoulders = []
    if len(maxima_idx) >= 3:
        for i in range(len(maxima_idx) - 2):
            i_ls, i_h, i_rs = maxima_idx[i], maxima_idx[i+1], maxima_idx[i+2]
            p_ls, p_h, p_rs = df['Close'].iloc[i_ls], df['Close'].iloc[i_h], df['Close'].iloc[i_rs]
            if p_h > p_ls and p_h > p_rs and abs(p_ls - p_rs) <= (1.5 * last_atr):
                head_shoulders.append((df.index[i_ls], df.index[i_h], df.index[i_rs]))

    # Inverse Head & Shoulders (Reversal Naik)
    inv_head_shoulders = []
    if len(minima_idx) >= 3:
        for i in range(len(minima_idx) - 2):
            i_ls, i_h, i_rs = minima_idx[i], minima_idx[i+1], minima_idx[i+2]
            p_ls, p_h, p_rs = df['Close'].iloc[i_ls], df['Close'].iloc[i_h], df['Close'].iloc[i_rs]
            if p_h < p_ls and p_h < p_rs and abs(p_ls - p_rs) <= (1.5 * last_atr):
                inv_head_shoulders.append((df.index[i_ls], df.index[i_h], df.index[i_rs]))

    # 6. VISUALISASI CHART UTAMA (CANDLESTICK)
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candlestick"
    )])
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
    fig.add_trace(go.Scatter(x=df.index[maxima_idx], y=df['Close'].iloc[maxima_idx], mode='markers', marker=dict(color='red', size=6), name='Puncak (Max)'))
    fig.add_trace(go.Scatter(x=df.index[minima_idx], y=df['Close'].iloc[minima_idx], mode='markers', marker=dict(color='green', size=6), name='Lembah (Min)'))

    fig.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 7. VISUALISASI CHART STOCHASTIC OSCILLATOR
    fig_stoch = go.Figure()
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df['%K'], line=dict(color='cyan', width=1.5), name='%K Line'))
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df['%D'], line=dict(color='magenta', width=1.5, dash='dash'), name='%D Line'))
    
    # Garis Level 80 (Overbought) dan 20 (Oversold)
    fig_stoch.add_hline(y=80, line_dash="dot", line_color="red", annotation_text="Overbought (80)")
    fig_stoch.add_hline(y=20, line_dash="dot", line_color="green", annotation_text="Oversold (20)")
    
    fig_stoch.update_layout(height=200, title="Stochastic Oscillator (14, 3, 3)", yaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_stoch, use_container_width=True)

    # 8. KESIMPULAN REKOMENDASI (CONFLUENCE SCORE)
    st.subheader("🎯 Rekomendasi Aksi & Sinyal Terpadu")
    
    last_sma20 = df['SMA_20'].iloc[-1]
    last_sma50 = df['SMA_50'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    last_stoch_k = df['%K'].iloc[-1]
    last_stoch_d = df['%D'].iloc[-1]
    
    score = 0  # Positif = Buy, Negatif = Sell
    
    # Tren (SMA)
    if last_sma20 > last_sma50: score += 1
    else: score -= 1
    
    # RSI
    if last_rsi < 30: score += 1
    elif last_rsi > 70: score -= 1
    
    # Stochastic Oscillator
    if last_stoch_k < 20:
        score += 1
        if last_stoch_k > last_stoch_d: score += 1  # Bullish Crossover
    elif last_stoch_k > 80:
        score -= 1
        if last_stoch_k < last_stoch_d: score -= 1  # Bearish Crossover

    # Pola Chart
    if double_bottoms or inv_head_shoulders: score += 2
    if head_shoulders: score -= 2

    c1, c2, c3 = st.columns([1.5, 1, 1])
    
    with c1:
        if score >= 3:
            st.success(f"### 🟢 Rekomendasi: STRONG BUY\nHarga saat ini: **{last_close:.5f}**")
        elif score >= 1:
            st.success(f"### 🟢 Rekomendasi: BUY\nHarga saat ini: **{last_close:.5f}**")
        elif score <= -3:
            st.error(f"### 🔴 Rekomendasi: STRONG SELL\nHarga saat ini: **{last_close:.5f}**")
        elif score <= -1:
            st.error(f"### 🔴 Rekomendasi: SELL\nHarga saat ini: **{last_close:.5f}**")
        else:
            st.warning(f"### 🟡 Rekomendasi: WAIT (Pasar Sideways/Ragu)\nHarga saat ini: **{last_close:.5f}**")

    # 9. KALKULATOR MANAJEMEN RISIKO (LOT SIZING)
    stop_loss_pips = (last_atr * 1.5)
    risk_amount = account_balance * (risk_percent / 100)
    
    pip_value_per_lot = 10
    pips_at_risk = stop_loss_pips * 10000 if "JPY" not in pair else stop_loss_pips * 100
    recommended_lot = risk_amount / (pips_at_risk * pip_value_per_lot) if pips_at_risk > 0 else 0.01

    with c2:
        st.markdown("**🛡️ Parameter Risiko (ATR):**")
        st.write(f"- Potensi Jarak Stop Loss: **{pips_at_risk:.1f} Pips**")
        st.write(f"- Maksimal Kerugian ({risk_percent}%): **${risk_amount:.2f}**")

    with c3:
        st.markdown("**📊 Ukuran Posisi Disarankan:**")
        st.info(f"### **{recommended_lot:.2f} Lot**")
        st.caption("Gunakan ukuran lot ini agar kerugian tidak melebihi batas toleransi risiko Anda.")

    # 10. DETAIL INDIKATOR INDIVIDUAL
    st.markdown("---")
    st.subheader("📊 Rincian Sinyal Indikator")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.write("**1. Moving Average:**")
        if last_sma20 > last_sma50: st.write("🟢 Uptrend")
        else: st.write("🔴 Downtrend")

    with col_b:
        st.write(f"**2. RSI ({last_rsi:.1f}):**")
        if last_rsi > 70: st.write("⚠️ Overbought")
        elif last_rsi < 30: st.write("🟢 Oversold")
        else: st.write("⚪ Netral")

    with col_c:
        st.write(f"**3. Stochastic (%K: {last_stoch_k:.1f}):**")
        if last_stoch_k > 80: st.write("⚠️ Overbought")
        elif last_stoch_k < 20: st.write("🟢 Oversold")
        else: st.write("⚪ Netral")

    with col_d:
        st.write("**4. Pola Chart:**")
        if double_bottoms: st.write("🟢 Double Bottom")
        elif inv_head_shoulders: st.write("🟢 Inv. H&S")
        elif head_shoulders: st.write("🔴 Head & Shoulders")
        else: st.write("⚪ Tidak ada pola")

else:
    st.error("Gagal mengambil data dari Yahoo Finance. Pastikan simbol pasangan mata uang benar dan koneksi internet aktif.")
