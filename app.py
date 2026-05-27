"""
FinSight — Phase 3: Interactive Business Analytics Dashboard
============================================================
Run:  streamlit run app.py

Pages
─────
  1. Overview        — Portfolio KPIs + allocation pie
  2. Market Data     — Price + volume explorer per ticker
  3. LSTM Insights   — Training history + volatility heatmap
  4. GA Optimisation — Evolution fitness + Sharpe frontier
  5. Sentiment       — Daily sentiment timeline per ticker
"""
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from config import DATA_DIR, TICKERS

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark financial terminal theme */
  [data-testid="stAppViewContainer"] { background: #0a0e1a; }
  [data-testid="stSidebar"]           { background: #0d1222; border-right: 1px solid #1e2d4a; }
  .main .block-container               { padding: 1.5rem 2rem; }

  h1, h2, h3 { color: #e8f4fd; font-family: 'Courier New', monospace; }
  p, li       { color: #b0c4d8; }

  /* KPI Cards */
  .kpi-card {
      background: linear-gradient(135deg, #0d1929 0%, #1a2744 100%);
      border: 1px solid #1e3a5f;
      border-radius: 8px;
      padding: 16px 20px;
      text-align: center;
  }
  .kpi-label  { color: #5a8fc2; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  .kpi-value  { color: #00d4aa; font-size: 28px; font-weight: 700; font-family: 'Courier New'; }
  .kpi-delta  { color: #7fb3d3; font-size: 12px; margin-top: 2px; }

  /* Metric delta override */
  [data-testid="metric-container"] { background: #0d1929; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#060d1a",
    font=dict(color="#b0c4d8", family="Courier New"),
    xaxis=dict(gridcolor="#132037", linecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#132037", linecolor="#1e3a5f"),
)

ACCENT = "#00d4aa"
ACCENT2 = "#f9a825"


# ─── Data loaders (cached) ────────────────────────────────────────────────────

@st.cache_data
def load_market_data():
    p = DATA_DIR / "merged_features.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_optimization():
    p = DATA_DIR / "optimization_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

@st.cache_data
def load_training_history():
    p = DATA_DIR / "training_history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

@st.cache_data
def load_forecasts():
    p = DATA_DIR / "volatility_forecasts.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ─── Sidebar navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 FinSight Platform")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📈 Market Data", "🧠 LSTM Insights",
         "🧬 GA Optimisation", "💬 Sentiment"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    selected_tickers = st.multiselect(
        "Filter Tickers", TICKERS, default=TICKERS[:4]
    )
    st.markdown("---")
    st.caption("FinSight MTech Project · 2024")


# ─── Helper: "data not ready" banner ─────────────────────────────────────────

def data_missing(name: str):
    st.warning(
        f"⚠️  **{name}** not found. "
        "Run `python run_all.py` first to generate all data artefacts.",
        icon="🔧"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("📊 FinSight — Portfolio Intelligence Platform")
    st.markdown("_AI-powered risk analytics · Deep learning volatility · Evolutionary optimisation_")
    st.divider()

    opt  = load_optimization()
    fore = load_forecasts()
    df   = load_market_data()

    # ── KPI row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = f"{opt['metrics']['sharpe_ratio']:.2f}" if opt else "—"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Sharpe Ratio</div>'
                    f'<div class="kpi-value">{val}</div>'
                    f'<div class="kpi-delta">GA-optimised portfolio</div></div>',
                    unsafe_allow_html=True)
    with col2:
        val = f"{opt['metrics']['expected_annual_return_pct']:.1f}%" if opt else "—"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Expected Return</div>'
                    f'<div class="kpi-value">{val}</div>'
                    f'<div class="kpi-delta">Annualised (LSTM-driven)</div></div>',
                    unsafe_allow_html=True)
    with col3:
        val = f"{opt['metrics']['portfolio_volatility_pct']:.1f}%" if opt else "—"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Portfolio Volatility</div>'
                    f'<div class="kpi-value">{val}</div>'
                    f'<div class="kpi-delta">Predicted σ</div></div>',
                    unsafe_allow_html=True)
    with col4:
        n = len(TICKERS)
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Assets Tracked</div>'
                    f'<div class="kpi-value">{n}</div>'
                    f'<div class="kpi-delta">S&P 500 mega-caps</div></div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Allocation pie + returns bar ──
    if opt:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Optimal Allocation")
            alloc  = opt["optimal_allocation"]
            labels = list(alloc.keys())
            values = [v * 100 for v in alloc.values()]
            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker=dict(colors=px.colors.qualitative.Safe),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_THEME, height=350, showlegend=False,
                              title_text="Portfolio Weights", title_font_color="#b0c4d8")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Volatility vs. Return")
            if fore is not None and df is not None:
                ret = (
                    df.groupby("ticker")
                    .apply(lambda g: g["daily_return"].mean() * 252 * 100)
                    .reset_index(name="ann_return_pct")
                )
                merged = fore.merge(ret, on="ticker")
                alloc_w = pd.DataFrame([
                    {"ticker": k, "weight": v * 100}
                    for k, v in alloc.items()
                ])
                merged = merged.merge(alloc_w, on="ticker", how="left")

                fig2 = px.scatter(
                    merged, x="predicted_volatility", y="ann_return_pct",
                    size="weight", text="ticker", color="ann_return_pct",
                    color_continuous_scale="Teal",
                    labels={"predicted_volatility": "Predicted Volatility",
                            "ann_return_pct": "Ann. Return (%)"},
                    height=350,
                )
                fig2.update_traces(textposition="top center",
                                   marker=dict(sizemin=8, line=dict(width=1, color="#00d4aa")))
                fig2.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                                   title_text="Risk vs. Return Scatter",
                                   title_font_color="#b0c4d8")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        data_missing("optimization_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Market Data
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Market Data":
    st.title("📈 Market Data Explorer")
    df = load_market_data()
    if df is None:
        data_missing("merged_features.parquet")
    else:
        ticker = st.selectbox("Select Ticker", options=sorted(df["ticker"].unique()))
        sub = df[df["ticker"] == ticker].sort_values("date")

        # Candlestick
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=sub["date"], open=sub["open"], high=sub["high"],
            low=sub["low"],  close=sub["close"],
            increasing_line_color=ACCENT, decreasing_line_color="#e53935",
            name="OHLC",
        ))
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["ma_7d"],  name="7d MA",
            line=dict(color=ACCENT2, width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["ma_30d"], name="30d MA",
            line=dict(color="#ab47bc", width=1.5, dash="dash"),
        ))
        fig.update_layout(**PLOTLY_THEME, height=420,
                          title=f"{ticker} — Price History with Moving Averages",
                          title_font_color="#b0c4d8",
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Volume + Volatility side-by-side
        c1, c2 = st.columns(2)
        with c1:
            fig_vol = px.bar(sub, x="date", y="volume",
                             color_discrete_sequence=[ACCENT],
                             title="Trading Volume", height=280)
            fig_vol.update_layout(**PLOTLY_THEME, title_font_color="#b0c4d8")
            st.plotly_chart(fig_vol, use_container_width=True)

        with c2:
            fig_risk = px.area(sub, x="date", y="rolling_volatility_20d",
                               color_discrete_sequence=[ACCENT2],
                               title="20-day Rolling Volatility", height=280)
            fig_risk.update_layout(**PLOTLY_THEME, title_font_color="#b0c4d8")
            st.plotly_chart(fig_risk, use_container_width=True)

        # Stats table
        st.subheader("Summary Statistics")
        stats = {
            "Min Close":      f"${sub['close'].min():.2f}",
            "Max Close":      f"${sub['close'].max():.2f}",
            "Ann. Return":    f"{sub['daily_return'].mean()*252*100:.1f}%",
            "Avg Volatility": f"{sub['rolling_volatility_20d'].mean()*100:.1f}%",
            "Avg Volume":     f"{sub['volume'].mean()/1e6:.1f}M",
        }
        st.dataframe(
            pd.DataFrame(stats.items(), columns=["Metric", "Value"]),
            hide_index=True, use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LSTM Insights
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 LSTM Insights":
    st.title("🧠 LSTM Volatility Forecaster")
    st.markdown("_2-layer LSTM + Temporal Attention trained on price + sentiment sequences_")

    hist  = load_training_history()
    fore  = load_forecasts()

    if hist:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training vs. Validation Loss")
            fig = go.Figure()
            epochs = list(range(1, len(hist["train_loss"]) + 1))
            fig.add_trace(go.Scatter(x=epochs, y=hist["train_loss"],
                                     name="Train Loss", line=dict(color=ACCENT, width=2.5)))
            fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"],
                                     name="Val Loss",   line=dict(color=ACCENT2, width=2.5, dash="dash")))
            fig.update_layout(**PLOTLY_THEME, height=320,
                              xaxis_title="Epoch", yaxis_title="MSE Loss",
                              title_font_color="#b0c4d8")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if fore is not None:
                st.subheader("Predicted Volatility per Asset")
                fig2 = px.bar(fore.sort_values("predicted_volatility"),
                              x="predicted_volatility", y="ticker",
                              orientation="h",
                              color="predicted_volatility",
                              color_continuous_scale="Teal",
                              height=320,
                              labels={"predicted_volatility": "Predicted σ"})
                fig2.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                                   title_font_color="#b0c4d8")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        data_missing("training_history.json")

    # Volatility heatmap
    df = load_market_data()
    if df is not None:
        st.subheader("Monthly Volatility Heatmap")
        df2 = df[df["ticker"].isin(selected_tickers)].copy()
        df2["month"] = df2["date"].dt.to_period("M").astype(str)
        heat = (
            df2.groupby(["month", "ticker"])["rolling_volatility_20d"]
            .mean().reset_index()
            .pivot(index="ticker", columns="month", values="rolling_volatility_20d")
        )
        fig3 = px.imshow(
            heat, color_continuous_scale="RdYlGn_r",
            aspect="auto", height=300,
            labels=dict(color="Volatility"),
        )
        fig3.update_layout(**PLOTLY_THEME, title_font_color="#b0c4d8",
                           coloraxis_colorbar=dict(tickfont=dict(color="#b0c4d8")))
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GA Optimisation
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧬 GA Optimisation":
    st.title("🧬 Evolutionary Portfolio Optimisation")
    st.latex(r"\text{Fitness} = \frac{R_p - R_f}{\sigma_p} \quad \text{(Sharpe Ratio)}")

    opt = load_optimization()
    if opt is None:
        data_missing("optimization_results.json")
    else:
        m = opt["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe Ratio",        f"{m['sharpe_ratio']:.3f}")
        c2.metric("Expected Return",     f"{m['expected_annual_return_pct']:.1f}%")
        c3.metric("Portfolio Volatility",f"{m['portfolio_volatility_pct']:.1f}%")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Evolution Curve — Max Sharpe per Generation")
            evo = pd.DataFrame(opt["evolution_log"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=evo["generation"], y=evo["max_sharpe"],
                name="Max Sharpe", line=dict(color=ACCENT, width=2.5),
                fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=evo["generation"], y=evo["mean_sharpe"],
                name="Mean Sharpe", line=dict(color=ACCENT2, width=1.5, dash="dot"),
            ))
            fig.update_layout(**PLOTLY_THEME, height=350,
                              xaxis_title="Generation", yaxis_title="Sharpe Ratio",
                              title_font_color="#b0c4d8")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Optimal Weight Distribution")
            alloc = opt["optimal_allocation"]
            fig2 = go.Figure(go.Bar(
                x=list(alloc.keys()),
                y=[v * 100 for v in alloc.values()],
                marker=dict(
                    color=[v * 100 for v in alloc.values()],
                    colorscale="Teal",
                    showscale=False,
                    line=dict(color=ACCENT, width=1),
                ),
            ))
            fig2.update_layout(**PLOTLY_THEME, height=350,
                               yaxis_title="Allocation (%)",
                               title_font_color="#b0c4d8")
            st.plotly_chart(fig2, use_container_width=True)

        # Full allocation table
        st.subheader("Allocation Matrix")
        alloc_df = pd.DataFrame([
            {"Ticker": k, "Weight (%)": round(v*100, 2),
             "Category": "Core (>15%)" if v > 0.15 else ("Satellite (5-15%)" if v > 0.05 else "Trim (<5%)")}
            for k, v in sorted(alloc.items(), key=lambda x: -x[1])
        ])
        st.dataframe(alloc_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Sentiment
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💬 Sentiment":
    st.title("💬 Market Sentiment Analysis")
    st.markdown("_Daily sentiment vectors extracted via Gemini API / mock NLP pipeline_")

    df = load_market_data()
    if df is None:
        data_missing("merged_features.parquet")
    else:
        sub = df[df["ticker"].isin(selected_tickers)].sort_values("date")

        # Sentiment timeline
        fig = px.line(
            sub, x="date", y="avg_sentiment", color="ticker",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Daily Average Sentiment Score per Ticker",
            height=380,
            labels={"avg_sentiment": "Sentiment Score"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#5a8fc2", opacity=0.6,
                      annotation_text="Neutral", annotation_font_color="#5a8fc2")
        fig.update_layout(**PLOTLY_THEME, title_font_color="#b0c4d8")
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment vs. next-day return
        st.subheader("Sentiment → Next-Day Return Scatter")
        sub2 = sub.copy()
        sub2["next_day_return"] = sub2.groupby("ticker")["daily_return"].shift(-1)
        sub2 = sub2.dropna(subset=["avg_sentiment", "next_day_return"])

        fig2 = px.scatter(
            sub2, x="avg_sentiment", y="next_day_return", color="ticker",
            color_discrete_sequence=px.colors.qualitative.Safe,
            opacity=0.5, height=340,
            trendline="ols",
            labels={"avg_sentiment": "Sentiment Score",
                    "next_day_return": "Next-Day Return"},
        )
        fig2.update_layout(**PLOTLY_THEME, title_font_color="#b0c4d8")
        st.plotly_chart(fig2, use_container_width=True)

        # Sentiment distribution
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.histogram(
                sub, x="avg_sentiment", color="ticker",
                nbins=40, barmode="overlay", opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Sentiment Distribution", height=300,
            )
            fig3.update_layout(**PLOTLY_THEME, title_font_color="#b0c4d8")
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            monthly = sub.copy()
            monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
            agg = monthly.groupby(["month", "ticker"])["avg_sentiment"].mean().reset_index()
            fig4 = px.line(agg, x="month", y="avg_sentiment", color="ticker",
                           color_discrete_sequence=px.colors.qualitative.Safe,
                           title="Monthly Average Sentiment", height=300)
            fig4.update_layout(**PLOTLY_THEME, title_font_color="#b0c4d8")
            st.plotly_chart(fig4, use_container_width=True)
