from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "portfolio_returns.csv"
OPTIMIZATION_PATH = PROJECT_ROOT / "data" / "processed" / "optimization_results.json"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "risk_metrics.json"
STRESS_PATH = PROJECT_ROOT / "data" / "processed" / "stress_results.json"

st.set_page_config(page_title="Portfolio Risk Lab", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
    [data-testid="stMetric"] { background: #f7f9fc; border: 1px solid #e4e9f0; padding: 1rem; border-radius: 10px; }
    [data-testid="stMetricLabel"] { color: #52606d; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Portfolio Risk Lab")
st.caption("Un poste de pilotage pour comprendre le rendement, le risque et les arbitrages de votre allocation.")

@st.cache_data
def load_artifacts():
    if not DATA_PATH.exists() or not OPTIMIZATION_PATH.exists() or not METRICS_PATH.exists():
        return None
    returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    with OPTIMIZATION_PATH.open(encoding="utf-8") as file:
        optimization = json.load(file)
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)
    stress = {}
    if STRESS_PATH.exists():
        with STRESS_PATH.open(encoding="utf-8") as file:
            stress = json.load(file)
    return returns, optimization, metrics, stress

artifacts = load_artifacts()
if artifacts is None:
    st.error("Les artefacts sont absents. Executez toutes les cellules du notebook avant de lancer Streamlit.")
    st.stop()

returns, optimization, metrics, stress = artifacts
assets = list(returns.columns)
with st.sidebar:
    st.header("Configuration")
    selected = st.multiselect("Actifs", assets, default=assets)
    if not selected:
        st.warning("Selectionnez au moins un actif.")
        st.stop()
    equal_weight = np.repeat(1 / len(selected), len(selected))
    st.caption("Les poids sont normalises automatiquement.")
    weights_input = []
    for asset in selected:
        weights_input.append(
            st.number_input(
                f"Poids {asset}",
                min_value=0.0,
                max_value=1.0,
                value=float(1 / len(selected)),
                step=0.01,
                format="%.2f",
            )
        )
    weights = np.array(weights_input, dtype=float)
    raw_weight_total = float(weights.sum())
    if weights.sum() == 0:
        weights = equal_weight
    weights = weights / weights.sum()
    confidence = st.select_slider("Confiance VaR", options=[0.95, 0.99], value=0.95, format_func=lambda value: f"{value:.0%}")
    horizon = st.select_slider("Horizon de risque", options=[1, 10, 21], value=1, format_func=lambda value: f"{value} jour" if value == 1 else f"{value} jours")

if abs(raw_weight_total - 1) > 0.01:
    st.info(f"Les poids saisis totalisent {raw_weight_total:.0%} et sont ramenes a 100% pour les calculs.")

portfolio_returns = returns[selected].to_numpy() @ weights
if horizon > 1:
    horizon_returns = (1 + pd.Series(portfolio_returns, index=returns.index)).rolling(horizon).apply(np.prod, raw=True) - 1
    risk_returns = horizon_returns.dropna().to_numpy()
else:
    risk_returns = portfolio_returns
losses = -risk_returns
alpha = 1 - confidence
historical_var = float(np.quantile(losses, confidence))
mean_daily = float(risk_returns.mean())
std_daily = float(risk_returns.std(ddof=1))
parametric_var = float(-(mean_daily + std_daily * norm.ppf(alpha)))
cvar = float(losses[losses >= historical_var].mean()) if np.any(losses >= historical_var) else historical_var
annual_return = float((1 + mean_daily) ** 252 - 1)
annual_volatility = float(std_daily * np.sqrt(252))
sharpe = (annual_return - float(metrics.get("risk_free_rate", 0.02))) / annual_volatility if annual_volatility else 0.0
max_drawdown = float(((1 + pd.Series(portfolio_returns)).cumprod() / (1 + pd.Series(portfolio_returns)).cumprod().cummax() - 1).min())

st.subheader("Vue d'ensemble")
metric_cols = st.columns(6)
for column, label, value, format_string in zip(
    metric_cols,
    ["VaR historique", "VaR parametrique", "CVaR", "Rendement annualise", "Volatilite annualisee", "Drawdown max"],
    [historical_var, parametric_var, cvar, annual_return, annual_volatility, max_drawdown],
    ["{:.2%}", "{:.2%}", "{:.2%}", "{:.2%}", "{:.2%}", "{:.2%}"],
):
    column.metric(label, format_string.format(value))
st.caption(f"Risque calcule sur {horizon} jour(s) | Sharpe estime : {sharpe:.2f} | Periode : {returns.index.min():%d/%m/%Y} au {returns.index.max():%d/%m/%Y}")

tab_risk, tab_performance, tab_allocation, tab_stress = st.tabs(["Risque", "Performance", "Allocation", "Stress tests"])

with tab_risk:
    left, right = st.columns([1, 1.5])
    with left:
        st.subheader("Distribution des pertes")
        histogram = go.Figure()
        histogram.add_trace(go.Histogram(x=losses, nbinsx=45, name="Pertes", marker_color="#2563eb", opacity=0.8))
        histogram.add_vline(x=historical_var, line_dash="dash", line_color="#dc2626")
        histogram.add_vline(x=cvar, line_dash="dot", line_color="#d97706")
        histogram.add_annotation(
            x=historical_var,
            y=1.08,
            yref="paper",
            text=f"VaR {confidence:.0%}",
            showarrow=False,
            xanchor="right",
            font={"color": "#dc2626", "size": 11},
        )
        histogram.add_annotation(
            x=cvar,
            y=1.18,
            yref="paper",
            text="CVaR",
            showarrow=False,
            xanchor="left",
            font={"color": "#d97706", "size": 11},
        )
        histogram.update_layout(height=390, xaxis_title="Perte", yaxis_title="Observations", showlegend=False, margin=dict(t=75, b=45))
        st.plotly_chart(histogram, use_container_width=True)
    with right:
        st.subheader("Frontiere efficiente")
        frontier = optimization["frontier"]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["return"], mode="lines", name="Frontiere", line={"color": "#2563eb", "width": 3}))
        figure.add_trace(go.Scatter(x=[optimization["minimum_variance"]["volatility"]], y=[optimization["minimum_variance"]["return"]], mode="markers", name="Variance minimale", marker={"size": 12, "color": "#16a34a"}))
        figure.add_trace(go.Scatter(x=[optimization["maximum_sharpe"]["volatility"]], y=[optimization["maximum_sharpe"]["return"]], mode="markers", name="Sharpe maximal", marker={"size": 12, "color": "#dc2626"}))
        figure.add_trace(go.Scatter(x=[annual_volatility], y=[annual_return], mode="markers", name="Votre portefeuille", marker={"size": 14, "color": "#111827", "symbol": "star"}))
        figure.update_layout(
            xaxis_title="Volatilite annualisee",
            yaxis_title="Rendement annualise",
            height=470,
            legend=dict(orientation="h", y=-0.24, x=0, xanchor="left", yanchor="top", font={"size": 10}),
            margin=dict(t=35, b=105, l=55, r=20),
        )
        st.plotly_chart(figure, use_container_width=True)

with tab_performance:
    wealth = (1 + pd.Series(portfolio_returns, index=returns.index)).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    performance = go.Figure()
    performance.add_trace(go.Scatter(x=wealth.index, y=wealth, name="Valeur de 1 EUR", line={"color": "#2563eb", "width": 2.5}))
    performance.update_layout(height=360, yaxis_title="Valeur", hovermode="x unified", margin=dict(t=25, b=35))
    st.plotly_chart(performance, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Drawdown")
        drawdown_figure = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", line={"color": "#dc2626"}, name="Drawdown"))
        drawdown_figure.update_layout(height=290, yaxis_tickformat=".0%", margin=dict(t=15, b=30))
        st.plotly_chart(drawdown_figure, use_container_width=True)
    with right:
        st.subheader("Volatilite glissante 60 jours")
        rolling_vol = pd.Series(portfolio_returns, index=returns.index).rolling(60).std() * np.sqrt(252)
        vol_figure = go.Figure(go.Scatter(x=rolling_vol.index, y=rolling_vol, line={"color": "#d97706"}, name="Volatilite"))
        vol_figure.update_layout(height=290, yaxis_tickformat=".0%", margin=dict(t=15, b=30))
        st.plotly_chart(vol_figure, use_container_width=True)

with tab_allocation:
    allocation = pd.DataFrame({"Actif": selected, "Poids": weights}).set_index("Actif")
    optimal = pd.DataFrame({"Poids Sharpe maximal": optimization["maximum_sharpe"]["weights"], "Poids variance minimale": optimization["minimum_variance"]["weights"]}, index=assets)
    left, right = st.columns(2)
    with left:
        st.subheader("Votre allocation")
        allocation_figure = go.Figure(go.Bar(x=allocation.index, y=allocation["Poids"], marker_color="#2563eb", text=[f"{value:.1%}" for value in allocation["Poids"]], textposition="outside"))
        allocation_figure.update_layout(height=350, yaxis_tickformat=".0%", yaxis_title="Poids", margin=dict(t=30, b=35))
        st.plotly_chart(allocation_figure, use_container_width=True)
    with right:
        st.subheader("Allocations optimales")
        st.dataframe(optimal.style.format("{:.2%}"), use_container_width=True)
    export = pd.DataFrame({"Actif": selected, "Poids": weights})
    st.download_button("Telecharger l'allocation CSV", export.to_csv(index=False).encode("utf-8"), "allocation_portefeuille.csv", "text/csv")
    st.subheader("Correlation entre actifs")
    correlation = returns[selected].corr()
    correlation_figure = go.Figure(go.Heatmap(z=correlation, x=correlation.columns, y=correlation.index, zmin=-1, zmax=1, colorscale="RdBu", texttemplate="%{z:.2f}"))
    correlation_figure.update_layout(height=430, margin=dict(t=25, b=25))
    st.plotly_chart(correlation_figure, use_container_width=True)

with tab_stress:
    st.subheader("Comportement pendant les crises historiques")
    stress_rows = []
    for name, scenario in stress.items():
        start = pd.Timestamp(scenario["start"])
        end = pd.Timestamp(scenario["end"])
        window = returns.loc[(returns.index >= start) & (returns.index <= end), selected]
        scenario_return = float((1 + window.to_numpy() @ weights).prod() - 1) if not window.empty else np.nan
        stress_rows.append({"Scenario": name, "Periode": f"{start:%d/%m/%Y} - {end:%d/%m/%Y}", "Observations": len(window), "Rendement du portefeuille": scenario_return})
    if stress_rows:
        stress_table = pd.DataFrame(stress_rows)
        st.dataframe(stress_table.style.format({"Rendement du portefeuille": "{:.2%}"}), use_container_width=True)
        stress_figure = go.Figure(go.Bar(x=stress_table["Scenario"], y=stress_table["Rendement du portefeuille"], marker_color="#dc2626", text=stress_table["Rendement du portefeuille"].map(lambda value: f"{value:.1%}"), textposition="outside"))
        stress_figure.update_layout(height=350, yaxis_tickformat=".0%", yaxis_title="Rendement cumule", margin=dict(t=30, b=50))
        st.plotly_chart(stress_figure, use_container_width=True)
    else:
        st.warning("Aucun scenario de crise disponible dans les artefacts.")

st.caption("Les resultats historiques et les optimisations ne constituent pas une recommandation d'investissement.")
