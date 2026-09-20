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

st.set_page_config(page_title="Portfolio Risk Lab", page_icon="📈", layout="wide")
st.title("Portfolio Risk Lab")
st.caption("Analyse reproductible du risque et de l'allocation, basée sur des données de marché yfinance.")

@st.cache_data
def load_artifacts():
    if not DATA_PATH.exists() or not OPTIMIZATION_PATH.exists() or not METRICS_PATH.exists():
        return None
    returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    with OPTIMIZATION_PATH.open(encoding="utf-8") as file:
        optimization = json.load(file)
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)
    return returns, optimization, metrics

artifacts = load_artifacts()
if artifacts is None:
    st.error("Les artefacts sont absents. Executez toutes les cellules du notebook avant de lancer Streamlit.")
    st.stop()

returns, optimization, metrics = artifacts
assets = list(returns.columns)
with st.sidebar:
    st.header("Portefeuille")
    selected = st.multiselect("Actifs", assets, default=assets)
    if not selected:
        st.warning("Selectionnez au moins un actif.")
        st.stop()
    equal_weight = np.repeat(1 / len(selected), len(selected))
    weights_input = []
    for asset in selected:
        weights_input.append(st.number_input(f"Poids {asset}", min_value=0.0, max_value=1.0, value=float(1 / len(selected)), step=0.01))
    weights = np.array(weights_input, dtype=float)
    if weights.sum() == 0:
        weights = equal_weight
    weights = weights / weights.sum()
    confidence = st.select_slider("Confiance VaR", options=[0.95, 0.99], value=0.95, format_func=lambda value: f"{value:.0%}")

portfolio_returns = returns[selected].to_numpy() @ weights
losses = -portfolio_returns
alpha = 1 - confidence
historical_var = float(np.quantile(losses, confidence))
mean_daily = float(portfolio_returns.mean())
std_daily = float(portfolio_returns.std(ddof=1))
parametric_var = float(-(mean_daily + std_daily * norm.ppf(alpha)))
cvar = float(losses[losses >= historical_var].mean())
annual_return = float((1 + mean_daily) ** 252 - 1)
annual_volatility = float(std_daily * np.sqrt(252))

metric_cols = st.columns(5)
for column, label, value, format_string in zip(
    metric_cols,
    ["VaR historique", "VaR parametrique", "CVaR", "Rendement annualise", "Volatilite annualisee"],
    [historical_var, parametric_var, cvar, annual_return, annual_volatility],
    ["{:.2%}", "{:.2%}", "{:.2%}", "{:.2%}", "{:.2%}"],
):
    column.metric(label, format_string.format(value))

left, right = st.columns(2)
with left:
    st.subheader("Allocation personnalisee")
    allocation = pd.DataFrame({"Actif": selected, "Poids": weights}).set_index("Actif")
    st.bar_chart(allocation)
    st.dataframe(allocation.style.format({"Poids": "{:.2%}"}), use_container_width=True)
with right:
    st.subheader("Frontiere efficiente")
    frontier = optimization["frontier"]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["return"], mode="lines", name="Frontiere"))
    figure.add_trace(go.Scatter(x=[optimization["minimum_variance"]["volatility"]], y=[optimization["minimum_variance"]["return"]], mode="markers", name="Variance minimale", marker={"size": 12}))
    figure.add_trace(go.Scatter(x=[optimization["maximum_sharpe"]["volatility"]], y=[optimization["maximum_sharpe"]["return"]], mode="markers", name="Sharpe maximal", marker={"size": 12}))
    figure.update_layout(xaxis_title="Volatilite annualisee", yaxis_title="Rendement annualise", height=430, legend=dict(orientation="h"))
    st.plotly_chart(figure, use_container_width=True)

st.subheader("Evolution de la valeur du portefeuille")
wealth = (1 + pd.Series(portfolio_returns, index=returns.index)).cumprod()
st.line_chart(wealth.rename("Valeur de 1 EUR"))

st.subheader("Allocation suggeree par l'optimisation")
optimal = pd.DataFrame({"Actif": assets, "Poids Sharpe maximal": optimization["maximum_sharpe"]["weights"], "Poids variance minimale": optimization["minimum_variance"]["weights"]}).set_index("Actif")
st.dataframe(optimal.style.format("{:.2%}"), use_container_width=True)
st.caption("Les resultats historiques et les optimisations ne constituent pas une recommandation d'investissement.")
