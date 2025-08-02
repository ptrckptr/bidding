import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Bayes optional laden ---
try:
    from skopt import gp_minimize
    bayes_available = True
except Exception:
    bayes_available = False

st.set_page_config(page_title="phi Bidding Strategizer", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("phi Bidding Strategizer")
st.caption("Monte-Carlo-basiertes Angebots-Tuning mit optionaler Bayes-Optimierung")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: Konfiguration
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Einstellungen")

# Seed/Nonce für frische Zufallsläufe
if "nonce" not in st.session_state:
    st.session_state.nonce = int(time.time())
if st.sidebar.button("🔄 Neu simulieren (frische Zufallsläufe)"):
    st.session_state.nonce = int(time.time())

quality        = st.sidebar.slider("Qualität (0–10)", 0, 10, 8)
tagessatz      = st.sidebar.slider("Brutto-Tagessatz (€)", 400, 1500, 950, step=50)
discount_pct   = st.sidebar.slider("Rabatt (%)", 0, 50, 0, step=5)
discount       = discount_pct / 100.0
projekttage    = st.sidebar.number_input("Projekttage", 1, 1000, 50)
cost_per_day   = st.sidebar.number_input("Kosten pro Tag (€)", 1, 10000, 500)
wettbewerber_n = st.sidebar.slider("Anzahl Wettbewerber", 1, 20, 10)
tage_dev       = st.sidebar.slider("Tage-Abweichung (%)", 0.0, 0.5, 0.3)
sim_runs       = st.sidebar.slider("Simulationsläufe", 100, 5000, 1000, step=100)

wQ = st.sidebar.slider("Gewichtung Qualität (wQ)", 0.0, 1.0, 0.3)
wP = 1 - wQ

P10            = st.sidebar.number_input("P10 Preis Wettbewerber (€)", 1, 10000, 500)
P50            = st.sidebar.number_input("P50 Preis Wettbewerber (€)", 1, 10000, 950)
P90            = st.sidebar.number_input("P90 Preis Wettbewerber (€)", 1, 10000, 1300)
comp_qual_mean = st.sidebar.slider("Mittl. Wettbewerber-Qualität (0–10)", 0.0, 10.0, 7.0)
sigma_mult     = st.sidebar.slider("Preis-Volatilität Faktor", 0.5, 2.0, 1.0)

threshold_pct  = st.sidebar.slider("Min. WinRate (%) (Top-5 Filter)", 0, 100, 70)
threshold      = threshold_pct / 100.0

# ──────────────────────────────────────────────────────────────────────────────
# Cache-Funktionen
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def simulate_comp_cached(sim_runs, wettbewerber_n, projekttage,
                         P10, P50, P90, wQ, wP, comp_qual_mean,
                         sigma_mult, tage_dev, nonce):
    """Simuliert Wettbewerber-Scores (Qualität + Preisattraktivität)."""
    rng    = np.random.default_rng(nonce)
    mu     = np.log(P50)
    sigma  = (np.log(P90) - np.log(P10)) / 2.563103131 * sigma_mult
    rates  = rng.lognormal(mu, sigma, (sim_runs, wettbewerber_n))
    days   = projekttage * rng.uniform(1 - tage_dev, 1 + tage_dev, (sim_runs, wettbewerber_n))
    total  = rates * days
    price_score = 1 - total / (P90 * projekttage)                       # 1 = sehr günstig
    qual_score  = np.clip(rng.normal(comp_qual_mean, 1, (sim_runs, wettbewerber_n)), 0, 10) / 10
    return wQ * qual_score + wP * price_score

@st.cache_data(ttl=600)
def compute_heatmap_grid(rates, discounts, comp_max, projekttage, P90, wQ, wP, quality):
    grid = np.empty((len(rates), len(discounts)), dtype=float)
    for i, r in enumerate(rates):
        for j, d in enumerate(discounts):
            net = r * (1 - d)
            our_score = wQ*(quality/10) + wP*(1 - (net*projekttage)/(P90*projekttage))
            grid[i, j] = (our_score > comp_max).mean()
    return grid

@st.cache_data(ttl=600)
def compute_pareto_df(rates, discounts, comp_max, projekttage, P90, wQ, wP, quality, cost_per_day):
    rows = []
    for r in rates:
        for d in discounts:
            net = r*(1-d)
            our_score = wQ*(quality/10) + wP*(1 - (net*projekttage)/(P90*projekttage))
            wr = (our_score > comp_max).mean()
            profit = wr * (net - cost_per_day) * projekttage
            rows.append({"Brutto": r, "Rabatt_%": d*100, "WinRate": wr, "Profit": profit})
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────────────────────
comp_scores = simulate_comp_cached(sim_runs, wettbewerber_n, projekttage,
                                   P10, P50, P90, wQ, wP, comp_qual_mean,
                                   sigma_mult, tage_dev, st.session_state.nonce)
comp_max = comp_scores.max(axis=1)

our_net        = tagessatz * (1 - discount)
our_score      = wQ*(quality/10) + wP*(1 - (our_net*projekttage)/(P90*projekttage))
win_rate       = (our_score > comp_max).mean()
profit_if_win  = (our_net - cost_per_day) * projekttage
exp_profit     = win_rate * profit_if_win

# KPIs
st.header(f"WinRate: {win_rate*100:.2f}%")
c1, c2, c3 = st.columns(3)
c1.metric("Profit bei Gewinn (€/Projekt)", f"{profit_if_win:,.0f}".replace(",", "."))
c2.metric("WinRate (%)", f"{win_rate*100:.2f}%")
c3.metric("ExpProfit (€/Projekt)", f"{exp_profit:,.0f}".replace(",", "."))

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Bayes-Optimierung
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Automatische ExpProfit-Optimierung (Bayes)")
if bayes_available:
    space = [(400, 1500), (0.0, 0.5)]  # (Brutto, Rabatt)
    def objective(x):
        r, d = x
        net = r*(1-d)
        score = wQ*(quality/10) + wP*(1 - (net*projekttage)/(P90*projekttage))
        wr = (score > comp_max).mean()
        return -(wr * (net - cost_per_day) * projekttage)
    res = gp_minimize(objective, space, n_calls=25, random_state=0)
    br, bd = res.x
    bp     = -res.fun
    st.info(f"Optimal: Brutto {br:.0f} €, Rabatt {bd*100:.0f}% → Effektiv {br*(1-bd):.0f} € | ExpProfit {bp:.0f} €")
else:
    st.info("Bayes-Optimierung verfügbar nach Installation:  `pip install scikit-optimize`")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Heatmap
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Heatmap: Gewinnwahrscheinlichkeit")
rates     = np.arange(400, 1501, 50)
discounts = np.arange(0, 51, 5) / 100.0
grid      = compute_heatmap_grid(rates, discounts, comp_max, projekttage, P90, wQ, wP, quality)
fig_heat  = px.imshow(
    grid,
    x=[f"{int(d*100)}%" for d in discounts],
    y=rates,
    labels={'x':'Rabatt','y':'Brutto-Tagessatz (€)','color':'WinRate'},
    color_continuous_scale='Blues',
    origin='lower',
    aspect='auto'
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Top-5-Angebote
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Top-5-Angebote (Marge > 0 & WinRate-Schwelle)")
records=[]
for r in rates:
    wr  = (wQ*(quality/10) + wP*(1 - (r*projekttage)/(P90*projekttage)) > comp_max).mean()
    mar = (r - cost_per_day)/r*100
    if mar > 0 and wr >= threshold:
        records.append({'Brutto (€)': int(r), 'WinRate': wr, 'Marge (%)': mar})
df_all = pd.DataFrame(records)
if df_all.empty:
    st.info("Keine Kombination erfüllt aktuell die Filter. Zeige beste 5 nach WinRate (Marge > 0).")
    for r in rates:
        wr  = (wQ*(quality/10) + wP*(1 - (r*projekttage)/(P90*projekttage)) > comp_max).mean()
        mar = (r - cost_per_day)/r*100
        if mar > 0:
            records.append({'Brutto (€)': int(r), 'WinRate': wr, 'Marge (%)': mar})
    df_all = pd.DataFrame(records)
if not df_all.empty:
    df_top5 = df_all.nlargest(5, 'WinRate').copy()
    df_top5['WinRate (%)'] = df_top5['WinRate']*100
    st.table(
        df_top5[['Brutto (€)','WinRate (%)','Marge (%)']].style.format({
            'WinRate (%)':'{:.2f}%',
            'Marge (%)':'{:.1f}%'
        })
    )
else:
    st.warning("Auch ohne Filter wurden keine positiven Margen gefunden.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Pareto
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Pareto-Analyse: WinRate vs. ExpProfit")
df_pareto = compute_pareto_df(rates, discounts, comp_max, projekttage, P90, wQ, wP, quality, cost_per_day)
if not df_pareto.empty and df_pareto['Profit'].notna().any():
    best = df_pareto.loc[df_pareto['Profit'].idxmax()]
    st.info(f"🎯 Sweet Spot: WinRate {best.WinRate:.1%}, ExpProfit {best.Profit:.0f} €")
    fig_par = px.scatter(df_pareto, x='WinRate', y='Profit', hover_data=['Brutto','Rabatt_%'])
    fig_par.update_layout(xaxis_title='WinRate', yaxis_title='ExpProfit (€)')
    st.plotly_chart(fig_par, use_container_width=True)
else:
    st.info("Pareto-Daten nicht verfügbar.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Szenario-Vergleich (inkl. „Aktuelle Eingaben“)
# ──────────────────────────────────────────────────────────────────────────────
defaults = {
    "Balanced Market": {"P10":500, "P50":950,  "P90":1300, "comp_qual_mean":7.0, "sigma_mult":1.0},
    "Discount Push":   {"P10":400, "P50":800,  "P90":1200, "comp_qual_mean":5.5, "sigma_mult":1.2},
    "Premium Focus":   {"P10":600, "P50":1100, "P90":1500, "comp_qual_mean":9.0, "sigma_mult":0.8},
}
current = {"P10":P10, "P50":P50, "P90":P90, "comp_qual_mean":comp_qual_mean, "sigma_mult":sigma_mult}
all_scenarios = {"Aktuelle Eingaben": current, **defaults}

st.subheader("Szenario-Vergleich")
opts = list(all_scenarios.keys())
chosen = st.multiselect("Szenarien auswählen", opts, default=["Aktuelle Eingaben", "Balanced Market"])
if chosen:
    rows=[]
    for name in chosen:
        p = all_scenarios[name]
        comp_scores_s = simulate_comp_cached(
            sim_runs, wettbewerber_n, projekttage,
            p['P10'], p['P50'], p['P90'], wQ, wP, p['comp_qual_mean'],
            p['sigma_mult'], tage_dev, st.session_state.nonce
        )
        comp_max_s = comp_scores_s.max(axis=1)
        wr   = (our_score > comp_max_s).mean()*100
        prof = (wr/100.0) * (our_net - cost_per_day) * projekttage
        rows.append({'Szenario': name, 'WinRate (%)': wr, 'ExpProfit (€)': prof})
    df_cmp = pd.DataFrame(rows)
    fig_cmp = make_subplots(specs=[[{"secondary_y":True}]])
    fig_cmp.add_trace(go.Bar(x=df_cmp['Szenario'], y=df_cmp['ExpProfit (€)'], name='ExpProfit (€)'), secondary_y=False)
    fig_cmp.add_trace(go.Scatter(x=df_cmp['Szenario'], y=df_cmp['WinRate (%)'], name='WinRate (%)', mode='lines+markers'), secondary_y=True)
    fig_cmp.update_layout(xaxis_title='Szenario', height=420)
    fig_cmp.update_yaxes(title_text='ExpProfit (€)', secondary_y=False)
    fig_cmp.update_yaxes(title_text='WinRate (%)',  secondary_y=True, range=[0,100])
    st.plotly_chart(fig_cmp, use_container_width=True)
