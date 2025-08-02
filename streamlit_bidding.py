import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from skopt import gp_minimize
    bayes_available = True
except Exception:
    bayes_available = False

st.set_page_config(page_title="phi Bidding Strategizer", layout="wide", initial_sidebar_state="expanded")
st.title("phi Bidding Strategizer")
st.caption("Monte-Carlo-basiertes Angebots-Tuning mit optionaler Bayes-Optimierung")
st.set_option("client.showErrorDetails", True)

if "nonce" not in st.session_state:
    st.session_state.nonce = int(time.time())
if "active_params" not in st.session_state:
    st.session_state.active_params = None

st.sidebar.caption("Build: compute-on-apply v1.3")

st.sidebar.header("Einstellungen")
quality        = st.sidebar.slider("Qualität (0–10)", 0, 10, 8)
tagessatz      = st.sidebar.slider("Brutto-Tagessatz (€)", 400, 1500, 950, step=50)
discount_pct   = st.sidebar.slider("Rabatt (%)", 0, 50, 0, step=5)
discount       = discount_pct/100.0
projekttage    = st.sidebar.number_input("Projekttage", min_value=1, max_value=1000, value=50, step=1)
cost_per_day   = st.sidebar.number_input("Kosten pro Tag (€)", min_value=1.0, max_value=10000.0, value=500.0, step=1.0)
wettbewerber_n = st.sidebar.slider("Anzahl Wettbewerber", 1, 20, 10)
tage_dev       = st.sidebar.slider("Tage-Abweichung (%)", 0.0, 0.5, 0.3)
sim_runs       = st.sidebar.slider("Simulationsläufe", 100, 10000, 2000, step=100)
wQ = st.sidebar.slider("Gewichtung Qualität (wQ)", 0.0, 1.0, 0.3)
wP = 1 - wQ
P10            = st.sidebar.number_input("P10 Preis Wettbewerber (€)", min_value=1.0, max_value=10000.0, value=500.0, step=1.0)
P50            = st.sidebar.number_input("P50 Preis Wettbewerber (€)", min_value=1.0, max_value=10000.0, value=950.0, step=1.0)
P90            = st.sidebar.number_input("P90 Preis Wettbewerber (€)", min_value=1.0, max_value=10000.0, value=1300.0, step=1.0)
comp_qual_mean = st.sidebar.slider("Mittl. Wettbewerber-Qualität (0–10)", 0.0, 10.0, 7.0)
sigma_mult     = st.sidebar.slider("Preis-Volatilität Faktor", 0.5, 2.0, 1.0)
threshold_pct  = st.sidebar.slider("Min. WinRate (%) (Top-5 Filter)", 0, 100, 70)
threshold      = threshold_pct / 100.0
st.sidebar.markdown("---")
apply_clicked   = st.sidebar.button("⚙️ Anwenden / Rechnen", use_container_width=True)
refresh_clicked = st.sidebar.button("🔄 Neu simulieren (frische Zufallsläufe)", use_container_width=True)

if apply_clicked:
    p10, p50, p90 = sorted([float(P10), float(P50), float(P90)])
    st.session_state.active_params = dict(
        quality=float(quality), tagessatz=float(tagessatz), discount=float(discount),
        projekttage=int(projekttage), cost_per_day=float(cost_per_day),
        wettbewerber_n=int(wettbewerber_n), tage_dev=float(tage_dev),
        sim_runs=int(sim_runs), wQ=float(wQ), wP=float(1-wQ),
        P10=float(p10), P50=float(p50), P90=float(p90),
        comp_qual_mean=float(comp_qual_mean), sigma_mult=float(sigma_mult),
        threshold=float(threshold)
    )
if refresh_clicked:
    st.session_state.nonce = int(time.time())

@st.cache_data(ttl=900)
def simulate_comp_cached(sim_runs, wettbewerber_n, projekttage,
                         P10, P50, P90, wQ, wP, comp_qual_mean,
                         sigma_mult, tage_dev, nonce):
    rng    = np.random.default_rng(nonce)
    mu     = float(np.log(max(P50, 1e-9)))
    sigma  = float(abs((np.log(max(P90, 1e-9)) - np.log(max(P10, 1e-9))) / 2.563103131) * sigma_mult)
    rates  = rng.lognormal(mean=mu, sigma=sigma, size=(sim_runs, wettbewerber_n))
    days   = projekttage * rng.uniform(1 - tage_dev, 1 + tage_dev, size=(sim_runs, wettbewerber_n))
    total  = rates * days
    price_score = 1 - total / max(P90 * projekttage, 1e-9)
    qual_raw    = rng.normal(comp_qual_mean, 1, size=(sim_runs, wettbewerber_n))
    qual_score  = np.clip(qual_raw, 0, 10) / 10.0
    comp_scores = wQ * qual_score + wP * price_score
    comp_scores = np.where(np.isfinite(comp_scores), comp_scores, 0.0)
    return comp_scores

@st.cache_data(ttl=900)
def compute_heatmap_grid_fast(rates, discounts, comp_max, projekttage, denom, wQ, wP, quality):
    R = rates[:, None].astype(float)
    D = discounts[None, :].astype(float)
    net  = R * (1.0 - D)
    baseQ = wQ * (quality / 10.0)
    price_term = 1.0 - (net * projekttage) / denom
    our_score  = baseQ + wP * price_term
    cm = comp_max[None, None, :]
    os = our_score[:, :, None]
    wr = (os > cm).mean(axis=2)
    wr = np.where(np.isfinite(wr), wr, 0.0)
    return wr

@st.cache_data(ttl=900)
def compute_pareto_df(rates, discounts, comp_max, projekttage, denom, wQ, wP, quality, cost_per_day):
    rows = []
    baseQ = wQ * (quality / 10.0)
    for r in rates:
        for d in discounts:
            net       = r * (1 - d)
            price_term= 1 - (net * projekttage) / denom
            our_score = baseQ + wP * price_term
            wr        = float((our_score > comp_max).mean())
            if not np.isfinite(wr): wr = 0.0
            profit    = wr * (net - cost_per_day) * projekttage
            if not np.isfinite(profit): profit = 0.0
            rows.append({"Brutto": r, "Rabatt_%": d*100, "WinRate": wr, "Profit": profit})
    return pd.DataFrame(rows)

params = st.session_state.active_params
if not params:
    st.info("Wähle links deine Einstellungen und klicke „⚙️ Anwenden / Rechnen“.")
    st.stop()

q         = params["quality"]
rate      = params["tagessatz"]
disc      = params["discount"]
days      = params["projekttage"]
cpd       = params["cost_per_day"]
N         = params["wettbewerber_n"]
dev       = params["tage_dev"]
T         = params["sim_runs"]
wQ_       = params["wQ"]; wP_ = params["wP"]
P10_      = params["P10"]; P50_ = params["P50"]; P90_ = params["P90"]
cqm       = params["comp_qual_mean"]; sgm = params["sigma_mult"]
thresh    = params["threshold"]

denom = max(P90_ * days, 1e-9)

comp_scores = simulate_comp_cached(T, N, days, P10_, P50_, P90_, wQ_, wP_, cqm, sgm, dev, st.session_state.nonce)
comp_max    = comp_scores.max(axis=1)

our_net   = rate * (1 - disc)
our_score = wQ_*(q/10.0) + wP_*(1 - (our_net * days) / denom)
our_score = float(our_score) if np.isfinite(our_score) else 0.0
win_rate  = float((our_score > comp_max).mean())
profit_if_win = (our_net - cpd) * days
exp_profit    = win_rate * profit_if_win

st.header(f"WinRate: {win_rate*100:.2f}%")
c1, c2, c3 = st.columns(3)
c1.metric("Profit bei Gewinn (€/Projekt)", f"{profit_if_win:,.0f}".replace(",", "."))
c2.metric("WinRate (%)", f"{win_rate*100:.2f}%")
c3.metric("ExpProfit (€/Projekt)", f"{exp_profit:,.0f}".replace(",", "."))

st.divider()


st.subheader("Automatische ExpProfit-Optimierung (Bayes)")
def _bayes_subset(arr: np.ndarray, target: int, seed: int) -> np.ndarray:
    s = arr.size
    if target >= s: return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(s, size=target, replace=False)
    return arr[idx]
if not bayes_available:
    st.info("Bayes-Optimierung verfügbar nach Installation: `pip install scikit-optimize`")
else:
    big_job = (T > 3000) or (N > 12)
    TIME_BUDGET = 5.5
    N_CALLS     = 10 if big_job else 14
    SUBSET      = min(1000, comp_max.size)
    if big_job:
        st.warning("Bayes in der Cloud gedrosselt: weniger Aufrufe & Downsampling.")
    if st.button("▶︎ Bayes-Optimierung starten", use_container_width=True):
        import time as _t
        from skopt import Optimizer
        start_t = _t.time()
        try:
            comp_ref = _bayes_subset(comp_max, SUBSET, st.session_state.nonce)
            space = [(400.0, 1500.0), (0.0, 0.5)]
            opt = Optimizer(space, base_estimator="GP", acq_func="EI", random_state=0, n_initial_points=max(4, N_CALLS//3))
            prog = st.progress(0.0, text="Bayes läuft …")
            calls_done = 0
            best_x = None
            best_y = 1e12
            while calls_done < N_CALLS and (_t.time() - start_t) < TIME_BUDGET:
                x = opt.ask()
                r, d = x
                net   = r * (1 - d)
                score = wQ_*(q/10.0) + wP_*(1 - (net * days) / denom)
                wr    = float((score > comp_ref).mean())
                if not np.isfinite(wr):
                    y = 1e12
                else:
                    val = wr * (net - cpd) * days
                    y = -val if np.isfinite(val) else 1e12
                opt.tell(x, y)
                calls_done += 1
                prog.progress(min(1.0, calls_done/max(1, N_CALLS)), text=f"Bayes läuft … ({calls_done}/{N_CALLS})")
                if y < best_y:
                    best_y = y
                    best_x = x
            if best_x is None and opt.yi:
                idx = int(np.argmin(opt.yi))
                best_x = opt.Xi[idx]
                best_y = opt.yi[idx]
            if best_x is None:
                st.error("Keine valide Lösung gefunden.")
            else:
                br, bd = best_x
                bp     = -best_y
                st.success(f"Optimal: Brutto {br:.0f} €, Rabatt {bd*100:.0f}% → Effektiv {br*(1-bd):.0f} € | ExpProfit {bp:.0f} €")
        except Exception as e:
            st.error("Bayes-Optimierung abgebrochen.")
            st.exception(e)
st.divider()

st.subheader("Heatmap: Gewinnwahrscheinlichkeit")
rates     = np.arange(400, 1501, 50)
discounts = np.arange(0, 51, 5) / 100.0
grid      = compute_heatmap_grid_fast(rates, discounts, comp_max, days, denom, wQ_, wP_, q)
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

st.subheader("Top-5-Angebote (Marge > 0 & WinRate-Schwelle)")
records=[]
baseQ = wQ_*(q/10.0)
for r in rates:
    price_term= 1 - (r * days) / denom
    our_sc_r  = baseQ + wP_ * price_term
    wr        = float((our_sc_r > comp_max).mean())
    mar       = (r - cpd) / max(r, 1e-9) * 100.0
    if np.isfinite(wr) and np.isfinite(mar) and (mar > 0) and (wr >= thresh):
        records.append({'Brutto (€)': int(r), 'WinRate': wr, 'Marge (%)': mar})
df_all = pd.DataFrame(records)
if df_all.empty:
    for r in rates:
        price_term= 1 - (r * days) / denom
        our_sc_r  = baseQ + wP_*price_term
        wr        = float((our_sc_r > comp_max).mean())
        mar       = (r - cpd) / max(r, 1e-9) * 100.0
        if np.isfinite(wr) and np.isfinite(mar) and (mar > 0):
            df_all = pd.concat([df_all, pd.DataFrame([{'Brutto (€)': int(r), 'WinRate': wr, 'Marge (%)': mar}])], ignore_index=True)
if not df_all.empty:
    df_top5 = df_all.nlargest(5, 'WinRate').copy()
    df_top5['WinRate (%)'] = df_top5['WinRate']*100.0
    st.dataframe(
        df_top5[['Brutto (€)','WinRate (%)','Marge (%)']].round({'WinRate (%)':2,'Marge (%)':1}),
        use_container_width=True
    )
else:
    st.info("Keine Kombination mit positiver Marge gefunden.")

st.divider()

st.subheader("Pareto-Analyse: WinRate vs. ExpProfit")
df_pareto = compute_pareto_df(rates, discounts, comp_max, days, denom, wQ_, wP_, q, cpd)
if not df_pareto.empty and df_pareto['Profit'].notna().any():
    best = df_pareto.loc[df_pareto['Profit'].idxmax()]
    st.info(f"🎯 Sweet Spot: WinRate {best.WinRate:.1%}, ExpProfit {best.Profit:.0f} €")
    fig_par = px.scatter(df_pareto, x='WinRate', y='Profit', hover_data=['Brutto','Rabatt_%'])
    fig_par.update_layout(xaxis_title='WinRate', yaxis_title='ExpProfit (€)')
    st.plotly_chart(fig_par, use_container_width=True)
else:
    st.info("Pareto-Daten nicht verfügbar.")

st.divider()
st.subheader("Szenario-Vergleich")
defaults = {
    "Balanced Market": {"P10":500, "P50":950,  "P90":1300, "comp_qual_mean":7.0, "sigma_mult":1.0},
    "Discount Push":   {"P10":400, "P50":800,  "P90":1200, "comp_qual_mean":5.5, "sigma_mult":1.2},
    "Premium Focus":   {"P10":600, "P50":1100, "P90":1500, "comp_qual_mean":9.0, "sigma_mult":0.8},
}
current = {"P10":P10_, "P50":P50_, "P90":P90_, "comp_qual_mean":cqm, "sigma_mult":sgm}
all_scenarios = {"Aktuelle Eingaben": current, **defaults}
opts = list(all_scenarios.keys())
chosen = st.multiselect("Szenarien auswählen", opts, default=["Aktuelle Eingaben", "Balanced Market"])
if chosen:
    rows=[]
    baseQ = wQ_*(q/10.0)
    for name in chosen:
        p = all_scenarios[name]
        comp_scores_s = simulate_comp_cached(
            T, N, days, p['P10'], p['P50'], p['P90'], wQ_, wP_, p['comp_qual_mean'],
            p['sigma_mult'], dev, st.session_state.nonce
        )
        comp_max_s = comp_scores_s.max(axis=1)
        denom_s    = max(p['P90'] * days, 1e-9)
        our_sc_s   = baseQ + wP_*(1 - (our_net * days) / denom_s)
        wr         = float((our_sc_s > comp_max_s).mean()) * 100.0
        prof       = (wr/100.0) * (our_net - cpd) * days
        rows.append({'Szenario': name, 'WinRate (%)': wr, 'ExpProfit (€)': prof})
    df_cmp = pd.DataFrame(rows)
    fig_cmp = make_subplots(specs=[[{"secondary_y":True}]])
    fig_cmp.add_trace(go.Bar(x=df_cmp['Szenario'], y=df_cmp['ExpProfit (€)'], name='ExpProfit (€)'), secondary_y=False)
    fig_cmp.add_trace(go.Scatter(x=df_cmp['Szenario'], y=df_cmp['WinRate (%)'], name='WinRate (%)', mode='lines+markers'), secondary_y=True)
    fig_cmp.update_layout(xaxis_title='Szenario', height=420)
    fig_cmp.update_yaxes(title_text='ExpProfit (€)', secondary_y=False)
    fig_cmp.update_yaxes(title_text='WinRate (%)',  secondary_y=True, range=[0,100])
    st.plotly_chart(fig_cmp, use_container_width=True)
