
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Check Bayesian optimization availability
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    bayes_available = True
except ImportError:
    bayes_available = False

st.set_page_config(layout="wide")
st.title("🎯 Angebots‑Simulation & Bidding‑Strategy‑Tool")

# ── Scenario Manager ──────────────────────────────────────────────────────────
default_scenarios = {
    "Freie Eingabe":   {"P10":500,"P50":950,"P90":1300,"comp_qual_mean":7.0,"sigma_mult":1.0},
    "Balanced Market": {"P10":500,"P50":950,"P90":1300,"comp_qual_mean":7.0,"sigma_mult":1.0},
    "Discount Push":   {"P10":400,"P50":800,"P90":1200,"comp_qual_mean":5.5,"sigma_mult":1.2},
    "Premium Focus":   {"P10":600,"P50":1100,"P90":1500,"comp_qual_mean":9.0,"sigma_mult":0.8},
}
if "scenarios" not in st.session_state:
    st.session_state.scenarios = default_scenarios.copy()
if "scenario" not in st.session_state:
    st.session_state.scenario = "Freie Eingabe"
for k,v in st.session_state.scenarios["Freie Eingabe"].items():
    st.session_state.setdefault(k, v)
def load_scenario():
    st.session_state.update(st.session_state.scenarios[st.session_state.scenario])

st.sidebar.header("💾 Szenario‑Manager")
st.sidebar.selectbox("Szenario laden:", list(st.session_state.scenarios.keys()),
                     key="scenario", on_change=load_scenario)

# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Globale Einstellungen")
quality        = st.sidebar.slider("Qualität (0–10)",0,10,8)
tagessatz      = st.sidebar.slider("Brutto-Tagessatz (€)",400,1500,st.session_state.P50,step=50)
discount_pct   = st.sidebar.slider("Rabatt (%)",0,50,0,step=5,format="%d%%")
discount       = discount_pct/100.0
projekttage    = st.sidebar.number_input("Projekttage",1,1000,50)
cost_per_day   = st.sidebar.number_input("Kosten pro Tag (€)",1,10000,500)
margin_pct     = st.sidebar.slider("Min. Deckungsbeitrag (%)",0,100,20)
wettbewerber_n = st.sidebar.slider("Anzahl Wettbewerber",1,20,10)
tage_dev       = st.sidebar.slider("Tage-Abweichung (%)",0.0,0.5,0.3,step=0.05)
sim_runs       = st.sidebar.slider("Simulationsläufe",100,5000,1000,step=100)
wQ             = st.sidebar.slider("Gewichtung Qualität (wQ)",0.0,1.0,0.3,step=0.1)
wP             = 1 - wQ
P10            = st.sidebar.number_input("P10 Preis WB (€)",1,10000,st.session_state.P10)
P50            = st.sidebar.number_input("P50 Preis WB (€)",1,10000,st.session_state.P50)
P90            = st.sidebar.number_input("P90 Preis WB (€)",1,10000,st.session_state.P90)
comp_qual_mean = st.sidebar.slider("Mittl. WB-Qualität (0–10)",0.0,10.0,st.session_state.comp_qual_mean,step=0.5)
sigma_mult     = st.sidebar.slider("Preis-Volatilität Faktor",0.5,2.0,st.session_state.sigma_mult,step=0.1)
threshold_pct  = st.sidebar.slider("Min. WinRate (%)",0,100,70,step=5,format="%d%%")
threshold      = threshold_pct/100.0

# ── Simulation Function ────────────────────────────────────────────────────────
def simulate_comp(qual_mean, sigma_mul, t_dev):
    mu=np.log(P50)
    sigma=(np.log(P90)-np.log(P10))/2.563103131*sigma_mul
    rates=np.random.lognormal(mu,sigma,(sim_runs,wettbewerber_n))
    days=projekttage*np.random.uniform(1-t_dev,1+t_dev,(sim_runs,wettbewerber_n))
    total=rates*days
    price_s=1-total/(P90*projekttage)
    qual_s=np.clip(np.random.normal(qual_mean,1,(sim_runs,wettbewerber_n)),0,10)/10
    return wQ*qual_s + wP*price_s

np.random.seed(42)
comp_scores=simulate_comp(comp_qual_mean,sigma_mult,tage_dev)
comp_max=comp_scores.max(axis=1)

# ── Own Metrics ───────────────────────────────────────────────────────────────
our_net      = tagessatz*(1-discount)
our_margin   = (our_net-cost_per_day)/our_net
our_score    = wQ*(quality/10)+wP*(1-(our_net*projekttage)/(P90*projekttage))
win_rate     = (our_score>comp_max).mean()
exp_profit   = win_rate*our_net*our_margin*projekttage

# ── 1. WinRate ────────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='text-align:center;'>🏆 WinRate: {win_rate*100:.2f}%</h1>",unsafe_allow_html=True)

# ── 2. Bayesian Optimization ─────────────────────────────────────────────────
st.subheader("🤖 Automatische ExpProfit-Optimierung")
if bayes_available:
    space=[Integer(400,1500),Real(0.0,0.5)]
    def obj(x):
        r,d=x;net=r*(1-d)
        wr=(wQ*(quality/10)+wP*(1-(net*projekttage)/(P90*projekttage)))>comp_max
        return -wr.mean()*net*((net-cost_per_day)/net)*projekttage
    res=gp_minimize(obj,space,n_calls=20,random_state=0)
    br,bd=res.x;bp=-res.fun
    st.write(f"Optimal: Brutto {br} €, Rabatt {bd*100:.0f}% → Effektiv {br*(1-bd):.0f} € | ExpProfit {bp:.2f} €")
else:
    st.info("pip install scikit-optimize für Optimierung")

# ── 3. Heatmap ─────────────────────────────────────────────────────────────────
st.subheader("🔍 Heatmap: Gewinnwahrscheinlichkeit")
rates=np.arange(400,1501,50)
discounts=np.arange(0,51,5)/100
grid=np.array([[ (wQ*(quality/10)+wP*(1-((r*(1-d))*projekttage)/(P90*projekttage))>comp_max).mean()
                 for d in discounts] for r in rates])
fig_heat=px.imshow(grid,x=[f"{int(d*100)}%" for d in discounts],y=rates,
                   labels={'x':'Rabatt','y':'Brutto-Tagessatz (€)','color':'WinRate'},
                   color_continuous_scale='Blues',origin='lower',aspect='auto')
fig_heat.update_layout(height=600,margin=dict(l=80,r=20,t=50,b=50))
st.plotly_chart(fig_heat,use_container_width=True)

# ── 4. Top-5 Empfehlungen ──────────────────────────────────────────────────────
st.subheader("🏅 Top‑5-Angebote")
records=[]
for r in rates:
    wr=(wQ*(quality/10)+wP*(1-(r*projekttage)/(P90*projekttage))>comp_max).mean()
    mar=(r-cost_per_day)/r*100
    if mar>0 and wr>=threshold:
        records.append({'Brutto (€)':int(r),'WinRate':wr,'Marge (%)':mar})
df=pd.DataFrame(records)
if df.empty:
    df=pd.DataFrame([{'Brutto (€)':int(r),
                      'WinRate':(wQ*(quality/10)+wP*(1-(r*projekttage)/(P90*projekttage))>comp_max).mean(),
                      'Marge (%)':(r-cost_per_day)/r*100} for r in rates])
df_top=df.nlargest(5,'WinRate')
df_top['WinRate (%)']=df_top['WinRate']*100
st.table(df_top[['Brutto (€)','WinRate (%)','Marge (%)']].style.format({'WinRate (%)':'{:.2f}%','Marge (%)':'{:.1f}%'}))

# ── 5. Pareto-Analyse ─────────────────────────────────────────────────────────
st.subheader("📊 Pareto-Analyse: WinRate vs. ExpProfit")
pareto=[]
for r in rates:
    for d in discounts:
        net=r*(1-d)
        wr=(wQ*(quality/10)+wP*(1-(net*projekttage)/(P90*projekttage))>comp_max).mean()
        prof=wr*net*((net-cost_per_day)/net)*projekttage
        pareto.append({'WinRate':wr,'Profit':prof})
df_p=pd.DataFrame(pareto)
best=df_p.loc[df_p.Profit.idxmax()]
st.info(f"🎯 Sweet Spot → WinRate {best.WinRate:.1%}, ExpProfit {best.Profit:.2f} €")
fig_p=px.scatter(df_p,x='WinRate',y='Profit')
fig_p.update_layout(xaxis_title='WinRate',yaxis_title='ExpProfit (€)')
st.plotly_chart(fig_p,use_container_width=True)

# ── 6. Szenario‑Vergleich ───────────────────────────────────────────────────────
st.subheader("🔍 Szenario‑Vergleich")
chosen=st.multiselect("Szenarien",list(default_scenarios.keys()),default=["Balanced Market"])
if chosen:
    rows=[]
    for scen in chosen:
        p=default_scenarios[scen]
        comp_s=simulate_comp(p['comp_qual_mean'],p['sigma_mult'],tage_dev)
        wr=(our_score>comp_s.max(axis=1)).mean()*100
        prof=wr/100*our_net*our_margin*projekttage
        rows.append({'Szenario':scen,'WinRate (%)':wr,'ExpProfit (€)':prof})
    df_cmp=pd.DataFrame(rows)
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=df_cmp['Szenario'],y=df_cmp['ExpProfit (€)'],name='ExpProfit €'),secondary_y=False)
    fig.add_trace(go.Scatter(x=df_cmp['Szenario'],y=df_cmp['WinRate (%)'],name='WinRate %',mode='lines+markers'),secondary_y=True)
    fig.update_layout(xaxis_title='Szenario',yaxis_title='ExpProfit (€)',height=400)
    fig.update_yaxes(title='WinRate (%)',secondary_y=True,range=[0,100])
    st.plotly_chart(fig,use_container_width=True)
