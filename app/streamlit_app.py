import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="Energy Consumption Forecast",
    layout="wide"
)

# --- STYLE CSS ---
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: #FAFAFA;
}

h1 {
    font-size: 40px;
    font-weight: 700;
}

h2, h3 {
    font-weight: 600;
}

.block-container {
    padding-top: 2rem;
}

.section {
    background: #161B22;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

hr {
    border: 1px solid #30363D;
}
</style>
""", unsafe_allow_html=True)

# --- LSTM MODEL ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    scaler = joblib.load("app/scaler.pkl")
    metrics = joblib.load("app/metrics.pkl")

    model = LSTMModel()
    model.load_state_dict(torch.load(
        "app/lstm_model.pth",
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model, scaler, metrics

# --- LOAD DATA (FIX BUG HTTP) ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/badaraaliouguindo/energy-consumption-forecast/main/data/AEP_hourly.csv"

    try:
        df = pd.read_csv(url)
    except Exception:
        st.warning("Impossible de charger depuis GitHub, utilisation du fichier local")
        df = pd.read_csv("app/AEP_hourly.csv")

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    df_daily = df['AEP_MW'].resample('D').mean().dropna()

    return df_daily

# --- LOAD ---
model, scaler, metrics = load_models()
df_daily = load_data()

# --- HEADER ---
st.markdown("<h1>Energy Consumption Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;'>Prévision de la consommation électrique avec Prophet et LSTM</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- METRICS ---
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("Comparaison des modèles")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Prophet MAE",  f"{metrics['prophet']['mae']:.0f} MW")
col2.metric("Prophet RMSE", f"{metrics['prophet']['rmse']:.0f} MW")
col3.metric("Prophet MAPE", f"{metrics['prophet']['mape']:.2f}%")

col4.metric("LSTM MAE", f"{metrics['lstm']['mae']:.0f} MW",
            delta=f"-{metrics['prophet']['mae'] - metrics['lstm']['mae']:.0f} MW")

col5.metric("LSTM RMSE", f"{metrics['lstm']['rmse']:.0f} MW",
            delta=f"-{metrics['prophet']['rmse'] - metrics['lstm']['rmse']:.0f} MW")

col6.metric("LSTM MAPE", f"{metrics['lstm']['mape']:.2f}%",
            delta=f"-{metrics['prophet']['mape'] - metrics['lstm']['mape']:.2f}%")

st.markdown("</div>", unsafe_allow_html=True)

# --- HISTORICAL ---
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("Données historiques")

col_left, col_right = st.columns([3, 1])
with col_right:
    annee_debut = st.slider("Année de début", 2004, 2017, 2015)

df_filtered = df_daily[df_daily.index.year >= annee_debut]

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=df_filtered.index,
    y=df_filtered.values,
    mode='lines',
    name='Consommation réelle',
    line=dict(color='#2EA043', width=2)
))

fig_hist.update_layout(
    title=f"Consommation depuis {annee_debut}",
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- PREDICTIONS ---
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("Prédictions LSTM")

n_days = st.slider("Nombre de jours à prédire", 7, 90, 30)

SEQ_LENGTH = 60
last_sequence = df_daily.values[-SEQ_LENGTH:]
data_scaled = scaler.transform(last_sequence.reshape(-1, 1))

predictions = []
current_seq = data_scaled.copy()

with torch.no_grad():
    for _ in range(n_days):
        x = torch.FloatTensor(current_seq).unsqueeze(0)
        pred = model(x).item()
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

predictions_real = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
).flatten()

last_date = df_daily.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=n_days,
    freq='D'
)

fig_pred = go.Figure()

fig_pred.add_trace(go.Scatter(
    x=df_daily.index[-90:],
    y=df_daily.values[-90:],
    mode='lines',
    name='Historique',
    line=dict(color='#2EA043', width=2)
))

fig_pred.add_trace(go.Scatter(
    x=future_dates,
    y=predictions_real,
    mode='lines',
    name='Prévisions',
    line=dict(color='#F78166', width=2, dash='dash')
))

fig_pred.update_layout(
    title=f"Prévision sur {n_days} jours",
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig_pred, use_container_width=True)

# --- STATS ---
st.markdown("### Statistiques des prédictions")

c1, c2, c3 = st.columns(3)
c1.metric("Moyenne", f"{predictions_real.mean():.0f} MW")
c2.metric("Maximum", f"{predictions_real.max():.0f} MW")
c3.metric("Minimum", f"{predictions_real.min():.0f} MW")

st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#8b949e;'>Projet portfolio — Data Science Master | Dataset : AEP Hourly Energy Consumption</p>",
    unsafe_allow_html=True
)
