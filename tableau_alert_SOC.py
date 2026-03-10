import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="SOC Dashboard - Détection d'anomalies",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CSS personnalisé (design sobre et professionnel)
# ------------------------------------------------------------
st.markdown("""
<style>
    /* Variables */
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --accent-blue: #38bdf8;
        --accent-red: #ef4444;
        --accent-orange: #f97316;
        --accent-green: #10b981;
        --border-color: #334155;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: var(--font-sans);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }

    h1, h2, h3 {
        color: var(--accent-blue) !important;
        font-weight: 500;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.25rem;
        margin-top: 1rem;
    }

    h4 {
        color: var(--text-primary) !important;
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.2rem;
    }

    .stMarkdown {
        color: var(--text-secondary);
    }

    /* Rendre les étiquettes des contrôles de la barre latérale plus visibles */
    section[data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Spécifique aux types de widgets si nécessaire */
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: var(--text-primary) !important;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Cartes de métriques */
    .metric-card {
        background-color: var(--bg-card);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        border: 1px solid var(--border-color);
        text-align: center;
    }
    .metric-card .label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 600;
        margin: 0.25rem 0;
    }
    .metric-card .delta {
        color: var(--accent-blue);
        font-size: 0.55rem;
    }

    /* Tableau */
    .dataframe {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }
    .dataframe th {
        background-color: var(--bg-secondary) !important;
        color: var(--accent-blue) !important;
        font-weight: 500;
        padding: 0.5rem;
    }
    .dataframe td {
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border-color);
        padding: 0.5rem;
    }

    /* Boutons */
    .stButton button {
        background-color: var(--accent-blue);
        color: var(--bg-primary);
        font-weight: 500;
        border: none;
        border-radius: 4px;
        padding: 0.4rem 1rem;
        transition: background-color 0.2s;
    }
    .stButton button:hover {
        background-color: #0ea5e9;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.8rem;
        margin-top: 2rem;
        border-top: 1px solid var(--border-color);
        padding-top: 1rem;
    }

    /* Graphiques Plotly - arrière-plan transparent */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    .plot-container {
        background-color: transparent !important;
    }

    /* En-tête personnalisé */
    .header {
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-green) 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: var(--bg-primary);
        font-size: 1.4rem;
        font-weight: 600;
        white-space: nowrap;            /* ensure single line */
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Cartes métriques dimension uniforme */
    .metric-card {
        background-color: var(--bg-card);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        border: 1px solid var(--border-color);
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: transform 0.2s;
        min-height: 120px;             /* force same height */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Amélioration des tableaux */
    .dataframe tbody tr:hover {
        background-color: rgba(255,255,255,0.1) !important;
    }
    .dataframe th {
        background-color: var(--bg-secondary) !important;
        color: var(--accent-blue) !important;
        font-weight: 600;
        padding: 0.5rem;
        font-size: 1rem;
        text-transform: uppercase;
    }
    .dataframe td {
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border-color);
        padding: 0.5rem;
        font-size: 0.9rem;
    }
    .dataframe thead th {
        position: sticky;
        top: 0;
        z-index: 2;
    }

    /* Effet sur les cartes métriques */
    .metric-card {
        background-color: var(--bg-card);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        border: 1px solid var(--border-color);
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }

    /* Style des boutons amélioré */
    .stButton button {
        background-color: var(--accent-blue);
        color: var(--bg-primary);
        font-weight: 500;
        border: none;
        border-radius: 4px;
        padding: 0.4rem 1rem;
        transition: background-color 0.2s, transform 0.1s;
    }
    .stButton button:hover {
        background-color: #0ea5e9;
        transform: scale(1.02);
    }

    /* Lignes alternées pour les tableaux */
    .dataframe tbody tr:nth-child(odd) {
        background-color: #1e293b;
    }
    .dataframe tbody tr:nth-child(even) {
        background-color: #161e2b;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Titre de l'application
# ------------------------------------------------------------

st.markdown("""
<div class="header">
    <span>SOC Industriel - Tableau de bord de détection d'anomalies</span>
</div>
""", unsafe_allow_html=True)
st.markdown("Analyse des données de capteurs et génération d'alertes pour le SOC.")

# ------------------------------------------------------------
# Chargement des données (fichier local)
# ------------------------------------------------------------
CSV_PATH = "SOC_IoT_Dataset_Attacks.csv"  # À adapter si nécessaire

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
        return df
    except FileNotFoundError:
        st.error(f"Fichier {CSV_PATH} introuvable. Veuillez placer le fichier dans le même répertoire que l'application.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()

df_raw = load_data()
st.sidebar.success(f"Données chargées : {len(df_raw)} lignes")

# ------------------------------------------------------------
# Sidebar - Paramètres de détection (lisibilité améliorée)
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Paramètres de détection")
    
    st.markdown("#### Seuils")
    z_threshold = st.slider("**Seuil Z-score**", 2.0, 5.0, 3.0, 0.1, help="Plus le seuil est bas, plus il y a d'alertes.")
    contamination = st.slider("**Contamination (Isolation Forest)**", 0.01, 0.2, 0.05, 0.01, help="Proportion attendue d'anomalies.")
    window = st.slider("**Fenêtre glissante (points)**", 5, 50, 10, 1, help="Taille de la fenêtre pour les statistiques glissantes.")
    
    st.markdown("#### Référence normale")
    norm_method = st.radio(
        "**Méthode**",
        ["Utiliser label=0 (si disponible)", "Premiers N points", "Tout le jeu (robuste)"],
        help="Définit la base de comparaison pour la détection."
    )
    
    if norm_method == "Premiers N points":
        n_first = st.number_input("**Nombre de premiers points**", min_value=10, value=100, step=10)
    
    st.markdown("---")
    st.markdown("Les alertes sont générées par union du Z-score et d'Isolation Forest.")

# ------------------------------------------------------------
# Prétraitement des données
# ------------------------------------------------------------
df = df_raw.copy()
df = df.sort_values('timestamp').reset_index(drop=True)

# Features glissantes
df['temp_roll_mean'] = df['temp_c'].rolling(window, min_periods=1).mean()
df['temp_roll_std'] = df['temp_c'].rolling(window, min_periods=1).std()
df['vib_roll_mean'] = df['vibration_g'].rolling(window, min_periods=1).mean()
df['vib_roll_std'] = df['vibration_g'].rolling(window, min_periods=1).std()
df['energy_roll_mean'] = df['energy_kw'].rolling(window, min_periods=1).mean()
df['energy_roll_std'] = df['energy_kw'].rolling(window, min_periods=1).std()

# Différences
df['temp_diff'] = df['temp_c'].diff().fillna(0)
df['vib_diff'] = df['vibration_g'].diff().fillna(0)
df['energy_diff'] = df['energy_kw'].diff().fillna(0)

# ------------------------------------------------------------
# Définition de la référence normale
# ------------------------------------------------------------
if norm_method == "Utiliser label=0 (si disponible)":
    if 'label' in df.columns:
        normal_data = df[df['label'] == 0]
        if len(normal_data) == 0:
            st.sidebar.warning("Aucune donnée avec label=0, utilisation de tous les points.")
            normal_data = df
    else:
        st.sidebar.warning("Colonne 'label' absente, utilisation de tous les points.")
        normal_data = df
elif norm_method == "Premiers N points":
    normal_data = df.iloc[:n_first]
else:  # Tout le jeu (robuste)
    # On utilisera la médiane et MAD
    pass

# Calcul des statistiques normales
if norm_method != "Tout le jeu (robuste)":
    mean_norm = normal_data[['temp_c','vibration_g','energy_kw']].mean()
    std_norm = normal_data[['temp_c','vibration_g','energy_kw']].std()
else:
    # Approche robuste : médiane et MAD (Median Absolute Deviation)
    median_norm = df[['temp_c','vibration_g','energy_kw']].median()
    mad_norm = (df[['temp_c','vibration_g','energy_kw']] - median_norm).abs().median()
    # Conversion MAD -> écart-type (approx. pour distribution normale)
    std_norm = mad_norm * 1.4826
    mean_norm = median_norm

# ------------------------------------------------------------
# Détection d'anomalies
# ------------------------------------------------------------
# Z-score
z_temp = (df['temp_c'] - mean_norm['temp_c']) / std_norm['temp_c']
z_vib = (df['vibration_g'] - mean_norm['vibration_g']) / std_norm['vibration_g']
z_energy = (df['energy_kw'] - mean_norm['energy_kw']) / std_norm['energy_kw']
z_max = np.maximum(np.abs(z_temp), np.abs(z_vib))
z_max = np.maximum(z_max, np.abs(z_energy))
df['z_max'] = z_max
df['anomaly_z'] = z_max > z_threshold

# Isolation Forest
features = ['temp_c','vibration_g','energy_kw',
            'temp_roll_mean','vib_roll_mean','energy_roll_mean',
            'temp_roll_std','vib_roll_std','energy_roll_std',
            'temp_diff','vib_diff','energy_diff']

df_clean = df.dropna(subset=features).copy()
if len(df_clean) > 10:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df_clean['anomaly_if'] = iso_forest.fit_predict(X_scaled) == -1
    df['anomaly_if'] = False
    df.loc[df_clean.index, 'anomaly_if'] = df_clean['anomaly_if']
else:
    df['anomaly_if'] = False
    st.sidebar.warning("Pas assez de données pour Isolation Forest")

# Combinaison
df['alert'] = df['anomaly_z'] | df['anomaly_if']

# Score de priorité (basé sur z_max)
df['priority_score'] = df['z_max'].fillna(0)

# Catégories de priorité
conditions = [
    (df['priority_score'] < z_threshold),
    (df['priority_score'] >= z_threshold) & (df['priority_score'] < z_threshold + 2),
    (df['priority_score'] >= z_threshold + 2)
]
choices = ['Low', 'Medium', 'High']
df['priority'] = np.select(conditions, choices, default='Low')

# Tableau des alertes
alerts = df[df['alert']].copy()
if not alerts.empty:
    alerts = alerts.reset_index()
    alert_cols = ['timestamp', 'temp_c', 'vibration_g', 'energy_kw',
                  'log_level', 'event', 'label', 'attack_type',
                  'priority', 'priority_score', 'anomaly_z', 'anomaly_if']
    alert_cols = [c for c in alert_cols if c in alerts.columns]
    alert_table = alerts[alert_cols].sort_values('timestamp')
else:
    alert_table = pd.DataFrame()

# ------------------------------------------------------------
# Affichage des métriques
# ------------------------------------------------------------
st.markdown("### Vue d'ensemble")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Total points</div>
        <div class="value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    alert_count = len(alert_table)
    pct = (alert_count / len(df) * 100) if len(df) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Alertes détectées</div>
        <div class="value">{alert_count}</div>
        <div class="delta">{pct:.1f}% du total</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    high_count = (alert_table['priority'] == 'High').sum() if not alert_table.empty else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Priorité Haute</div>
        <div class="value">{high_count}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    medium_count = (alert_table['priority'] == 'Medium').sum() if not alert_table.empty else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Priorité Moyenne</div>
        <div class="value">{medium_count}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# Filtres interactifs
# ------------------------------------------------------------

st.markdown("### Filtres")

if not alert_table.empty:
    with st.expander("Afficher/masquer les filtres", expanded=True):
        colf1, colf2, colf3, colf4 = st.columns(4)
        with colf1:
            min_date = alert_table['timestamp'].min().date()
            max_date = alert_table['timestamp'].max().date()
            start_date = st.date_input("Date début", min_date, key="start")
            end_date = st.date_input("Date fin", max_date, key="end")
        with colf2:
            prio_options = ["All"] + list(alert_table['priority'].unique())
            priorities = st.selectbox("Priorité", options=prio_options,
                                      index=0, key="prio")
        with colf3:
            atk_options = ["All"] + [x for x in alert_table['attack_type'].unique() if pd.notna(x)]
            attack_types = st.selectbox("Type d'attaque", 
                                        options=atk_options,
                                        index=0, key="attack")
        with colf4:
            log_options = ["All"] + [x for x in alert_table['log_level'].unique() if pd.notna(x)]
            log_levels = st.selectbox("Log level",
                                      options=log_options,
                                      index=0, key="log")

    # build mask respecting 'All' selections
    mask = (
        (alert_table['timestamp'].dt.date >= start_date) &
        (alert_table['timestamp'].dt.date <= end_date)
    )
    if priorities != "All":
        mask &= (alert_table['priority'] == priorities)
    if attack_types != "All":
        mask &= (alert_table['attack_type'] == attack_types)
    if log_levels != "All":
        mask &= (alert_table['log_level'] == log_levels)
    filtered_alerts = alert_table[mask]
    st.markdown(f"**{len(filtered_alerts)} alertes affichées**")
else:
    st.info("Aucune alerte détectée avec les paramètres actuels.")
    filtered_alerts = pd.DataFrame()

# ------------------------------------------------------------
# Tableau des alertes
# ------------------------------------------------------------
st.markdown("### Liste des alertes")

if not filtered_alerts.empty:
    # Style conditionnel sur la priorité
    def color_priority(val):
        if val == 'High':
            return 'background-color: rgba(239, 68, 68, 0.2); color: #fecaca;'
        elif val == 'Medium':
            return 'background-color: rgba(249, 115, 22, 0.2); color: #fed7aa;'
        elif val == 'Low':
            return 'background-color: rgba(16, 185, 129, 0.2); color: #a7f3d0;'
        return ''

    styled = filtered_alerts.style.applymap(color_priority, subset=['priority'])
    st.dataframe(styled, use_container_width=True, height=400)

    # Téléchargement CSV
    csv = filtered_alerts.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les alertes filtrées (CSV)",
        data=csv,
        file_name=f"alertes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.write("Aucune alerte à afficher.")

# ------------------------------------------------------------
# Graphiques
# ------------------------------------------------------------
st.markdown("### Analyses graphiques")

if not filtered_alerts.empty:
    colg1, colg2 = st.columns(2)

    with colg1:
        fig1 = px.scatter(filtered_alerts, x='timestamp', y='priority_score',
                          color='priority', symbol='attack_type',
                          hover_data=['temp_c', 'vibration_g', 'energy_kw', 'event'],
                          title="Alertes par priorité et type",
                          color_discrete_map={'High':'#ef4444', 'Medium':'#f97316', 'Low':'#10b981'})
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),  # Texte général
            title_font=dict(color='#f1f5f9'),  # Titre
            xaxis=dict(
                gridcolor='#334155',
                title_font=dict(color='#f1f5f9'),
                tickfont=dict(color='#f1f5f9')
            ),
            yaxis=dict(
                gridcolor='#334155',
                title_font=dict(color='#f1f5f9'),
                tickfont=dict(color='#f1f5f9')
            ),
            legend=dict(
                font=dict(color='#f1f5f9'),
                title_font=dict(color='#f1f5f9')
            )
        )
        st.plotly_chart(fig1, use_container_width=True)

    with colg2:
        attack_counts = filtered_alerts['attack_type'].value_counts().reset_index()
        attack_counts.columns = ['attack_type', 'count']
        fig2 = px.bar(attack_counts, x='attack_type', y='count', color='attack_type',
                      title="Nombre d'alertes par type",
                      color_discrete_sequence=px.colors.sequential.Blues)
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            title_font=dict(color='#f1f5f9'),
            xaxis=dict(
                gridcolor='#334155',
                title_font=dict(color='#f1f5f9'),
                tickfont=dict(color='#f1f5f9')
            ),
            yaxis=dict(
                gridcolor='#334155',
                title_font=dict(color='#f1f5f9'),
                tickfont=dict(color='#f1f5f9')
            ),
            legend=dict(
                font=dict(color='#f1f5f9'),
                title_font=dict(color='#f1f5f9')
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Histogramme des scores
    fig3 = px.histogram(filtered_alerts, x='priority_score', nbins=20,
                        title="Distribution des scores de priorité",
                        color_discrete_sequence=['#38bdf8'])
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9'),
        title_font=dict(color='#f1f5f9'),
        xaxis=dict(
            gridcolor='#334155',
            title_font=dict(color='#f1f5f9'),
            tickfont=dict(color='#f1f5f9')
        ),
        yaxis=dict(
            gridcolor='#334155',
            title_font=dict(color='#f1f5f9'),
            tickfont=dict(color='#f1f5f9')
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Ajustez les paramètres pour générer des alertes et visualiser les graphiques.")

# ------------------------------------------------------------
# Pied de page
# ------------------------------------------------------------
st.markdown("""
<div class="footer">
    SOC Dashboard v3.0 | Détection temps réel | Propulsé par Streamlit, Pandas, Scikit-learn
</div>
""", unsafe_allow_html=True)