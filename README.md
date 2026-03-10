# SOC Dashboard - Détection d'anomalies industrielles

Application Streamlit pour la visualisation et la détection d'anomalies sur des données de capteurs IoT/OT, destinée aux analystes SOC.

## Fonctionnalités

- Chargement automatique d'un fichier CSV local (format spécifié)
- Détection d'anomalies par Z-score et Isolation Forest
- Génération d'alertes avec priorisation (Low, Medium, High)
- Tableau interactif avec filtres par date, priorité, type d'attaque, log level
- Graphiques interactifs (Plotly) avec thème sombre
- Export CSV des alertes filtrées

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/soc-dashboard.git
   cd soc-dashboard
