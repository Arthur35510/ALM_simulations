"""Point d'entrée principal de l'application ALM."""
import streamlit as st
from database import init_database

# Configuration de la page
st.set_page_config(
    page_title="Pricing CI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de la base au premier lancement
init_database()

# Navigation entre les pages
#pages = {
#    "🏠 Dashboard": st.Page("pages/01_Dashboard.py", title="Dashboard"),
#    "📈 Courbes de taux": st.Page("pages/02_Courbes.py", title="Courbes de taux"),
#    "⚡ Scénarios de choc": st.Page("pages/03_Scenarios.py", title="Scénarios de choc"),
#    "📂 Portefeuilles": st.Page("pages/04_Portefeuilles.py", title="Portefeuilles"),
#    "🔬 Simulation ALM": st.Page("pages/05_Simulation.py", title="Simulation ALM"),
#}

pages = [
    st.Page("pages/01_Dashboard.py", title="Dashboard", icon="🏠"),
    st.Page("pages/02_Courbes.py", title="Courbes de taux", icon="📈"),
    st.Page("pages/03_Contrats_CI.py", title="Contrats Crédits Immobiliers", icon="📂"),
    st.Page("pages/04_CI_valo.py", title="Valorisation Crédits Immobiliers", icon="🔬")
    #st.Page("pages/05_Simulation.py", title="Simulation ALM", icon="🔬")
]

pg = st.navigation(pages)
pg.run()
