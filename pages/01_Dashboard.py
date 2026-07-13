"""Page Dashboard - Vue synthétique de l'application."""
import streamlit as st
import pandas as pd
from datetime import date
from database import execute_query
from engine.curves import get_curve_for_date

st.title("🏠 Dashboard ALM")
st.markdown("Vue d'ensemble des données et simulations disponibles.")
