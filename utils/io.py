"""Import et export de fichiers plats (CSV, Excel).

Responsabilités :
- Lecture des fichiers de taux marché / ZC / chocs
- Mapping automatique des colonnes
- Export des résultats de simulation au format Excel structuré
"""
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import streamlit as st


def read_rates_file(file_path: Path) -> pd.DataFrame:
    """Lit un fichier de taux (CSV ou Excel) et standardise les colonnes.

    Colonnes attendues après mapping : [date, instrument_type, tenor_months, rate_value]
    """
    # TODO: détection du format, mapping flexible des en-têtes
    pass


def read_products_file(file_path: Path) -> pd.DataFrame:
    """Lit un fichier de produits et standardise les colonnes.

    Colonnes attendues : dépendent du product_type (loan, swap, cap, floor, swaption)
    """
    # TODO: parsing avec validation des champs obligatoires
    pass


def read_shock_file(file_path: Path) -> pd.DataFrame:
    """Lit un fichier de chocs par maturité.

    Colonnes attendues : [tenor_months, shock_value] ou [tenor_years, shock_value]
    """
    # TODO: conversion automatique des maturités en mois si nécessaire
    pass


def export_simulation_to_excel(
    simulation_id: int,
    output_path: Path,
    liquidity_gap_df: pd.DataFrame,
    value_sens_df: pd.DataFrame,
    income_sens_df: pd.DataFrame,
    metadata: Dict
) -> None:
    """Exporte les résultats d'une simulation dans un fichier Excel structuré.

    Onglets générés :
    - Paramètres
    - Impasse_Liquidite
    - Sensibilite_Valeur
    - Sensibilite_Revenus
    - Cashflows_Detail (optionnel)
    """
    # TODO: utiliser pandas.ExcelWriter avec openpyxl, mise en forme optionnelle
    pass
