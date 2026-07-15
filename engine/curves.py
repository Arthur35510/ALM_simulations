"""Construction et gestion des courbes de taux.

Responsabilités :
- Chargement des taux marché (Euribor + Swaps)
- Bootstrapping classique des taux zéro-coupon
- Interpolation spline cubique sur maturités mensuelles
- Calcul des taux forward (instantanés et tenor)
- Stockage et récupération depuis SQLite avec logique de fallback
"""
import sqlite3
from datetime import date
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator

from database import get_connection, execute_query
from config import ZC_TENORS, EURIBOR_TENORS, SWAP_TENORS, FWD_HORIZONS


# =============================================================================
# 1. CHARGEMENT DEPUIS LA BASE
# =============================================================================

def load_market_rates(curve_date: date) -> pd.DataFrame:
    """Charge les taux marché pour une date donnée depuis la base.

    Returns:
        DataFrame avec colonnes [instrument_type, tenor_months, rate_value]
    """
    query = """
        SELECT mr.instrument_type, mr.tenor_months, mr.rate_value
        FROM market_rates mr
        JOIN curve_dates cd ON mr.curve_date_id = cd.id
        WHERE cd.curve_date = ?
        ORDER BY mr.tenor_months
    """
    rows = execute_query(query, (curve_date.isoformat(),))
    if not rows:
        return pd.DataFrame(columns=["instrument_type", "tenor_months", "rate_value"])
    return pd.DataFrame(rows,columns=["instrument_type", "tenor_months", "rate_value"])


def load_zero_coupons(curve_date: date) -> pd.DataFrame:
    """Charge les taux zéro-coupon pour une date donnée.

    Returns:
        DataFrame avec colonnes [tenor_months, rate_value]
    """
    query = """
        SELECT zc.tenor_months, zc.rate_value
        FROM zero_coupons zc
        JOIN curve_dates cd ON zc.curve_date_id = cd.id
        WHERE cd.curve_date = ?
        ORDER BY zc.tenor_months
    """
    rows = execute_query(query, (curve_date.isoformat(),))

    if not rows:
        return pd.DataFrame(columns=["tenor_months", "rate_value"])
    return pd.DataFrame(rows,columns=["tenor_months", "rate_value"])


def load_forward_rates(curve_date: date) -> pd.DataFrame:
    """Charge les taux forward pour une date donnée.

    Returns:
        DataFrame avec colonnes [forward_months, tenor_months, rate_value]
    """
    query = """
        SELECT fr.forward_month, fr.tenor_month, fr.rate_value
        FROM forward_rates fr
        JOIN curve_dates cd ON fr.curve_date_id = cd.id
        WHERE cd.curve_date = ?
        ORDER BY fr.forward_month, fr.tenor_month
    """
    rows = execute_query(query, (curve_date.isoformat(),))

    if not rows:
        return pd.DataFrame(columns=["forward_month", "tenor_month", "rate_value"])
    return pd.DataFrame(rows,columns=["forward_month", "tenor_month", "rate_value"])


def load_discount_factors(curve_date: date) -> pd.DataFrame:
    """Charge les discount factors pour une date donnée.

    Returns:
        DataFrame avec colonnes [forward_months, factor_value]
    """
    query = """
        SELECT df.forward_month, df.factor_value
        FROM discount_factors df
        JOIN curve_dates cd ON df.curve_date_id = cd.id
        WHERE cd.curve_date = ?
        ORDER BY df.forward_month
    """
    rows = execute_query(query, (curve_date.isoformat(),))

    if not rows:
        return pd.DataFrame(columns=["forward_month", "factor_value"])
    return pd.DataFrame(rows,columns=["forward_month", "factor_value"])


# =============================================================================
# 2. BOOTSTRAPPING CLASSIQUE
# =============================================================================

def _year_frac_from_months(months: int, convention: str = "act/360") -> float:
    """Convertit des mois en fraction d'année selon la convention."""
    if convention == "30/360":
        return months / 12.0
    return months / 12.0


def bootstrap_zero_coupons(market_df: pd.DataFrame) -> pd.DataFrame:
    """Construit les taux zéro-coupon par bootstrapping classique EUR.

    Méthode :
    - Euribor (1M-12M) : discount factor direct par capitalisation simple.
    - Swaps (2Y-30Y) : bootstrapping itératif. Relation swap au par :
      K * sum_{i=1}^{n-1} DF(T_i) + (1 + K) * DF(T_n) = 1
      => DF(T_n) = (1 - K * sum_{i=1}^{n-1} DF(T_i)) / (1 + K)
    - Conversion en taux zéro-coupon continus : r(t) = -ln(DF(t)) / t

    Si des taux marché sont manquants pour certaines maturités attendues,
    une interpolation linéaire est effectuée avant le bootstrapping.

    Args:
        market_df: DataFrame avec colonnes [instrument_type, tenor_months, rate_value]
            instrument_type ∈ {'euribor', 'swap'}

    Returns:
        DataFrame avec colonnes [tenor_months, rate_value, source_origin='calculated']
            rate_value = taux zéro-coupon continu
    """
    if market_df.empty:
        raise ValueError("Aucun taux marché fourni pour le bootstrapping.")

    # =================================================================
    # INTERPOLATION DES TAUX MARCHÉ MANQUANTS
    # =================================================================

    # Séparation Euribor / Swaps
    euribor_raw = market_df[market_df["instrument_type"] == "euribor"].sort_values("tenor_months")
    swaps_raw = market_df[market_df["instrument_type"] == "swap"].sort_values("tenor_months")

    # --- Interpolation Euribor ---
    euribor = _interpolate_market_rates(euribor_raw, EURIBOR_TENORS, "euribor")

    # --- Interpolation Swaps ---
    swaps = _interpolate_market_rates(swaps_raw, SWAP_TENORS, "swap")

    # =================================================================
    # BOOTSTRAPPING
    # =================================================================

    # Dictionnaire tenor (mois) -> discount factor
    df_map: Dict[int, float] = {}

    # --- Euribor : capitalisation simple ACT/360 ---
    for _, row in euribor.iterrows():
        months = int(row["tenor_months"])
        r = float(row["rate_value"])
        tau = _year_frac_from_months(months, "act/360")
        df_map[months] = 1.0 / (1.0 + r * tau)

    # --- Swaps : bootstrapping itératif ---
    # Convention swap EUR annual 30/360 vs 6M Euribor
    for _, row in swaps.iterrows():
        months = int(row["tenor_months"])
        K = float(row["rate_value"])
        n_years = months // 12

        # Somme des DF connus pour les années précédentes
        sum_df_known = 0.0
        for y in range(1, n_years):
            prev_months = y * 12
            if prev_months not in df_map:
                raise ValueError(
                    f"Impossible de bootstrapper le swap {months}M : "
                    f"manque le DF pour {prev_months}M"
                )
            sum_df_known += df_map[prev_months]

        df_n = (1.0 - K * sum_df_known) / (1.0 + K)
        df_map[months] = df_n

    # --- Conversion en taux ZC continus ---
    records = []
    for months in sorted(df_map.keys()):
        df_val = df_map[months]
        tau = _year_frac_from_months(months, "act/360")  # t en années
        if tau <= 0:
            continue
        r_continuous = -np.log(df_val) / tau
        records.append({
            "tenor_months": months,
            "rate_value": r_continuous
        })

    return pd.DataFrame(records)


def _interpolate_market_rates(
    market_df: pd.DataFrame,
    expected_tenors: List[int],
    instrument_type: str
) -> pd.DataFrame:
    """Interpole les taux marché manquants sur les maturités attendues.

    Args:
        market_df: DataFrame avec [tenor_months, rate_value]
        expected_tenors: liste des maturités attendues
        instrument_type: 'euribor' ou 'swap'

    Returns:
        DataFrame complet avec toutes les maturités attendues
    """
    if market_df.empty:
        return pd.DataFrame(columns=["instrument_type", "tenor_months", "rate_value"])

    # Création d'un DataFrame avec toutes les maturités attendues
    full_df = pd.DataFrame({
        "tenor_months": expected_tenors,
        "instrument_type": instrument_type
    })

    # Merge avec les données existantes
    merged = pd.merge(
        full_df,
        market_df[["tenor_months", "rate_value"]],
        on="tenor_months",
        how="left"
    )

    # Vérification des données manquantes
    missing_count = merged["rate_value"].isna().sum()
    if missing_count > 0:
        # Interpolation linéaire sur les taux
        merged["rate_value"] = merged["rate_value"].interpolate(method="linear")

        # Extrapolation constante aux bords
        merged["rate_value"] = merged["rate_value"].ffill().bfill()

        # Avertissement si extrapolation
        if missing_count > 0:
            import warnings
            missing_tenors = merged[merged["rate_value"].isna()]["tenor_months"].tolist()
            if not missing_tenors:  # Après ffill/bfill, il ne devrait plus y en avoir
                warnings.warn(
                    f"{missing_count} taux {instrument_type} interpolés/extrapolés "
                    f"sur les maturités attendues.",
                    RuntimeWarning
                )

    return merged[["instrument_type", "tenor_months", "rate_value"]]


# =============================================================================
# 3. INTERPOLATION SPLINE CUBIQUE
# =============================================================================

def interpolate_zero_coupons(
    zc_df: pd.DataFrame, 
    target_tenors: Optional[List[int]] = None
) -> pd.DataFrame:
    """Interpole les taux ZC continus sur les maturités cibles via spline cubique.

    Args:
        zc_df: DataFrame avec colonnes [tenor_months, rate_value] (taux continus)
        target_tenors: liste de maturités mensuelles souhaitées (défaut 1..360)

    Returns:
        DataFrame avec colonnes [tenor_months, rate_value, source_origin='interpolated']
    """
    if zc_df.empty:
        raise ValueError("Courbe ZC vide, impossible d'interpoler.")

    if target_tenors is None:
        target_tenors = ZC_TENORS.copy()

    # Points connus
    x_known = zc_df["tenor_months"].values / 12.0
    y_known = zc_df["rate_value"].values

    # Spline cubique naturelle
    cs = CubicSpline(x_known, y_known, bc_type="not-a-knot")
    pchip = PchipInterpolator(x_known, y_known)

    # Évaluation
    x_target = np.array(target_tenors) / 12.0
    y_target = pchip(x_target)

    # Avertissement si taux négatifs (spline peut osciller)
    if np.any(y_target < -0.01):
        import warnings
        warnings.warn(
            "Interpolation spline cubique a produit des taux ZC très négatifs. "
            "Vérifiez la qualité des données marché.",
            RuntimeWarning
        )

    records = []
    for months, rate in zip(target_tenors, y_target):
        records.append({
            "tenor_months": months,
            "rate_value": float(rate)
        })

    return pd.DataFrame(records)


# =============================================================================
# 4. TAUX FORWARD
# =============================================================================

def compute_forward_tenor(zc_df: pd.DataFrame, tenor_months: int) -> pd.DataFrame:
    """Calcule les taux forward de tenor donné (taux simple annualisé).

    Formule : F(t, T) = [P(t) / P(t+T) - 1] / T
    où P(t) = exp(-r(t) * t) est le facteur d'actualisation,
    t = forward_month / 12, T = tenor_months / 12.

    Args:
        zc_df: DataFrame avec colonnes [tenor_months, rate_value] (taux continus)
        tenor_months: tenor du forward en mois (ex: 3 pour forward 3M)

    Returns:
        DataFrame avec colonnes [forward_month, tenor_month, rate_value]
    """
    if zc_df.empty:
        raise ValueError("Courbe ZC vide.")

    if tenor_months <= 0:
        raise ValueError("tenor_months doit être > 0. Utilisez compute_forward_instantaneous pour le taux instantané.")

    max_months = int(zc_df["tenor_months"].max())
    records = []

    for fwd_months in FWD_HORIZONS:
        
        if fwd_months==0:
            fwd_rate = float(zc_df.loc[zc_df.tenor_months==tenor_months, "rate_value"].iloc[0])
        elif (fwd_months > 0) & (tenor_months+fwd_months <= max_months):
            fwd_rate = (
                float(zc_df.loc[zc_df.tenor_months==tenor_months+fwd_months, "rate_value"].iloc[0]) *
                (tenor_months+fwd_months) -
                float(zc_df.loc[zc_df.tenor_months==fwd_months, "rate_value"].iloc[0]) *
                fwd_months
            ) / tenor_months

        records.append({
            "forward_month": fwd_months,
            "tenor_month": tenor_months,
            "rate_value": fwd_rate
        })

    return pd.DataFrame(records)


# =============================================================================
# 5. DISCOUNT FACTOR
# =============================================================================

def compute_discount_factor(zc_df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les discount factors pour différents horizons.

    Formule : DF(t) = exp(-ZC(t)*t)

    Args:
        zc_df: DataFrame avec colonnes [tenor_months, rate_value] (taux continus)

    Returns:
        DataFrame avec colonnes [forward_month, factor_value]
    """

    if zc_df.empty:
        raise ValueError("Courbe ZC vide.")
    
    df_df = zc_df.copy()
    df_df.loc[-1] = [0, 0]
    df_df.index = df_df.index + 1
    df_df = df_df.sort_index().rename(columns={"tenor_months":"forward_month", "rate_value":"factor_value"})
    df_df["factor_value"] = np.exp(- df_df["factor_value"] * df_df["forward_month"] / 12)

    return df_df


# =============================================================================
# 6. STOCKAGE DANS LA BASE
# =============================================================================

def _get_or_create_curve_date(conn: sqlite3.Connection, curve_date: date, source_type: str) -> int:
    """Récupère ou crée un enregistrement curve_dates et retourne son ID."""
    cursor = conn.execute(
        "SELECT id FROM curve_dates WHERE curve_date = ? AND source_type = ?",
        (curve_date.isoformat(), source_type)
    )
    row = cursor.fetchone()
    if row:
        return row["id"]

    cursor = conn.execute(
        "INSERT INTO curve_dates (curve_date, source_type) VALUES (?, ?)",
        (curve_date.isoformat(), source_type)
    )
    return cursor.lastrowid


def store_curve_data(
    curve_date: date, 
    source_type: str, 
    data_df: pd.DataFrame, 
    table_name: str
) -> None:
    """Stocke une courbe (marché, ZC ou forward) dans la base.

    Args:
        curve_date: date de la courbe
        source_type: 'market', 'zc', 'forward'
        data_df: DataFrame à stocker
        table_name: 'market_rates', 'zero_coupons' ou 'forward_rates'
    """
    conn = get_connection()
    try:
        curve_date_id = _get_or_create_curve_date(conn, curve_date, source_type)

        if table_name == "market_rates":
            conn.execute(
                """DELETE FROM market_rates 
                   WHERE curve_date_id = ?""",
                (curve_date_id,)
            )
            for _, row in data_df.iterrows():
                conn.execute(
                    """INSERT INTO market_rates 
                       (curve_date_id, instrument_type, tenor_months, rate_value)
                       VALUES (?, ?, ?, ?)""",
                    (
                        curve_date_id,
                        row["instrument_type"],
                        int(row["tenor_months"]),
                        float(row["rate_value"])
                    )
                )

        elif table_name == "zero_coupons":
            conn.execute(
                "DELETE FROM zero_coupons WHERE curve_date_id = ?",
                (curve_date_id,)
            )
            for _, row in data_df.iterrows():
                conn.execute(
                    """INSERT INTO zero_coupons 
                       (curve_date_id, tenor_months, rate_value)
                       VALUES (?, ?, ?)""",
                    (
                        curve_date_id,
                        int(row["tenor_months"]),
                        float(row["rate_value"])
                    )
                )

        elif table_name == "forward_rates":
            conn.execute(
                "DELETE FROM forward_rates WHERE curve_date_id = ?",
                (curve_date_id,)
            )
            for _, row in data_df.iterrows():
                conn.execute(
                    """INSERT INTO forward_rates 
                       (curve_date_id, forward_month, tenor_month, rate_value)
                       VALUES (?, ?, ?, ?)""",
                    (
                        curve_date_id,
                        int(row["forward_month"]),
                        int(row["tenor_month"]),
                        float(row["rate_value"])
                    )
                )
                
        elif table_name == "discount_factors":
            conn.execute(
                "DELETE FROM discount_factors WHERE curve_date_id = ?",
                (curve_date_id,)
            )
            for _, row in data_df.iterrows():
                conn.execute(
                    """INSERT INTO discount_factors 
                       (curve_date_id, forward_month, factor_value)
                       VALUES (?, ?, ?)""",
                    (
                        curve_date_id,
                        int(row["forward_month"]),
                        float(row["factor_value"])
                    )
                )

        else:
            raise ValueError(f"Table {table_name} non reconnue.")

        conn.commit()
    finally:
        conn.close()


# =============================================================================
# 6. RÉCUPÉRATION AVEC FALLBACK
# =============================================================================

def get_curve_for_date(curve_date: date, source_type: str = "zero") -> Optional[pd.DataFrame]:
    """Récupère une courbe complète pour une date et un type donnés.

    Logique de fallback :
    1. Si source_type='zero' : cherche d'abord les ZC stockés (calculated/file/interpolated).
       Si trouvé, retourne directement.
    2. Si pas de ZC, cherche les taux marché pour cette date, bootstrappe + interpole,
       stocke le résultat, et le retourne.
    3. Si pas de taux marché pour cette date, cherche les deux dates les plus proches
       ayant des ZC, et interpole linéairement les taux ZC entre ces deux dates.

    Returns:
        DataFrame avec colonnes [tenor_months, rate_value, source_origin] (pour ZC)
        ou None si impossible.
    """
    # --- Cas 1 : ZC directement disponibles ---
    if source_type in ("zero", "zc_interp"):
        zc_df = load_zero_coupons(curve_date)
        if not zc_df.empty and len(zc_df) == len(ZC_TENORS):
            return zc_df[["tenor_months", "rate_value", "source_origin"]]

    # --- Cas 2 : Bootstrapping depuis taux marché ---
    market_df = load_market_rates(curve_date)
    if not market_df.empty:
        try:
            zc_bootstrapped = bootstrap_zero_coupons(market_df)
            zc_interpolated = interpolate_zero_coupons(zc_bootstrapped, ZC_TENORS)

            # Stockage automatique
            store_curve_data(curve_date, "zero", zc_interpolated, "zero_coupons")

            return zc_interpolated
        except Exception as e:
            import warnings
            warnings.warn(f"Bootstrapping échoué pour {curve_date}: {e}", RuntimeWarning)

    # --- Cas 3 : Interpolation temporelle entre dates voisines ---
    # Recherche des dates les plus proches avec des ZC complets
    query = """
        SELECT DISTINCT cd.curve_date
        FROM curve_dates cd
        JOIN zero_coupons zc ON cd.id = zc.curve_date_id
        WHERE cd.source_type = 'zero'
        GROUP BY cd.curve_date
        HAVING COUNT(DISTINCT zc.tenor_months) >= 300
        ORDER BY cd.curve_date
    """
    available_dates = execute_query(query)
    if not available_dates:
        return None

    available = [date.fromisoformat(r["curve_date"]) for r in available_dates]

    # Trouver les deux dates encadrantes
    before = [d for d in available if d < curve_date]
    after = [d for d in available if d > curve_date]

    if before and after:
        d_before = max(before)
        d_after = min(after)
    elif before and not after:
        # Extrapolation vers l'avant (moins fiable, on prend la plus proche)
        return None
    elif after and not before:
        return None
    else:
        return None

    # Chargement des deux courbes
    zc_before = load_zero_coupons(d_before)
    zc_after = load_zero_coupons(d_after)

    if zc_before.empty or zc_after.empty:
        return None

    # Merge sur tenor_months
    merged = pd.merge(
        zc_before[["tenor_months", "rate_value"]].rename(columns={"rate_value": "r_before"}),
        zc_after[["tenor_months", "rate_value"]].rename(columns={"rate_value": "r_after"}),
        on="tenor_months",
        how="outer"
    ).sort_values("tenor_months")

    # Interpolation linéaire temporelle
    total_days = (d_after - d_before).days
    target_days = (curve_date - d_before).days
    if total_days <= 0:
        return None
    alpha = target_days / total_days

    merged["rate_value"] = merged["r_before"] * (1 - alpha) + merged["r_after"] * alpha
    merged["source_origin"] = "interpolated"

    result = merged[["tenor_months", "rate_value", "source_origin"]].dropna()

    # Stockage du résultat interpolé temporellement
    if not result.empty:
        store_curve_data(curve_date, "zc_interp", result, "zero_coupons")

    return result


# =============================================================================
# 7. FONCTIONS DE CONVERSION UTILITAIRES
# =============================================================================

def zero_coupon_to_discount_factor(zc_df: pd.DataFrame) -> pd.DataFrame:
    """Convertit une courbe ZC (taux continus) en facteurs d'actualisation.

    P(t) = exp(-r(t) * t)
    """
    df = zc_df.copy()
    t = df["tenor_months"] / 12.0
    df["discount_factor"] = np.exp(-df["rate_value"] * t)
    return df


def discount_factor_to_zero_coupon(df_df: pd.DataFrame) -> pd.DataFrame:
    """Convertit des facteurs d'actualisation en taux ZC continus.

    r(t) = -ln(P(t)) / t
    """
    df = df_df.copy()
    t = df["tenor_months"] / 12.0
    # Éviter division par zéro
    t = t.replace(0, np.nan)
    df["rate_value"] = -np.log(df["discount_factor"]) / t
    return df
