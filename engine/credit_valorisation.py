"""
Fonctions de calcul la valeur de crédits immobiliers.

Principe général
-----------------
La valeur du crédit est calculée comme l'espérance, sur l'ensemble des scénarios
du simulateur, de la somme des cash flows futurs (capital + intérêts) actualisés :
 
    Valeur = (1 / n_scenarios) * somme_scenarios [ somme_t ( CF(t, scenario) * DF(t, scenario) ) ]
 
Modélisation des colle mensuelle que sim_params.horizons_months.
"""
 
 
import sqlite3
import numpy as np
import pandas as pd

from typing import List, Union, Dict, Optional
from datetime import date
from database import get_connection, execute_query

def apply_rarn(
    df_in: pd.DataFrame,
    taux_ra_annuel: float = 0.02,
    marge_rn: float = 0.003,
    delta_rn: float = 0.02,
    dt: float = 1/12,
    seed: int = 51
) -> pd.DataFrame:
    
    df_out = df_in.sort_values(["scenario","horizon_mois"])

    # Modèle de RA constant
    df_out["taux_survie"] = (1 - taux_ra_annuel) ** dt
    df_out["crd_ra"] = df_out["crd"] * (df_out["taux_survie"] ** df_out["horizon_mois"])

    # Modèle de renégociation
    rng = np.random.default_rng(seed)
    df_out["alea_rn"] = rng.standard_normal(df_out.shape[0])
    df_out["proba_rn"] = np.minimum(
        np.maximum(
            ((df_out.taux_client - df_out.taux_forward) - marge_rn) / delta_rn,
            np.zeros(df_out.shape[0])),
        np.ones(df_out.shape[0])
    )
    df_out["taux_rn"] = np.minimum(df_out["taux_client"], df_out["taux_forward"] + marge_rn).where(
        (df_out.proba_rn > df_out.alea_rn) & (df_out.horizon_mois > 0), df_out.taux_client
    )

    #df_out["taux_forward_min"] = df_out.groupby(["scenario"]).taux_forward.cummin()
    #df_out["taux_rn"] = np.minimum(df_out["taux_client"], df_out["taux_forward_min"] + marge_rn)

    return df_out.drop(columns=["taux_survie", "alea_rn", "proba_rn"])

def valorisation_ci(
    discount_factors: pd.DataFrame,
    taux_forward: pd.DataFrame,
    ci_ecoulement: pd.DataFrame,
    taux_client: pd.DataFrame,
    dt: float = 1/12
) -> dict:
    
    # Agregat des informations
    df_agg = (
        discount_factors
        .merge(
            taux_forward.loc[taux_forward.tenor_mois==120].reset_index(drop=True),
            how="inner", on=["scenario", "horizon_mois"]
        )
        .merge(
            ci_ecoulement.rename(columns={"t_months":"horizon_mois"}),
            how="inner", on="horizon_mois"
        )
    )

    df_agg["taux_client"] = taux_client

    # Application du modele de rarn
    df_rarn = apply_rarn(df_agg)

    # Patch
    mt_init = float(df_rarn.loc[df_rarn.horizon_mois==0, "crd"].iloc[0])
    # Fin du patch

    # Calcul des CF (nominal et interets)
    df_rarn["cf_nom_ctrl"] = df_rarn.groupby(["scenario"]).crd.shift(1, fill_value=0.0) - df_rarn["crd"]
    df_rarn["cf_nom_ra"] = df_rarn.groupby(["scenario"]).crd_ra.shift(1, fill_value=0.0) - df_rarn["crd_ra"]
    df_rarn["cf_int_ctrl"] = df_rarn.groupby(["scenario"]).crd.shift(1, fill_value=0.0) * (np.exp(df_rarn["taux_client"] * dt) - 1)
    df_rarn["cf_int_ra"] = df_rarn.groupby(["scenario"]).crd_ra.shift(1, fill_value=0.0) * (np.exp(df_rarn["taux_client"] * dt) - 1)
    df_rarn["cf_int_rn"] = df_rarn.groupby(["scenario"]).crd.shift(1, fill_value=0.0) * (np.exp(df_rarn["taux_rn"] * dt) - 1)
    df_rarn["cf_int_rarn"] = df_rarn.groupby(["scenario"]).crd_ra.shift(1, fill_value=0.0) * (np.exp(df_rarn["taux_rn"] * dt) - 1)

    # Calcul des CF actualises
    df_rarn["cfdf_ctrl"] = (df_rarn["cf_nom_ctrl"] + df_rarn["cf_int_ctrl"]) * df_rarn["discount_factor"] / mt_init
    df_rarn["cfdf_ra"] = (df_rarn["cf_nom_ra"] + df_rarn["cf_int_ra"]) * df_rarn["discount_factor"] / mt_init
    df_rarn["cfdf_rn"] = (df_rarn["cf_nom_ctrl"] + df_rarn["cf_int_rn"]) * df_rarn["discount_factor"] / mt_init
    df_rarn["cfdf_rarn"] = (df_rarn["cf_nom_ra"] + df_rarn["cf_int_rarn"]) * df_rarn["discount_factor"] / mt_init

    # Calcul des valeurs
    dict_valo = (
        df_rarn
        .groupby(["scenario"])
        .agg({
            "cfdf_ctrl":"sum",
            "cfdf_ra":"sum",
            "cfdf_rn":"sum",
            "cfdf_rarn":"sum"
        })
        .reset_index()
        .agg({
            "cfdf_ctrl":"mean",
            "cfdf_ra":"mean",
            "cfdf_rn":"mean",
            "cfdf_rarn":"mean"
        })
        .to_dict()
    )

    return df_rarn, dict_valo

def _get_or_create_simul(
    conn: sqlite3.Connection,
    hw_a: float,
    hw_s: float,
    curve_date: date,
    nb_scenarios: int,
    h_max_months: int,
    time_step_months: int
) -> int:

    """Récupère ou crée un enregistrement simulations_config et retourne son ID."""

    cursor = conn.execute(
        """
        SELECT id
        FROM simulations_config
        WHERE
            hw_a = ? AND
            hw_s = ? AND
            curve_date = ? AND
            nb_scenarios = ? AND
            h_max_months = ? AND
            time_step_months = ?
        """,
        (hw_a, hw_s, curve_date.isoformat(), nb_scenarios, h_max_months, time_step_months)
    )
    row = cursor.fetchone()
    if row:
        return row["id"]

    cursor = conn.execute(
        """
        INSERT INTO simulations_config
        (hw_a, hw_s, curve_date, nb_scenarios, h_max_months, time_step_months) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (hw_a, hw_s, curve_date.isoformat(), nb_scenarios, h_max_months, time_step_months)
    )
    return cursor.lastrowid

def store_valorisation(
    ci_id: int,
    hw_a: float,
    hw_s: float,
    curve_date: date,
    nb_scenarios: int,
    h_max_months: int,
    time_step_months: int,
    dict_valo: dict
) -> None:
    
    """Sauvegarde une liste de valo."""

    conn = get_connection()
    simul_id = _get_or_create_simul(conn, hw_a, hw_s, curve_date, nb_scenarios, h_max_months, time_step_months)
    conn.execute(
        """DELETE FROM ci_valorisations 
           WHERE simul_id = ? and ci_id = ?""",
        (simul_id, ci_id)
    )
    cur = conn.execute(
        """INSERT INTO ci_valorisations 
            (simul_id, ci_id, valo_ctrl, valo_ra, valo_rn, valo_rarn)
            VALUES (?, ?, ?, ?, ?, ?)""",
        (
            simul_id,
            ci_id,
            dict_valo["cfdf_ctrl"],
            dict_valo["cfdf_ra"],
            dict_valo["cfdf_rn"],
            dict_valo["cfdf_rarn"],
        )
    )

    conn.commit()
    valo_id = cur.lastrowid
    conn.close()
    return valo_id

def load_simulations(simul_id: Optional[int] = None) -> pd.DataFrame:
    """Retourne l'écoulement contractuel d'un crédit."""
    conn = get_connection()
    if simul_id is None:
        df = pd.read_sql_query(
            f"""
                SELECT *
                FROM simulations_config
            """,
            conn
        )
    else:
        df = pd.read_sql_query(
            f"""
                SELECT *
                FROM simulations_config
                WHERE id = {simul_id}
            """,
            conn
        )
    conn.close()
    return df

def load_valorisations(valo_id: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:

    conn = get_connection()
    if valo_id is None:
        df = pd.read_sql_query(
            f"""
                SELECT *
                FROM ci_valorisations
            """,
            conn
        )
    else:
        if valo_id is int:
            valo_id = [str(valo_id)]
        else:
            valo_id = [str(i_v) for i_v in valo_id]
        valo_id = ", ".join(valo_id)

        df = pd.read_sql_query(
            f"""
                SELECT *
                FROM ci_valorisations
                WHERE id in ({valo_id})
            """,
            conn
        )
    conn.close()
    return df

