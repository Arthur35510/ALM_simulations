"""
Fonctions de calcul et de persistance pour la gestion de crédits immobiliers.
 
Ce module contient :
- la construction de l'échéancier contractuel selon 3 modes d'amortissement
- les fonctions d'accès à une base SQLite pour sauvegarder / lister les crédits
 
Convention retenue : taux mensuel = taux annuel / 12 (taux proportionnel,
convention la plus courante pour les crédits immobiliers en France).
"""
 
import sqlite3
from datetime import datetime
 
import pandas as pd

from database import get_connection, execute_query
from config import DB_PATH, CI_AMORTISSEMENT

# ---------------------------------------------------------------------------
# Base de données
# ---------------------------------------------------------------------------

def save_credit(
    nom: str,
    nominal: float,
    taux: float,
    mode: str,
    duree_annees: int
) -> int:
    """
    Sauvegarde un crédit dans la base et retourne son id.
 
    Parameters
    ----------
    nom : libellé libre du crédit (ex: "Résidence principale")
    nominal : montant emprunté en euros
    taux : taux d'intérêt annuel en décimal (ex: 0.035 pour 3.5%)
    mode : un des CI_AMORTISSEMENT
    duree_annees : durée du crédit en années
    """
    if mode not in CI_AMORTISSEMENT:
        raise ValueError(f"mode doit être l'un de {CI_AMORTISSEMENT}")
 
    conn = get_connection()
    conn.execute(
        """DELETE FROM credits_immo 
           WHERE mode = ? AND duree_annees = ? AND taux = ? AND nominal = ?""",
        (mode, duree_annees, taux, nominal)
    )
    cur = conn.execute(
        """
        INSERT INTO credits_immo (nom, mode, duree_annees, taux, nominal, date_creation)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (nom, mode, duree_annees, taux, nominal, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    credit_id = cur.lastrowid
    conn.close()

    return credit_id
 
 
def load_credits() -> pd.DataFrame:
    """Retourne l'ensemble des crédits enregistrés, triés du plus récent au plus ancien."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM credits_immo ORDER BY id DESC", conn)
    conn.close()
    return df
 
 
def load_credit_by_id(credit_id: int) -> dict:
    """Retourne les caractéristiques d'un crédit sous forme de dict."""
    conn = get_connection()
    cur = conn.execute("SELECT * FROM credits_immo WHERE id = ?", (credit_id,))
    row = cur.fetchone()
    columns = [c[0] for c in cur.description]
    conn.close()
    if row is None:
        raise ValueError(f"Aucun crédit trouvé avec l'id {credit_id}")
    return dict(zip(columns, row))
 
 
# ---------------------------------------------------------------------------
# Calcul de l'échéancier
# ---------------------------------------------------------------------------
 
def build_amortization_schedule(
    nominal: float,
    taux_annuel: float,
    mode: str,
    duree_annees: int,
) -> pd.DataFrame:
    """
    Construit l'échéancier contractuel mensuel d'un crédit à taux fixe.
 
    Parameters
    ----------
    nominal : montant emprunté en euros
    taux_annuel : taux d'intérêt annuel en décimal (ex: 0.035 pour 3.5%)
    mode : 'Echéance fixe', 'In fine' ou 'Amortissement constant'
    duree_annees : durée du crédit en années
 
    Returns
    -------
    pd.DataFrame avec les colonnes :
        mois, capital_restant_du_debut, interets, principal_rembourse,
        echeance, capital_restant_du_fin
    """
    if mode not in CI_AMORTISSEMENT:
        raise ValueError(f"mode doit être l'un de {CI_AMORTISSEMENT}")
 
    n = int(duree_annees * 12)
    i = taux_annuel / 12.0
 
    rows = []
    crd = nominal
 
    if mode == "Echéance fixe":
        if i == 0:
            echeance_fixe = nominal / n
        else:
            echeance_fixe = nominal * i / (1 - (1 + i) ** (-n))
 
        for mois in range(1, n + 1):
            interets = crd * i
            if mois == n:
                principal = crd
                echeance = principal + interets
            else:
                principal = echeance_fixe - interets
                echeance = echeance_fixe
            crd_fin = crd - principal
            rows.append((mois, crd, interets, principal, echeance, crd_fin))
            crd = crd_fin
 
    elif mode == "In fine":
        for mois in range(1, n + 1):
            interets = crd * i
            principal = crd if mois == n else 0.0
            echeance = principal + interets
            crd_fin = crd - principal
            rows.append((mois, crd, interets, principal, echeance, crd_fin))
            crd = crd_fin
 
    elif mode == "Amortissement constant":
        principal_constant = nominal / n
        for mois in range(1, n + 1):
            interets = crd * i
            principal = principal_constant if mois < n else crd
            echeance = principal + interets
            crd_fin = crd - principal
            rows.append((mois, crd, interets, principal, echeance, crd_fin))
            crd = crd_fin
 
    schedule = pd.DataFrame(
        rows,
        columns=[
            "mois",
            "capital_restant_du_debut",
            "interets",
            "principal_rembourse",
            "echeance",
            "capital_restant_du_fin",
        ],
    )

    return schedule
 
def build_schedule_for_credit(credit_row: dict) -> pd.DataFrame:
    """Reconstruit l'échéancier d'un crédit à partir de ses caractéristiques stockées en base."""
    return build_amortization_schedule(
        nominal=credit_row["nominal"],
        taux_annuel=credit_row["taux"],
        mode=credit_row["mode"],
        duree_annees=credit_row["duree_annees"],
    )


def save_schedule(
    ci_id: int,
    nominal: float,
    df_schedule: pd.DataFrame
) -> int:
    """
    Sauvegarde un écoulement contractuel dans la base.
 
    Parameters
    ----------
    ci_id : id du crédit
    nominal : montant emprunté en euros
    schedule : dataframe contenant l'écoulement
    """
 
    conn = get_connection()
    conn.execute(
        """DELETE FROM ci_ecoulements 
           WHERE ci_id = ?""",
        (ci_id,)
    )

    conn.execute(
        """INSERT INTO ci_ecoulements 
            (ci_id, t_months, crd)
            VALUES (?, ?, ?)""",
        (ci_id,0,nominal)
    )
    
    for _, row in df_schedule.iterrows():
        conn.execute(
            """INSERT INTO ci_ecoulements 
                (ci_id, t_months, crd)
                VALUES (?, ?, ?)""",
            (
                ci_id,
                row["mois"],
                float(row["capital_restant_du_fin"])
            )
        )

    conn.commit()
    conn.close()

    return 1

def load_schedule(ci_id: int) -> pd.DataFrame:
    """Retourne l'écoulement contractuel d'un crédit."""
    conn = get_connection()
    df = pd.read_sql_query(
        f"""
            SELECT *
            FROM ci_ecoulements
            WHERE ci_id = {ci_id}
            ORDER BY t_months
        """,
        conn
    )
    conn.close()
    return df
