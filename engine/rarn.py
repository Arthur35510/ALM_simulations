import numpy as np
import pandas as pd

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