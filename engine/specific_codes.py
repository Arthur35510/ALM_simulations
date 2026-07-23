import numpy as np
import pandas as pd

def model_rarn(
    df_in: pd.DataFrame,
    mode_ci: str,
    duree_mois_ci: int,
    taux_ra_annuel: float = 0.02,
    marge_rn: float = 0.003,
    delta_rn: float = 0.02,
    dt: float = 1/12,
    seed: int = 51
) -> pd.DataFrame:
    
    df_out = df_in.sort_values(["scenario","horizon_mois"])

    # Modèle de RA / RN
    df_out["crd_ra"] = df_out["crd"]
    df_out["crd_rn"] = df_out["crd"]
    df_out["crd_rarn"] = df_out["crd"]

    df_out["taux_survie"] = (1 - taux_ra_annuel) ** dt

    rng = np.random.default_rng(seed)
    df_out["alea_rn"] = rng.random(df_out.shape[0])
    df_out["proba_rn"] = 0
    df_out["top_rn"] = 0
    df_out["taux_rn"] = df_out.taux_client
    df_out["ech_rn"] = (
        df_out.loc[df_out.horizon_mois==0, "crd"].iloc[0] *
        df_out.taux_client * dt /
        (1 - np.power(1 + df_out.taux_client * dt, -duree_mois_ci))
    )
    df_out["ech_rarn"] = df_out["ech_rn"]

    for i_hor in range(1, df_out.horizon_mois.max()):
        print(i_hor)

        # Modèle de RA constant
        df_out["crd_ra"] = (df_out["crd_ra"] * df_out["taux_survie"]).where(df_out.horizon_mois >= i_hor, df_out["crd_ra"])
        df_out["crd_rarn"] = (df_out["crd_rarn"] * df_out["taux_survie"]).where(df_out.horizon_mois >= i_hor, df_out["crd_rarn"])

        # Identification des cas de RN
        df_out["proba_rn"] = np.minimum(
            np.maximum(
                ((df_out.taux_rn - df_out.taux_forward) - marge_rn) / delta_rn,
                np.zeros(df_out.shape[0])),
            np.ones(df_out.shape[0])
        ).where(df_out.horizon_mois == i_hor, df_out.proba_rn)

        df_out["top_rn"] = pd.Series([1] * df_out.shape[0]).where(
            (df_out.horizon_mois == i_hor) &
            (df_out.proba_rn > df_out.alea_rn) &
            (df_out.crd_rn > 0),
            df_out.top_rn
        )
        l_scen_rn = pd.unique(df_out.loc[(df_out.horizon_mois == i_hor) & (df_out.top_rn == 1), "scenario"])

        # Mise à jour des taux client
        df_out["taux_rn"] = np.minimum(df_out["taux_rn"], df_out["taux_forward"] + marge_rn).where(
            (df_out.horizon_mois == i_hor) &
            (df_out.top_rn == 1),
            df_out.taux_rn
        )
        df_out["taux_rn"] = df_out.groupby(["scenario"]).taux_rn.cummin()

        df_out["taux_rn_capi"] = (
            np.power(1 + df_out.taux_rn * dt, df_out.horizon_mois-(i_hor+1))
            .where(df_out.horizon_mois>i_hor, 0.0)
        )
        df_out["taux_rn_capi"] = df_out.groupby("scenario").taux_rn_capi.cumsum()

        # Mise à jour des CRD (si échéance fixe)
        if mode_ci == "Echéance fixe":
            cond_rn = (
                (df_out.scenario.isin(l_scen_rn)) &
                (df_out.horizon_mois > i_hor) &
                (df_out.horizon_mois <= duree_mois_ci)
            )

            df_out = (
                df_out
                .merge(
                    df_out
                    .loc[df_out.horizon_mois==i_hor, ["scenario","crd_rn","crd_rarn"]]
                    .rename(columns={"crd_rn":"crd_tmp_rn", "crd_rarn":"crd_tmp_rarn"})
                    .reset_index(drop=True),
                    how="left", on="scenario"
                )
            )

            for i_type in ["rn","rarn"]:
                df_out[f"ech_{i_type}"] = (
                    df_out[f"crd_tmp_{i_type}"] *
                    df_out.taux_rn * dt /
                    (1 - np.power(1 + df_out.taux_rn * dt, -(duree_mois_ci - i_hor)))
                ).where(cond_rn, df_out[f"ech_{i_type}"])

                df_out[f"crd_{i_type}"] = (
                    df_out[f"crd_tmp_{i_type}"] *
                    np.power(1 + df_out.taux_rn * dt, df_out.horizon_mois - i_hor) -
                    df_out[f"ech_{i_type}"] * df_out.taux_rn_capi
                ).where(cond_rn, df_out[f"crd_{i_type}"])

            df_out = df_out.drop(columns=["crd_tmp_rn", "crd_tmp_rarn"])

    """
    df_out["proba_rn"] = np.minimum(
        np.maximum(
            ((df_out.taux_client - df_out.taux_forward) - marge_rn) / delta_rn,
            np.zeros(df_out.shape[0])),
        np.ones(df_out.shape[0])
    )
    df_out["taux_rn"] = np.minimum(df_out["taux_client"], df_out["taux_forward"] + marge_rn).where(
        (df_out.proba_rn > df_out.alea_rn) & (df_out.horizon_mois > 0), df_out.taux_client
    )

    df_out["taux_forward_min"] = df_out.groupby(["scenario"]).taux_forward.cummin()
    df_out["taux_rn"] = np.minimum(df_out["taux_client"], df_out["taux_forward_min"] + marge_rn)
    """

    return df_out.drop(columns=["taux_survie", "alea_rn", "proba_rn", "top_rn", "ech_rn", "ech_rarn", "taux_rn_capi"])