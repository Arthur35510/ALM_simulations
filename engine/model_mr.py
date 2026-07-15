"""
Simulation de taux de marché par un modèle de Hull & White à 1 facteur (HW1F).

Le modèle s'écrit :
    dr(t) = [theta(t) - a * r(t)] dt + sigma * dW(t)

où :
    a       : vitesse de retour à la moyenne (mean reversion)
    sigma   : volatilité du taux court
    theta(t): fonction de dérive calée sur la courbe initiale des taux
              (formule fermée standard du modèle HW1F, garantissant que le
               modèle restitue exactement la courbe de marché à t=0)
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import itertools
from utils.classes import SimulationParameters


# =============================================================================
# Modèle de Hull & White à 1 facteur
# =============================================================================
class HullWhite1F:
    """
    Modèle de Hull & White à 1 facteur.

    Attributs
    ---------
    a : float
        Paramètre de mean reversion.
    sigma : float
        Paramètre de volatilité.
    sim_params : SimulationParameters
        Objet décrivant les caractéristiques de la simulation.
    forward_rates : pd.DataFrame
        Taux forward simulés (calculé via compute_forward_rates()).
        Colonnes : ['scenario', 'horizon_mois', 'tenor_mois', 'taux_forward']
    discount_factors : pd.DataFrame
        Discount factors simulés (calculé via compute_discount_factors()).
        Colonnes : ['scenario', 'horizon_mois', 'discount_factor']

    Paramètres additionnels (nécessaires pour caler le modèle sur une courbe
    initiale, mais non listés comme "paramètres du modèle" au sens strict) :
    r0 : taux court initial
    initial_curve : fonction t (années) -> taux zéro-coupon continu R(0,t).
                    Par défaut, courbe plate au niveau r0.
    tenors : liste des tenors (en mois) pour lesquels les taux forward
             simulés seront calculés (ex: 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y).
    """

    def __init__(
        self,
        mean_reversion: float,
        volatility: float,
        simulation_params: SimulationParameters,
        tenors: List[int],
        initial_curve: pd.DataFrame,
        initiale_df: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
    ):
        # --- paramètres du modèle ---
        self.a = mean_reversion
        self.sigma = volatility

        # --- objet décrivant la simulation ---
        self.sim_params = simulation_params

        # --- paramètres de calage sur la courbe initiale ---
        self.initial_curve = initial_curve.rename(columns={"forward_month":"horizon_mois","tenor_month":"tenor_mois"})
        self.tenors = tenors
        self.seed = seed

        # --- résultats de simulation (calculés à la demande) ---
        self._stochastic_term: Optional[pd.DataFrame] = None
        self.forward_rates: Optional[pd.DataFrame] = None
        self.discount_factors: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------
    # Fonctions analytiques du modèle (courbe initiale, theta, A, B)
    # -------------------------------------------------------------------
    def _generate_simul_df(self, top_tenor=False) -> pd.DataFrame:
        if top_tenor:
            df_dict = {
                "scenario": [i for i in range(self.sim_params.n_scenarios)],
                "horizon_mois": self.sim_params.horizons_months,
                "tenor_mois": self.tenors
            }
        else:
            df_dict = {
                "scenario": [i for i in range(self.sim_params.n_scenarios)],
                "horizon_mois": self.sim_params.horizons_months
            }
        df_simul = pd.DataFrame(itertools.product(*df_dict.values()), columns=df_dict.keys())
        return df_simul

    def _generate_stochastic_term(self) -> pd.DataFrame:
        df_stoch = self._generate_simul_df().sort_values(["scenario","horizon_mois"])
        rng = np.random.default_rng(self.seed)
        #df_stoch["alea"] = norm.rvs(size=df_stoch.shape[0])
        df_stoch["alea"] = rng.standard_normal(df_stoch.shape[0])
        df_stoch.loc[df_stoch.horizon_mois==0, "alea"] = 0
        df_stoch["alea2"] = (
            df_stoch.alea * self.sigma *
            np.sqrt((1 - np.exp(-2 * self.a * self.sim_params.dt)) / (2 * self.a)) *
            np.exp(self.a * self.sim_params.dt * df_stoch.horizon_mois)
        )
        df_stoch["alea3"] = df_stoch.groupby(["scenario"])["alea2"].cumsum()
        df_stoch["v_stoch"] = df_stoch.alea3 * np.exp(-self.a * self.sim_params.dt * df_stoch.horizon_mois)
        self._stochastic_term = df_stoch[["scenario","horizon_mois","v_stoch"]]
        return self._stochastic_term

    # -------------------------------------------------------------------
    # Calcul des taux forward simulés
    # -------------------------------------------------------------------
    def _compute_forward_rates(self) -> pd.DataFrame:
        """
        Calcule, pour chaque scénario et chaque horizon de projection, les
        taux forward simples pour les tenors définis dans self.tenors.

        Retourne un DataFrame avec les colonnes :
            ['scenario', 'horizon_mois', 'tenor_mois', 'taux_forward']
        """
        df = self._generate_simul_df(top_tenor=True)
        df = (
            df
            .merge(self.initial_curve, how="inner", on=["tenor_mois","horizon_mois"])
            .merge(self._stochastic_term, how="inner", on=["scenario","horizon_mois"])
            .sort_values(["scenario","tenor_mois","horizon_mois"])
        )
        df["bm"] = (
            (1 - np.exp(-self.a * df.tenor_mois * self.sim_params.dt)) /
            (self.a * df.tenor_mois * self.sim_params.dt)
        )
        df["cm"] = (
            (
                self.sigma ** 2 *
                (1 - np.exp(-2 * self.a * df.horizon_mois * self.sim_params.dt)) *
                (1 - np.exp(-self.a * df.tenor_mois * self.sim_params.dt)) ** 2
            ) /
            (4 * (self.a ** 3) * df.tenor_mois * self.sim_params.dt)
        )
        df["taux_forward"] = (
            df.rate_value +
            df.bm * df.v_stoch +
            df.bm * (self.sigma * (1 - np.exp(-self.a * df.horizon_mois * self.sim_params.dt)) / self.a) ** 2 / 2 +
            df.cm
        )
        self.forward_rates = df[["scenario","horizon_mois","tenor_mois","taux_forward"]]
        return self.forward_rates

    # -------------------------------------------------------------------
    # Calcul des discount factors simulés
    # -------------------------------------------------------------------
    def _compute_discount_factors(self) -> pd.DataFrame:
        """
        Calcule, pour chaque scénario et chaque horizon de projection, le
        discount factor à partir des taux 1M.

        Retourne un DataFrame avec les colonnes :
            ['scenario', 'horizon_mois', 'discount_factor']
        """
        df = self.forward_rates.copy()
        df = df.loc[df.tenor_mois==1].reset_index(drop=True).sort_values(["scenario","horizon_mois"])
        df["v1"] = df.groupby("scenario").taux_forward.shift(fill_value=0.0)
        df["v2"] = df.groupby("scenario").v1.cumsum()
        df["discount_factor"] = np.exp(-df.v2 * self.sim_params.dt)
        self.discount_factors = df[["scenario","horizon_mois","discount_factor"]]
        return self.discount_factors

    def _check_df_consistency(self):
        df_compare = (
            self.initiale_df.rename(colulmns={"forward_month":"horizon_mois"})
            .merge(
                self.discount_factors.groupby("horizon_mois").discount_factor.mean().reset_index(),
                how="inner", on="horizon_mois"
            )
            .eval("delta = discount_factor / factor_value - 1")
        )
        if (df_compare.delta.max() > 0.01) & (df_compare.delta.min() < 0.01):
            raise ValueError("Les discount factors simulés ne cadrent pas avec les valeurs initiales.")
        return 1
    
    def _check_vol_consistency(self):
        # A compléter
        return 1

    # -------------------------------------------------------------------
    def run(self):
        """Exécute la simulation complète et renseigne tous les attributs."""
        self._generate_stochastic_term()
        self._compute_forward_rates()
        self._compute_discount_factors()
        return self.forward_rates, self.discount_factors


# =============================================================================
# 3. Exemple d'utilisation
# =============================================================================
if __name__ == "__main__":

    """
    # Courbe initiale des taux zéro-coupon (exemple : courbe plate à 3%)
    def courbe_initiale(t: float) -> float:
        return 0.03

    # Caractéristiques de la simulation :
    # 1 000 scénarios, horizon de projection 5 ans (60 mois), pas mensuel
    sim_params = SimulationParameters(
        n_scenarios=1000,
        horizon_max=60,
        time_step=1,
    )

    # Instanciation du modèle HW1F
    model = HullWhite1F(
        mean_reversion=0.05,
        volatility=0.01,
        simulation_params=sim_params,
        r0=0.03,
        initial_curve=courbe_initiale,
        tenors=[1, 3, 6, 12, 24, 60],
        seed=42,
    )

    forward_rates, discount_factors = model.run()

    print("=== Taux forward simulés (extrait) ===")
    print(forward_rates.head(10))
    print("\nShape :", forward_rates.shape)

    print("\n=== Discount factors simulés (extrait) ===")
    print(discount_factors.head(10))
    print("\nShape :", discount_factors.shape)

    # Exemple de vérification : moyenne des DF sur tous les scénarios à
    # chaque horizon, comparée au DF théorique de la courbe initiale
    moyenne_df = discount_factors.groupby("horizon_mois")["discount_factor"].mean()
    print("\n=== Discount factor moyen (Monte Carlo) par horizon ===")
    print(moyenne_df)
    """