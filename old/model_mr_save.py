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
        r0: float = 0.02,
        initial_curve: Optional[Callable[[float], float]] = None,
        tenors: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        # --- paramètres du modèle ---
        self.a = mean_reversion
        self.sigma = volatility

        # --- objet décrivant la simulation ---
        self.sim_params = simulation_params

        # --- paramètres de calage sur la courbe initiale ---
        self.r0 = r0
        self.initial_curve = initial_curve if initial_curve is not None else (lambda t: r0)
        self.tenors = tenors if tenors is not None else [1, 3, 6, 12, 24, 60, 120]
        self.seed = seed

        # --- résultats de simulation (calculés à la demande) ---
        self._short_rates: Optional[np.ndarray] = None      # (n_scenarios, n_steps+1)
        self.forward_rates: Optional[pd.DataFrame] = None
        self.discount_factors: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------
    # Fonctions analytiques du modèle (courbe initiale, theta, A, B)
    # -------------------------------------------------------------------
    def _zero_coupon_price(self, t: float) -> float:
        """Prix zéro-coupon initial P(0,t) déduit de la courbe de marché."""
        if t <= 0:
            return 1.0
        return np.exp(-self.initial_curve(t) * t)

    def _instantaneous_forward(self, t: float, h: float = 1e-3) -> float:
        """
        Taux forward instantané initial f(0,t) = -d ln P(0,t)/dt, calculé par
        différences finies. Utilise un schéma décentré près de t=0 pour rester
        dans le domaine de définition (t >= 0) et éviter toute instabilité liée
        à un pas asymétrique.
        """
        if t < h:
            p0 = self._zero_coupon_price(t)
            p1 = self._zero_coupon_price(t + h)
            return -(np.log(p1) - np.log(p0)) / h
        p_plus = self._zero_coupon_price(t + h)
        p_minus = self._zero_coupon_price(t - h)
        return -(np.log(p_plus) - np.log(p_minus)) / (2 * h)

    def _theta(self, t: float, h: float = 1e-3) -> float:
        """
        theta(t) = df(0,t)/dt + a*f(0,t) + sigma^2/(2a) * (1 - exp(-2at))

        Formule fermée assurant que le modèle réplique exactement la courbe
        initiale des taux (calage "on the run"). Le même schéma décentré près
        de t=0 est utilisé pour la dérivée de f afin d'éviter toute instabilité
        numérique liée à la double différentiation.
        """
        if t < h:
            f0 = self._instantaneous_forward(t)
            f1 = self._instantaneous_forward(t + h)
            df_dt = (f1 - f0) / h
        else:
            f_plus = self._instantaneous_forward(t + h)
            f_minus = self._instantaneous_forward(t - h)
            df_dt = (f_plus - f_minus) / (2 * h)
        f_t = self._instantaneous_forward(t)

        a, sigma = self.a, self.sigma
        if abs(a) < 1e-10:
            return df_dt
        return df_dt + a * f_t + (sigma ** 2) / (2 * a) * (1 - np.exp(-2 * a * t))

    def _B(self, t: float, T: float) -> float:
        """Fonction B(t,T) du modèle HW1F."""
        a = self.a
        if abs(a) < 1e-10:
            return T - t
        return (1 - np.exp(-a * (T - t))) / a

    def _A(self, t: float, T: float) -> float:
        """Fonction A(t,T) du modèle HW1F (forme fermée du prix zéro-coupon)."""
        a, sigma = self.a, self.sigma
        B = self._B(t, T)
        term1 = np.log(self._zero_coupon_price(T) / self._zero_coupon_price(t))
        term2 = B * self._instantaneous_forward(t)
        if abs(a) < 1e-10:
            term3 = (sigma ** 2 / 4) * t * B ** 2
        else:
            term3 = (sigma ** 2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B ** 2
        return term1 + term2 - term3

    def _zc_price_from_short_rate(self, t: float, T: float, r_t: np.ndarray) -> np.ndarray:
        """P(t,T | r(t)) = exp(A(t,T) - B(t,T) * r(t))."""
        A = self._A(t, T)
        B = self._B(t, T)
        return np.exp(A - B * r_t)

    # -------------------------------------------------------------------
    # Simulation du taux court (schéma d'Euler)
    # -------------------------------------------------------------------
    def simulate_short_rates(self) -> np.ndarray:
        """Simule les trajectoires du taux court r(t) par un schéma d'Euler."""
        rng = np.random.default_rng(self.seed)
        n_scen = self.sim_params.n_scenarios
        n_steps = self.sim_params.n_steps
        dt = self.sim_params.dt
        time_grid = self.sim_params.time_grid

        rates = np.zeros((n_scen, n_steps + 1))
        rates[:, 0] = self.r0
        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            t = time_grid[i]
            theta_t = self._theta(t)
            dW = rng.standard_normal(n_scen) * sqrt_dt
            rates[:, i + 1] = (
                rates[:, i]
                + (theta_t - self.a * rates[:, i]) * dt
                + self.sigma * dW
            )

        self._short_rates = rates
        return rates

    # -------------------------------------------------------------------
    # Calcul des taux forward simulés
    # -------------------------------------------------------------------
    def compute_forward_rates(self) -> pd.DataFrame:
        """
        Calcule, pour chaque scénario et chaque horizon de projection, les
        taux forward simples pour les tenors définis dans self.tenors.

        Retourne un DataFrame avec les colonnes :
            ['scenario', 'horizon_mois', 'tenor_mois', 'taux_forward']
        """
        if self._short_rates is None:
            self.simulate_short_rates()

        n_scen = self.sim_params.n_scenarios
        time_grid = self.sim_params.time_grid
        horizons_months = self.sim_params.horizons_months

        records = []
        for h_idx, h_months in enumerate(horizons_months):
            t = time_grid[h_idx]
            r_t = self._short_rates[:, h_idx]
            for tenor_months in self.tenors:
                tenor_years = tenor_months / 12.0
                T = t + tenor_years
                P_t_T = self._zc_price_from_short_rate(t, T, r_t)
                # taux forward simple (convention linéaire / actuariel simple)
                fwd_rate = (1.0 / P_t_T - 1.0) / tenor_years
                for s in range(n_scen):
                    records.append((s, int(h_months), tenor_months, fwd_rate[s]))

        df = pd.DataFrame(
            records,
            columns=["scenario", "horizon_mois", "tenor_mois", "taux_forward"],
        )
        self.forward_rates = df
        return df

    # -------------------------------------------------------------------
    # Calcul des discount factors simulés
    # -------------------------------------------------------------------
    def compute_discount_factors(self) -> pd.DataFrame:
        """
        Calcule, pour chaque scénario et chaque horizon de projection, le
        discount factor DF(0,t) = exp(-∫[0,t] r(s) ds), approximé par la
        méthode des trapèzes le long de la trajectoire simulée du taux court.

        Retourne un DataFrame avec les colonnes :
            ['scenario', 'horizon_mois', 'discount_factor']
        """
        if self._short_rates is None:
            self.simulate_short_rates()

        n_scen = self.sim_params.n_scenarios
        n_steps = self.sim_params.n_steps
        dt = self.sim_params.dt
        horizons_months = self.sim_params.horizons_months

        df_matrix = np.zeros((n_scen, n_steps + 1))
        df_matrix[:, 0] = 1.0
        for i in range(1, n_steps + 1):
            r_prev = self._short_rates[:, i - 1]
            r_curr = self._short_rates[:, i]
            integral_step = 0.5 * (r_prev + r_curr) * dt
            df_matrix[:, i] = df_matrix[:, i - 1] * np.exp(-integral_step)

        records = []
        for h_idx, h_months in enumerate(horizons_months):
            for s in range(n_scen):
                records.append((s, int(h_months), df_matrix[s, h_idx]))

        df = pd.DataFrame(records, columns=["scenario", "horizon_mois", "discount_factor"])
        self.discount_factors = df
        return df

    # -------------------------------------------------------------------
    def run(self):
        """Exécute la simulation complète et renseigne tous les attributs."""
        self.simulate_short_rates()
        self.compute_forward_rates()
        self.compute_discount_factors()
        return self.forward_rates, self.discount_factors


# =============================================================================
# 3. Exemple d'utilisation
# =============================================================================
if __name__ == "__main__":

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