from dataclasses import dataclass
import numpy as np

# =============================================================================
# Caractéristiques d'une simulation
# =============================================================================

@dataclass
class SimulationParameters:
    """
    Caractéristiques de la simulation Monte Carlo.

    Attributs
    ---------
    n_scenarios : int
        Nombre de trajectoires (scénarios) simulées.
    horizon_max : int
        Horizon maximal de projection, exprimé en mois.
    time_step : int
        Pas de temps de la simulation, exprimé en mois.
    """
    n_scenarios: int
    horizon_max: int   # en mois
    time_step: int      # en mois

    def __post_init__(self):
        if self.horizon_max % self.time_step != 0:
            raise ValueError("horizon_max doit être un multiple de time_step.")
        self.n_steps = int(self.horizon_max / self.time_step)
        self.dt = self.time_step / 12.0
        self.time_grid = np.arange(0, self.n_steps + 1) * self.dt
        self.horizons_months = np.round(self.time_grid * 12).astype(int)
