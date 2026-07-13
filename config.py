"""Configuration globale de l'application ALM."""
import os
from pathlib import Path

# Répertoire racine du projet
BASE_DIR = Path(__file__).parent.resolve()

# Base de données SQLite
DB_PATH = BASE_DIR / "data" / "alm_data.db"

# Répertoire des données
DATA_DIR = BASE_DIR / "data"

# Paramètres par défaut
DEFAULT_CURRENCY = "EUR"
DEFAULT_BUCKET_TYPE = "monthly"
DEFAULT_HORIZON_MONTHS = 120
DEFAULT_PAYMENT_FREQ_MONTHS = 3

# Taux marché : maturités attendues (en mois)
EURIBOR_TENORS = list(range(1, 13, 1))
SWAP_TENORS = list(range(24, 361, 12))

# Zéro-coupon : maturités mensuelles cibles
ZC_TENORS = list(range(1, 361))

# Forward : horizon de projection
FWD_HORIZONS = list(range(361))
FWD_TENORS = [1, 12, 60, 120]

# Paramètres des simulations monte-carlo
SIMUL_N_SCEN = 100
SIMUL_TMAX_MONTHS = 360
SIMUL_STEP_MONTHS = 1

# Hull-White paramètres par défaut
HW_MEAN_REVERSION = 0.03
HW_VOLATILITY = 0.01

# Mode amortissement des crédits immobiliers
CI_AMORTISSEMENT = ["Echéance fixe", "In fine", "Amortissement constant"]