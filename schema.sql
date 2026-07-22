-- ============================================================
-- Schéma de base de données ALM Tool (SQLite)
-- Version corrigée selon spécifications utilisateur
-- ============================================================

-- --------------------------------------------------------------
-- A. COURBES DE TAUX
-- --------------------------------------------------------------

CREATE TABLE IF NOT EXISTS curve_dates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curve_date DATE NOT NULL,
    source_type TEXT CHECK(source_type IN ('market','zc','forward','df')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(curve_date, source_type)
);

CREATE TABLE IF NOT EXISTS market_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curve_date_id INTEGER NOT NULL REFERENCES curve_dates(id),
    instrument_type TEXT CHECK(instrument_type IN ('euribor','swap')),
    tenor_months INTEGER NOT NULL,
    rate_value DECIMAL(10,6) NOT NULL,
    UNIQUE(curve_date_id, tenor_months)
);

CREATE TABLE IF NOT EXISTS zero_coupons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curve_date_id INTEGER NOT NULL REFERENCES curve_dates(id),
    tenor_months INTEGER NOT NULL CHECK(tenor_months BETWEEN 1 AND 360),
    rate_value DECIMAL(10,6) NOT NULL,
    UNIQUE(curve_date_id, tenor_months)
);

CREATE TABLE IF NOT EXISTS forward_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curve_date_id INTEGER NOT NULL REFERENCES curve_dates(id),
    forward_month INTEGER NOT NULL,
    tenor_month INTEGER NOT NULL,
    rate_value DECIMAL(10,6) NOT NULL,
    UNIQUE(curve_date_id, forward_month, tenor_month)
);

CREATE TABLE IF NOT EXISTS discount_factors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curve_date_id INTEGER NOT NULL REFERENCES curve_dates(id),
    forward_month INTEGER NOT NULL,
    factor_value DECIMAL(10,6) NOT NULL,
    UNIQUE(curve_date_id, forward_month)
);

-- --------------------------------------------------------------
-- B. CONTRATS
-- --------------------------------------------------------------

CREATE TABLE IF NOT EXISTS credits_immo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nom TEXT,
    mode TEXT NOT NULL,
    duree_annees INTEGER NOT NULL,
    taux DECIMAL(10,6) NOT NULL,
    nominal INTEGER NOT NULL,
    date_creation TEXT NOT NULL,
    UNIQUE(mode, duree_annees, taux, nominal)
);

CREATE TABLE IF NOT EXISTS ci_ecoulements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ci_id INTEGER NOT NULL REFERENCES credits_immo(id),
    t_months INTEGER NOT NULL,
    crd DECIMAL(10,6) NOT NULL,
    UNIQUE(ci_id, t_months)
);


-- --------------------------------------------------------------
-- C. SIMULATION
-- --------------------------------------------------------------

CREATE TABLE IF NOT EXISTS simulations_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hw_a DECIMAL(10,6) NOT NULL,
    hw_s DECIMAL(10,6) NOT NULL,
    curve_date DATE NOT NULL,
    nb_scenarios INTEGER NOT NULL,
    h_max_months INTEGER NOT NULL,
    time_step_months INTEGER NOT NULL,
    UNIQUE(hw_a, hw_s, curve_date, nb_scenarios, h_max_months, time_step_months)
);

CREATE TABLE IF NOT EXISTS ci_valorisations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    simul_id INTEGER NOT NULL REFERENCES simulations_config(id),
    ci_id INTEGER NOT NULL REFERENCES credits_immo(id),
    valo_ctrl DECIMAL(10,6) NOT NULL,
    valo_ra DECIMAL(10,6) NOT NULL,
    valo_rn DECIMAL(10,6) NOT NULL,
    valo_rarn DECIMAL(10,6) NOT NULL,
    UNIQUE(simul_id, ci_id)
);

