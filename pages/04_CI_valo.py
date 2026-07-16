"""
Page Streamlit de gestion de crédits immobiliers.

Volet 1 : saisie des caractéristiques d'un crédit, sauvegarde en base SQLite
           et calcul de l'échéancier contractuel.
Volet 2 : sélection d'un crédit existant et visualisation du profil
           d'écoulement du capital restant dû.

Lancement : streamlit run app_streamlit.py
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date

from engine.curves import load_forward_rates, load_discount_factors
from engine.credit_calculation import load_credits, load_credit_by_id, load_schedule
from engine.model_mr import SimulationParameters, HullWhite1F
from engine.credit_valorisation import valorisation_ci, save_simul, save_valo, load_valo

from database import execute_query
from config import HW_MEAN_REVERSION, HW_VOLATILITY, SIMUL_N_SCEN, SIMUL_TMAX_MONTHS, SIMUL_STEP_MONTHS

st.title("🔬 Valorisation de crédits immobiliers")

tab_creation, tab_visualisation = st.tabs(
    ["Lancer une valorisation", "Visualiser des résultats"]
)

# ---------------------------------------------------------------------------
# Volet 1 : Valorisation d'un crédit
# ---------------------------------------------------------------------------
with tab_creation:
    st.subheader("Paramètres de la simulation")

    # Liste des dates de courbes disponibles
    dates_rows = execute_query("""
        SELECT DISTINCT curve_date FROM curve_dates 
        ORDER BY curve_date DESC
    """)

    if not dates_rows:
        st.warning("Aucune courbe disponible. Chargez-en une d'abord dans l'onglet Charger.")
        st.stop()

    available_dates = []
    for row in dates_rows:
        available_dates.append(row[0] if hasattr(row, "__getitem__") else getattr(row, "curve_date", None))
    available_dates = [str(d) for d in available_dates if d is not None]

    # Liste des credits chargés disponibles
    credits_df = load_credits()

    if credits_df.empty:
        st.info("Aucun crédit enregistré pour le moment. Créez-en un dans le premier onglet.")
    else:
        label_map = {
            row.id: (
                f"#{row.id} — {row.nom} — {row.nominal:,.0f} € — "
                f"{row.taux * 100:.2f}% — {row.mode} — {row.duree_annees} ans"
            )
            for row in credits_df.itertuples()
        }

    with st.form("form_creation_simulation"):
        col1, col2 = st.columns(2)
        with col1:
            hw_a = st.number_input(
                "H&W mean reversion", min_value=0.0, value=HW_MEAN_REVERSION, step=0.001, format="%.3f"
            )
            view_date_str = st.selectbox("Date", available_dates)
            view_date = date.fromisoformat(view_date_str)
            horizon_max = st.number_input(
                "Horizon simulation (mois)", min_value=1, value=SIMUL_TMAX_MONTHS, step=1
            )
            credit_id = st.selectbox(
                "Crédit",
                options=list(label_map.keys()),
                format_func=lambda cid: label_map[cid],
            )

        with col2:
            hw_s = st.number_input(
                "H&W volatility", min_value=0.0, value=HW_VOLATILITY, step=0.001, format="%.3f"
            )
            n_scenarios = st.number_input(
                "Nombre de scénarios", min_value=1, value=SIMUL_N_SCEN, step=1
            )
            time_step = st.number_input(
                "Pas de temps (mois)", min_value=1, value=SIMUL_STEP_MONTHS, step=1
            )

        submitted = st.form_submit_button("💾 Enregistrer et lancer la simulation")

    if submitted:
        
        simul_id = save_simul(credit_id, hw_a, hw_s, view_date, n_scenarios, horizon_max, time_step)

        st.markdown("### 1. Taux de marché et discount-factors")

        # Récupération des taux forward et discount factors
        forward_rates_0 = load_forward_rates(view_date)
        discount_factors_0 = load_discount_factors(view_date)

        # Initialisation des parametres de simulation
        sim_params = SimulationParameters(
            n_scenarios=n_scenarios,
            horizon_max=horizon_max,
            time_step=time_step,
        )

        # Instanciation du modèle HW1F and simulation des market rates
        model = HullWhite1F(
            mean_reversion=hw_a,
            volatility=hw_s,
            simulation_params=sim_params,
            tenors=[1, 120],
            initial_curve=forward_rates_0,
            initial_df=discount_factors_0,
            seed=42,
        )
        forward_rates, discount_factors = model.run()

        if not forward_rates.empty:
            fig1 = go.Figure()

            tenors_a_afficher = [1, 120]
            palette = ["#1f77b4", "#d62728"]

            for tenor, couleur in zip(tenors_a_afficher, palette):
                df_tenor = forward_rates[forward_rates["tenor_mois"] == tenor]

                # Agrégation par horizon : moyenne + bornes de l'intervalle de confiance à 95%
                stats = (
                    df_tenor.groupby("horizon_mois")["taux_forward"]
                    .agg(
                        moyenne="mean",
                        borne_basse=lambda x: x.quantile(0.025),
                        borne_haute=lambda x: x.quantile(0.975),
                    )
                    .reset_index()
                    .sort_values("horizon_mois")
                )

                # Conversion de la couleur hex en rgba pour l'enveloppe semi-transparente
                r, g, b = tuple(int(couleur.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

                # Enveloppe de l'intervalle de confiance (bande continue borne haute -> borne basse)
                fig1.add_trace(
                    go.Scatter(
                        x=pd.concat([stats["horizon_mois"], stats["horizon_mois"][::-1]]),
                        y=pd.concat([stats["borne_haute"], stats["borne_basse"][::-1]]),
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{b},0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"IC 95% - tenor {tenor}M",
                    )
                )

                # Courbe du taux moyen
                fig1.add_trace(
                    go.Scatter(
                        x=stats["horizon_mois"],
                        y=stats["moyenne"],
                        mode="lines",
                        line=dict(color=couleur, width=2),
                        name=f"Taux moyen - tenor {tenor}M",
                    )
                )

            fig1.update_layout(
                title="Taux forward simulés : moyenne et intervalle de confiance à 95%",
                xaxis_title="Horizon de projection (mois)",
                yaxis_title="Taux forward",
                yaxis_tickformat=".2%",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            st.plotly_chart(fig1, use_container_width=True)

        if not discount_factors.empty:
            fig2 = go.Figure()

            couleur = ["#1f77b4", "#b41f1f"]

            # Agrégation par horizon : moyenne + bornes de l'intervalle de confiance à 95%
            stats = (
                discount_factors.groupby("horizon_mois")["discount_factor"]
                .agg(
                    moyenne="mean",
                    borne_basse=lambda x: x.quantile(0.025),
                    borne_haute=lambda x: x.quantile(0.975),
                )
                .reset_index()
                .sort_values("horizon_mois")
            )

            # Conversion de la couleur hex en rgba pour l'enveloppe semi-transparente
            r, g, b = tuple(int(couleur[0].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

            # Enveloppe de l'intervalle de confiance (bande continue borne haute -> borne basse)
            fig2.add_trace(
                go.Scatter(
                    x=pd.concat([stats["horizon_mois"], stats["horizon_mois"][::-1]]),
                    y=pd.concat([stats["borne_haute"], stats["borne_basse"][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"IC 95%",
                )
            )

            # Courbe du DF moyen
            fig2.add_trace(
                go.Scatter(
                    x=stats["horizon_mois"],
                    y=stats["moyenne"],
                    mode="lines",
                    line=dict(color=couleur[0], width=2),
                    name=f"Discount factor moyen",
                )
            )

            # Courbe du DF initial
            fig2.add_trace(
                go.Scatter(
                    x=discount_factors_0["forward_month"],
                    y=discount_factors_0["factor_value"],
                    mode="lines",
                    line=dict(color=couleur[1], width=2),
                    name=f"Discount factor initial",
                )
            )

            fig2.update_layout(
                title="Discount factor simulés : moyenne et intervalle de confiance à 95%",
                xaxis_title="Horizon de projection (mois)",
                yaxis_title="Discount factor",
                yaxis_tickformat=".2%",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.markdown("### 2. Modélisation RARN")

        # Récupération du crédit
        credit_row = load_credit_by_id(credit_id)
        ci_schedule = load_schedule(credit_id)

        # Simulation et valorisation
        df_rarn, dict_valo = valorisation_ci(discount_factors, forward_rates, ci_schedule, credit_row["taux"])
        valo_id = save_valo(simul_id, dict_valo)
        
        if not df_rarn.empty:

            simul_schedule = df_rarn.groupby(["horizon_mois"]).agg({"crd":"mean", "crd_ra":"mean"}).reset_index()

            fig2 = go.Figure()

            colors_schedule = {
                "crd": '#2ca02c',
                "crd_ra": '#ff7f0e'
            }

            labels_schedule = {
                "crd": 'CRD ctrl',
                "crd_ra": 'CRD ra'
            }
            for type_schedule in ["crd", "crd_ra"]:

                color = colors_schedule[type_schedule]
                label = labels_schedule[type_schedule]

                fig2.add_trace(go.Scatter(
                    x=simul_schedule["horizon_mois"] / 12,
                    y=simul_schedule[type_schedule],
                    mode='lines',
                    name=label,
                    line=dict(width=1.5, color=color)
                ))

            fig2.update_layout(
                title=f"Ecoulement du crédit avant et après application du modèle",
                xaxis_title="Horizon de projection (années)",
                yaxis_title="CRD",
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.markdown("### 3. Valorisation")

        df_valo = load_valo(valo_id)
        st.dataframe(df_valo, use_container_width=True)



# ---------------------------------------------------------------------------
# Volet 2 : visualisation d'un crédit existant
# ---------------------------------------------------------------------------
with tab_visualisation:
    st.subheader("Sélection d'un crédit enregistré")
