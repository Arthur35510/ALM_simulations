"""
Page Streamlit de gestion de crédits immobiliers.

Volet 1 : saisie des caractéristiques d'un crédit, sauvegarde en base SQLite
           et calcul de l'échéancier contractuel.
Volet 2 : sélection d'un crédit existant et visualisation du profil
           d'écoulement du capital restant dû.

Lancement : streamlit run app_streamlit.py
"""

import streamlit as st

from engine.credit_calculation import (
    CI_AMORTISSEMENT,
    save_credit,
    load_credits,
    load_credit_by_id,
    build_amortization_schedule,
    save_schedule,
    load_schedule
)

st.title("📂 Saisie de crédits immobiliers")

tab_creation, tab_visualisation = st.tabs(
    ["Créer un crédit", "Visualiser un crédit existant"]
)

FORMAT_MONETAIRE = {
    "capital_restant_du_debut": "{:,.2f}",
    "interets": "{:,.2f}",
    "principal_rembourse": "{:,.2f}",
    "echeance": "{:,.2f}",
    "capital_restant_du_fin": "{:,.2f}",
}

# ---------------------------------------------------------------------------
# Volet 1 : création d'un crédit
# ---------------------------------------------------------------------------
with tab_creation:
    st.subheader("Caractéristiques du crédit")

    with st.form("form_creation_credit"):
        col1, col2 = st.columns(2)
        with col1:
            nom = st.text_input("Nom / libellé du crédit", value="Mon crédit immobilier")
            nominal = st.number_input(
                "Nominal (€)", min_value=1000.0, value=200_000.0, step=1000.0, format="%.2f"
            )
            taux_pct = st.number_input(
                "Taux d'intérêt annuel (%)", min_value=0.0, value=3.5, step=0.05, format="%.3f"
            )
        with col2:
            mode = st.selectbox("Mode d'amortissement", CI_AMORTISSEMENT)
            duree_annees = st.number_input(
                "Durée (années)", min_value=1, max_value=40, value=20, step=1
            )

        submitted = st.form_submit_button("💾 Enregistrer et calculer l'échéancier")

    if submitted:
        taux = taux_pct / 100.0
        credit_id = save_credit(
            nom=nom, nominal=nominal, taux=taux, mode=mode, duree_annees=int(duree_annees)
        )
        st.success(f"Crédit « {nom} » enregistré avec l'id {credit_id}.")

        schedule = build_amortization_schedule(
            nominal=nominal, taux_annuel=taux, mode=mode, duree_annees=int(duree_annees)
        )

        save_schedule(credit_id, nominal, schedule)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total intérêts payés", f"{schedule['interets'].sum():,.0f} €")
        col_b.metric("Total remboursé", f"{schedule['echeance'].sum():,.0f} €")
        col_c.metric("Nombre d'échéances", f"{len(schedule)}")

        st.markdown("**Échéancier contractuel**")
        st.dataframe(
            schedule.style.format(FORMAT_MONETAIRE),
            use_container_width=True,
            height=400,
        )

        st.markdown("**Décomposition intérêts / capital par échéance**")
        st.bar_chart(schedule.set_index("mois")[["interets", "principal_rembourse"]])

# ---------------------------------------------------------------------------
# Volet 2 : visualisation d'un crédit existant
# ---------------------------------------------------------------------------
with tab_visualisation:
    st.subheader("Sélection d'un crédit enregistré")

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

        selected_id = st.selectbox(
            "Crédit",
            options=list(label_map.keys()),
            format_func=lambda cid: label_map[cid],
        )

        credit_row = load_credit_by_id(selected_id)
        #schedule = build_schedule_for_credit(credit_row)
        schedule = load_schedule(selected_id)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Nominal", f"{credit_row['nominal']:,.0f} €")
        col_b.metric("Taux", f"{credit_row['taux'] * 100:.2f} %")
        col_c.metric("Mode", credit_row["mode"])
        col_d.metric("Durée", f"{credit_row['duree_annees']} ans")

        st.markdown("**Profil d'écoulement du capital restant dû**")
        st.line_chart(schedule.set_index("t_months")["crd"])
