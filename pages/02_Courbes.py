"""Page Courbes de taux - Chargement, calcul et visualisation."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from io import StringIO

from engine.curves import (
    load_market_rates, load_zero_coupons, load_discount_factors,
    bootstrap_zero_coupons, interpolate_zero_coupons,
    compute_forward_tenor, compute_discount_factor,
    store_curve_data, get_curve_for_date
)
from utils.io import read_rates_file
from database import execute_query
from config import ZC_TENORS, FWD_TENORS

st.title("📈 Courbes de taux")

tab_load, tab_view, tab_compare = st.tabs(["Charger", "Visualiser", "Comparer"])

# =============================================================================
# ONGLET CHARGER (inchangé)
# =============================================================================
with tab_load:
    st.subheader("Chargement des données")

    uploaded_file = st.file_uploader(
        "Fichier CSV ou Excel",
        type=["csv", "xlsx"],
        help="""Format attendu : date (format YYYY-MM-DD) | instrument_type | tenor_months | rate_value"""
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write("Aperçu du fichier :")
            st.dataframe(df_upload.head(10), use_container_width=True)

            # Détection automatique de la colonne date (première colonne)
            date_col = df_upload.columns[0]
            st.info(f"📅 Colonne date détectée : '{date_col}'")

            # Conversion des dates
            df_upload[date_col] = pd.to_datetime(df_upload[date_col], errors='coerce')
            invalid_dates = df_upload[date_col].isna().sum()
            if invalid_dates > 0:
                st.warning(f"⚠️ {invalid_dates} lignes avec date invalide ignorées")
                df_upload = df_upload.dropna(subset=[date_col])

            # Extraction des dates uniques
            unique_dates = df_upload[date_col].dt.date.unique()
            st.success(f"✅ {len(unique_dates)} date(s) trouvée(s) : {', '.join([str(d) for d in sorted(unique_dates)[:5]])}{'...' if len(unique_dates) > 5 else ''}")

            # Mapping automatique des colonnes (hors première colonne date)
            remaining_cols = list(df_upload.columns[1:])
            col_mapping = {}
            cols_lower = [c.lower().strip() for c in remaining_cols]

            # Taux marché
            for c in remaining_cols:
                cl = c.lower().strip()
                if cl in ['instrument_type', 'type', 'instrument']:
                    col_mapping['instrument_type'] = c
                elif cl in ['tenor_months', 'tenor', 'maturity_months', 'maturite_mois']:
                    col_mapping['tenor_months'] = c
                elif cl in ['rate_value', 'rate', 'taux', 'value']:
                    col_mapping['rate_value'] = c

            if all(k in col_mapping for k in ['instrument_type', 'tenor_months', 'rate_value']):
                st.success(f"✅ Format taux marché reconnu ({len(remaining_cols)} colonnes)")

                # Traitement par date
                if st.button("💾 Stocker et calculer les courbes", type="primary"):
                    progress = st.progress(0)
                    status = st.empty()

                    total_dates = len(unique_dates)
                    processed = 0

                    for curve_date in sorted(unique_dates):
                        try:
                            status.text(f"Traitement du {curve_date}...")

                            # Filtrage par date
                            df_date = df_upload[df_upload[date_col].dt.date == curve_date].copy()

                            # Renommage
                            df_mapped = df_date.rename(columns={
                                col_mapping['instrument_type']: 'instrument_type',
                                col_mapping['tenor_months']: 'tenor_months',
                                col_mapping['rate_value']: 'rate_value'
                            })

                            # Validation
                            valid_types = {'euribor', 'swap'}
                            df_mapped['instrument_type'] = df_mapped['instrument_type'].str.lower().str.strip()
                            df_mapped = df_mapped[df_mapped['instrument_type'].isin(valid_types)]
                            df_mapped['tenor_months'] = pd.to_numeric(df_mapped['tenor_months'], errors='coerce')
                            df_mapped['rate_value'] = pd.to_numeric(df_mapped['rate_value'], errors='coerce')
                            df_mapped = df_mapped.dropna(subset=['tenor_months', 'rate_value'])

                            if df_mapped.empty:
                                st.warning(f"⚠️ Aucun taux valide pour le {curve_date}")
                                continue

                            # Stockage des taux marché
                            store_curve_data(curve_date, "market", df_mapped, "market_rates")
                            
                            # Bootstrapping
                            zc_boot = bootstrap_zero_coupons(df_mapped)
                            
                            # Interpolation mensuelle
                            zc_interp = interpolate_zero_coupons(zc_boot, ZC_TENORS)

                            # Stockage ZC
                            store_curve_data(curve_date, "zc", zc_interp, "zero_coupons")
                            
                            # Calcul et stockage des forwards
                            l_df_fwd = []
                            for tenor in FWD_TENORS:
                                fwd_tenor = compute_forward_tenor(zc_interp, tenor)
                                l_df_fwd.append(fwd_tenor)

                            store_curve_data(curve_date, "forward", pd.concat(l_df_fwd), "forward_rates")
                            
                            # Calcul et stockage des DF
                            df_df = compute_discount_factor(zc_interp)
                            store_curve_data(curve_date, "df", df_df, "discount_factors")

                            processed += 1
                            progress.progress(processed / total_dates)

                        except Exception as e:
                            st.warning(f"⚠️ Erreur pour le {curve_date} : {e}")
                            continue

                    status.empty()
                    progress.empty()

                    if processed > 0:
                        st.success(f"✅ {processed}/{total_dates} courbe(s) calculée(s) et stockée(s) !")
                        st.balloons()
                    else:
                        st.error("❌ Aucune courbe n'a pu être calculée.")
            else:
                missing = [k for k in ['instrument_type', 'tenor_months', 'rate_value'] if k not in col_mapping]
                st.error(f"Colonnes requises non trouvées : {missing}")

        except Exception as e:
            st.error(f"❌ Erreur lecture fichier : {e}")

# =============================================================================
# ONGLET VISUALISER - REFONTE
# =============================================================================
with tab_view:
    st.subheader("Visualisation des courbes")

    # --- Sélection unique de la date ---
    dates_rows = execute_query("""
        SELECT DISTINCT curve_date FROM curve_dates 
        ORDER BY curve_date DESC
    """)

    if not dates_rows:
        st.warning("Aucune courbe disponible. Chargez-en une d'abord dans l'onglet Charger.")
        st.stop()

    # Conversion explicite des sqlite3.Row
    available_dates = []
    for row in dates_rows:
        available_dates.append(row[0] if hasattr(row, "__getitem__") else getattr(row, "curve_date", None))
    available_dates = [str(d) for d in available_dates if d is not None]

    view_date_str = st.selectbox("Date", available_dates)
    view_date = date.fromisoformat(view_date_str)

    st.divider()

    # =================================================================
    # GRAPHIQUE 1 : Données de marché
    # =================================================================
    st.markdown("### 1. Taux de marché et zéro-coupon")

    mr = load_market_rates(view_date)
    zc = load_zero_coupons(view_date)
    
    if not mr.empty:
        fig1 = go.Figure()

        # Séparation Euribor / Swaps
        euribor = mr[mr["instrument_type"] == "euribor"]
        swaps = mr[mr["instrument_type"] == "swap"]

        if not zc.empty:
            fig1.add_trace(go.Scatter(
                x=zc["tenor_months"] / 12,
                y=zc["rate_value"] * 100,
                mode='lines',
                name='Zero-coupon',
                line=dict(width=1.5, color="#b41f1f")
            ))

        if not euribor.empty:
            fig1.add_trace(go.Scatter(
                x=euribor["tenor_months"] / 12,
                y=euribor["rate_value"] * 100,
                mode='markers',
                name='Euribor',
                marker=dict(size=5, color='#2ca02c', symbol='circle')
            ))

        if not swaps.empty:
            fig1.add_trace(go.Scatter(
                x=swaps["tenor_months"] / 12,
                y=swaps["rate_value"] * 100,
                mode='markers',
                name='Swaps',
                marker=dict(size=5, color='#1f77b4', symbol='diamond')
            ))


        fig1.update_layout(
            title=f"Taux de marché au {view_date.strftime('%d/%m/%Y')}",
            xaxis_title="Maturité (années)",
            yaxis_title="Taux (%)",
            height=350,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig1, use_container_width=True)

    else:
        st.info("Aucune donnée de marché disponible pour cette date.")

    st.divider()


    # =================================================================
    # GRAPHIQUE 2 : Courbes forward
    # =================================================================
    st.markdown("### 2. Courbes de taux forward")

    # Récupération de tous les forwards pour cette date
    fwd_rows = execute_query("""
        SELECT DISTINCT tenor_month
        FROM forward_rates fr
        JOIN curve_dates cd ON fr.curve_date_id = cd.id
        WHERE cd.curve_date = ?
        ORDER BY tenor_month
    """, (view_date_str,))
    

    fig3 = go.Figure()
    has_fwd = False

    colors_fwd = {
        1: '#d62728',
        12: '#ff7f0e',
        60: '#2ca02c',
        120: '#1f77b4'
    }

    labels_fwd = {
        1: 'Forward 1M',
        12: 'Forward 12M',
        60: 'Forward 60M',
        120: 'Forward 120M',
    }

    if fwd_rows:
        for row in fwd_rows:
            print(row[0])
            tenor = int(row[0] if hasattr(row, "__getitem__") else getattr(row, "tenor_month", 0))

            # Chargement des forwards pour ce tenor
            fwd_data = execute_query("""
                SELECT forward_month, rate_value
                FROM forward_rates fr
                JOIN curve_dates cd ON fr.curve_date_id = cd.id
                WHERE cd.curve_date = ? AND fr.tenor_month = ?
                ORDER BY forward_month
            """, (view_date_str, tenor))

            if fwd_data:
                fwd_df = pd.DataFrame([
                    {"forward_month": r[0], "rate_value": r[1]} 
                    for r in fwd_data
                ])

                if not fwd_df.empty:
                    has_fwd = True
                    color = colors_fwd.get(tenor, '#8c564b')
                    label = labels_fwd.get(tenor, f'Forward {tenor}M')

                    # Style spécial pour l'instantané (ligne pointillée)
                    dash_style = 'dash' if tenor == 0 else 'solid'

                    fig3.add_trace(go.Scatter(
                        x=fwd_df["forward_month"] / 12,
                        y=fwd_df["rate_value"] * 100,
                        mode='lines',
                        name=label,
                        line=dict(width=1.5, color=color, dash=dash_style)
                    ))

    if has_fwd:
        fig3.update_layout(
            title=f"Courbes de taux forward au {view_date.strftime('%d/%m/%Y')}",
            xaxis_title="Départ du forward (années)",
            yaxis_title="Taux (%)",
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Aucune courbe forward disponible pour cette date.")

    # =================================================================
    # GRAPHIQUE 3 : Discount factors
    # =================================================================
    st.markdown("### 3. Discount factors")

    df = load_discount_factors(view_date)
    
    if not df.empty:
        fig1 = go.Figure()

        fig1.add_trace(go.Scatter(
            x=df["forward_month"] / 12,
            y=df["factor_value"] * 100,
            mode='lines',
            name='Discount factor',
            line=dict(width=1.5, color="#b41f1f")
        ))

        fig1.update_layout(
            title=f"Discount factor au {view_date.strftime('%d/%m/%Y')}",
            xaxis_title="Horizon (années)",
            yaxis_title="(%)",
            height=350,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig1, use_container_width=True)

    else:
        st.info("Aucune donnée de marché disponible pour cette date.")

    st.divider()

# =============================================================================
# ONGLET COMPARER (inchangé)
# =============================================================================
with tab_compare:
    st.subheader("Comparaison multi-dates")

    dates_rows = execute_query("""
        SELECT DISTINCT curve_date FROM curve_dates 
        WHERE source_type IN ('zero', 'zc_interp')
        ORDER BY curve_date DESC
    """)
    if not dates_rows:
        st.info("Aucune courbe ZC disponible.")
        st.stop()

    # Conversion explicite
    available_dates = []
    for row in dates_rows:
        available_dates.append(row[0] if hasattr(row, "__getitem__") else getattr(row, "curve_date", None))
    available_dates = [str(d) for d in available_dates if d is not None]

    selected_dates = st.multiselect(
        "Sélectionnez les dates à comparer",
        available_dates,
        default=available_dates[:2] if len(available_dates) >= 2 else available_dates
    )

    if len(selected_dates) < 2:
        st.info("Sélectionnez au moins 2 dates pour comparer.")
        st.stop()

    if st.button("🔄 Comparer", type="primary"):
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, d_str in enumerate(selected_dates):
            d = date.fromisoformat(d_str)
            zc = get_curve_for_date(d, source_type="zero")
            if zc is not None and not zc.empty:
                fig.add_trace(go.Scatter(
                    x=zc["tenor_months"] / 12,
                    y=zc["rate_value"] * 100,
                    mode='lines',
                    name=d_str,
                    line=dict(width=2, color=colors[i % len(colors)])
                ))

        fig.update_layout(
            title="Comparaison des courbes ZC",
            xaxis_title="Maturité (années)",
            yaxis_title="Taux (%)",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tableau d'écart
        with st.expander("📊 Écarts entre dates"):
            if len(selected_dates) == 2:
                d1 = date.fromisoformat(selected_dates[0])
                d2 = date.fromisoformat(selected_dates[1])
                zc1 = get_curve_for_date(d1, source_type="zero")
                zc2 = get_curve_for_date(d2, source_type="zero")

                if zc1 is not None and zc2 is not None:
                    merged = pd.merge(
                        zc1[["tenor_months", "rate_value"]].rename(columns={"rate_value": "r1"}),
                        zc2[["tenor_months", "rate_value"]].rename(columns={"rate_value": "r2"}),
                        on="tenor_months"
                    )
                    merged["ecart_bp"] = (merged["r2"] - merged["r1"]) * 10000
                    merged["ecart_pct"] = (merged["r2"] - merged["r1"]) * 100
                    merged = merged[["tenor_months", "r1", "r2", "ecart_bp", "ecart_pct"]]
                    merged.columns = ["Mois", f"Taux {selected_dates[0]}", f"Taux {selected_dates[1]}", "Écart (bp)", "Écart (%)"]
                    st.dataframe(merged, use_container_width=True, hide_index=True)
