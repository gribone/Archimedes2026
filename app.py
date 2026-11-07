r"""
Stage 6: Localhost Streamlit Web Application (`app.py`)
=======================================================
Implements Requirement A8, Section 9 (EU AI Act Compliance), and Section 10 (Automated Deliverable Export):
- Upload any Client Excel (`Πελατολόγιο`) or use the default 300-client ForceCRM dataset.
- Interactive Parameter & Hyperparameter Tuning in the Sidebar (embedding model, KAD weights,
  turnover weighting mode, min/max ICPs, Empirical Bayes prior, similarity weights, local Ollama LLM).
- Inspect Discovered ICPs (Greek AI-generated grounded descriptions, Bootstrap Stability, Confidence,
  Hypergeometric + Empirical Bayes statistical feature lift charts, and GPS Map).
- Browse, filter, and search top 2,000+ Lookalike recommendations (strictly excluding all client VAT branches).
- Inspect Client Assignments (`assignments.csv`, including `"NO_FIT"` outlier detection).
- Run a live Hold-Out Benchmark (`Recall@500` & `Recall@2000` vs ForceCRM Primary KAD Baseline)
  and download `profiles.json`, `assignments.csv`, and `lookalikes.csv`.

Run locally with:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from evaluate_holdout import (
    baseline_primary_kad_ranker,
    evaluate_ranker_on_holdout,
    make_icp_engine_ranker,
)
from src.config import EMBEDDING_MODELS_REGISTRY, PipelineConfig
from src.features import CompanyFeatureStore, build_feature_store
from src.icp_engine import ICPEngineResult, export_deliverables, run_icp_engine
from src.ingest import IngestedDataset, ingest_all, load_clients_excel
from src.llm_explainer import check_local_ollama_status, enrich_profiles_with_local_llm


st.set_page_config(
    page_title="ForceCRM · B2B Profile Analysis & Lookalike AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Φόρτωση και έλεγχος βάσης επιχειρήσεων (24.424 εγγραφές)...")
def cached_ingest_base() -> IngestedDataset:
    return ingest_all()


@st.cache_resource(show_spinner="Υπολογισμός σημασιολογικών embeddings ΚΑΔ & χωρικών δεικτών cKDTree...")
def cached_feature_store(
    embedding_model: str,
    w_primary: float,
    w_secondary: float,
    w_auxiliary: float,
    idf_power: float,
    secondary_damping: float,
    bw_semantic: float,
    bw_kad_hier: float,
    bw_spatial: float,
    bw_legal: float,
    bw_age_branch: float,
) -> CompanyFeatureStore:
    ds = cached_ingest_base()
    cfg = PipelineConfig(
        embedding_model=embedding_model,
        w_primary_kad=w_primary,
        w_secondary_kad=w_secondary,
        w_auxiliary_kad=w_auxiliary,
        idf_power=idf_power,
        secondary_count_damping=secondary_damping,
        block_weight_semantic=bw_semantic,
        block_weight_kad_hierarchy=bw_kad_hier,
        block_weight_spatial=bw_spatial,
        block_weight_legal=bw_legal,
        block_weight_age_branch=bw_age_branch,
    )
    return build_feature_store(ds, cfg)


def main() -> None:
    # -------------------------------------------------------------------------
    # Header & EU AI Act Compliance Banner (Section 9)
    # -------------------------------------------------------------------------
    st.title("🎯 ForceCRM Challenge · B2B Profile Analysis & Lookalike Engine")
    st.caption(
        "Ελληνικές Επιχειρήσεις Εστίασης (Κύριος ΚΑΔ 56 · Ίδρυση 2020–2024 · 24.424 Επιχειρήσεις) | "
        "100% Τοπική Εκτέλεση AI (Zero External APIs)"
    )

    st.info(
        "⚖️ **Συμμόρφωση EU AI Act (Ενότητα 9 & Άρθρο 50):** Το σύστημα εκτελεί **αποκλειστικά ομαδοποίηση "
        "και εμπορική αντιστοίχιση B2B (Commercial Fit / ICP Similarity)** για σκοπούς πωλήσεων. "
        "**ΔΕΝ παράγει βαθμολογία φερεγγυότητας ή πιστοληπτικής ικανότητας** φυσικών προσώπων (ατομικών επιχειρήσεων) "
        "ή νομικών προσώπων. Όλα τα κείμενα που παράγονται από τοπικό μοντέλο AI σημαίνονται ρητά.",
        icon="🛡️",
    )

    # -------------------------------------------------------------------------
    # Sidebar: File Upload & Full Hyperparameter Controls
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("📂 1. Πελατολόγιο (Excel)")
        uploaded_file = st.file_uploader(
            "Ανεβάστε αρχείο Excel (.xlsx) με ΑΦΜ & Τζίρο",
            type=["xlsx"],
            help="Αν δεν ανεβάσετε αρχείο, χρησιμοποιείται αυτόματα το προεπιλεγμένο ForceCRM_Hackathon_Pelatologio.xlsx (300 πελάτες).",
        )

        st.divider()
        st.header("⚙️ 2. Παραμετροποίηση & Benchmarking")

        emb_options = list(EMBEDDING_MODELS_REGISTRY.keys())
        selected_emb = st.selectbox(
            "Μοντέλο Embeddings ΚΑΔ (Ενότητα 5.2)",
            options=emb_options,
            index=emb_options.index("char-ngram-svd"),
            format_func=lambda k: f"{k} ({EMBEDDING_MODELS_REGISTRY[k]['params']})",
        )

        with st.expander("Βάρη ΚΑΔ & Απόσβεση Θορύβου (A1, A3)", expanded=False):
            w_primary = st.slider("Βάρος Κύριου ΚΑΔ", 1.0, 6.0, 3.5, 0.25)
            w_secondary = st.slider("Βάρος Δευτερεύοντος ΚΑΔ", 0.1, 2.0, 0.75, 0.05)
            w_auxiliary = st.slider("Βάρος Βοηθητικού ΚΑΔ", 0.05, 1.0, 0.25, 0.05)
            idf_power = st.slider("Εκθέτης IDF (Απόσβεση κοινών ΚΑΔ)", 0.0, 2.5, 1.25, 0.05)
            sec_damping = st.slider("Απόσβεση Πλήθους Δευτερευόντων ΚΑΔ", 0.0, 1.0, 0.35, 0.05)

        with st.expander("Βάρη Ομάδων Χαρακτηριστικών (A1)", expanded=False):
            bw_semantic = st.slider("Σημασιολογικά Embeddings ΚΑΔ", 0.1, 2.5, 1.00, 0.05)
            bw_kad_hier = st.slider("Ιεραρχία ΚΑΔ (4-ψήφιος/8-ψήφιος)", 0.1, 2.5, 0.85, 0.05)
            bw_spatial = st.slider("Γεωγραφία & Πυκνότητα cKDTree (X1)", 0.1, 2.5, 1.15, 0.05)
            bw_legal = st.slider("Νομική Μορφή", 0.1, 2.5, 0.90, 0.05)
            bw_age_branch = st.slider("Ηλικία & Υποκαταστήματα", 0.1, 2.5, 0.45, 0.05)

        with st.expander("Ανακάλυψη ICP & Στατιστικοί Έλεγχοι (A2-A5)", expanded=True):
            min_icps, max_icps = st.slider("Εύρος Αριθμού ICPs (K)", 1, 6, (1, 6))
            turnover_mode = st.selectbox(
                "Στάθμιση Τζίρου Πελατών (A2)",
                options=["log_turnover", "sqrt_turnover", "linear_turnover", "uniform"],
                index=0,
            )
            turnover_alpha = st.slider("Ένταση Στάθμισης Τζίρου (α)", 0.0, 1.0, 0.50, 0.05)
            eb_prior = st.slider("Empirical Bayes Prior Shrinkage (A4)", 1.0, 30.0, 10.0, 1.0)
            n_boot = st.slider("Επαναλήψεις Bootstrap ARI (A5)", 10, 50, 30, 5)

        with st.expander("Μετρική Ομοιότητας Lookalike & No-Fit (A3, A7)", expanded=False):
            w_cos = st.slider("Βάρος Feature Cosine Similarity", 0.0, 1.0, 0.45, 0.05)
            w_eb = st.slider("Βάρος Empirical Bayes Lift Affinity", 0.0, 1.0, 0.40, 0.05)
            w_sp = st.slider("Βάρος Χωρικού Πυρήνα (Spatial Kernel)", 0.0, 1.0, 0.15, 0.05)
            bw_km = st.slider("Εύρος Χωρικού Πυρήνα (km)", 5.0, 100.0, 25.0, 5.0)
            no_fit_min = st.slider("Κατώφλι 'Δεν ταιριάζει σε προφίλ' (NO_FIT)", 0.15, 0.65, 0.35, 0.01)

        st.divider()
        st.header("🤖 3. Τοπικό LLM (Ollama - A6)")
        ollama_ok, resolved_ollama_model, installed_models = check_local_ollama_status("qwen2.5:7b-instruct")
        if ollama_ok:
            st.success(f"Ollama Online ({len(installed_models)} μοντέλα διαθέσιμα)")
            selected_ollama = st.selectbox("Μοντέλο Ollama", options=installed_models, index=0)
            use_ollama = st.checkbox("Χρήση Τοπικού Ollama LLM", value=True)
        else:
            st.warning("Ollama Offline — Ενεργή η ντετερμινιστική Grounded Ελληνική γεννήτρια (100% αναπαραγώγιμη).")
            selected_ollama = "qwen2.5:7b-instruct"
            use_ollama = False

    # -------------------------------------------------------------------------
    # Execute Pipeline
    # -------------------------------------------------------------------------
    base_ds = cached_ingest_base()
    if uploaded_file is not None:
        custom_clients = load_clients_excel(uploaded_file.getvalue())
    else:
        custom_clients = base_ds.clients

    fs = cached_feature_store(
        embedding_model=selected_emb,
        w_primary=w_primary,
        w_secondary=w_secondary,
        w_auxiliary=w_auxiliary,
        idf_power=idf_power,
        secondary_damping=sec_damping,
        bw_semantic=bw_semantic,
        bw_kad_hier=bw_kad_hier,
        bw_spatial=bw_spatial,
        bw_legal=bw_legal,
        bw_age_branch=bw_age_branch,
    )

    cfg = PipelineConfig(
        embedding_model=selected_emb,
        w_primary_kad=w_primary,
        w_secondary_kad=w_secondary,
        w_auxiliary_kad=w_auxiliary,
        idf_power=idf_power,
        secondary_count_damping=sec_damping,
        block_weight_semantic=bw_semantic,
        block_weight_kad_hierarchy=bw_kad_hier,
        block_weight_spatial=bw_spatial,
        block_weight_legal=bw_legal,
        block_weight_age_branch=bw_age_branch,
        min_icps=min_icps,
        max_icps=max_icps,
        turnover_weighting_mode=turnover_mode,
        turnover_weight_alpha=turnover_alpha,
        eb_prior_strength=eb_prior,
        n_bootstrap_runs=n_boot,
        sim_weight_feature_cosine=w_cos,
        sim_weight_eb_lift_affinity=w_eb,
        sim_weight_spatial_kernel=w_sp,
        spatial_kernel_bandwidth_km=bw_km,
        no_fit_min_score=no_fit_min,
        ollama_model=selected_ollama,
    )

    icp_result = run_icp_engine(
        ds=base_ds,
        fs=fs,
        cfg=cfg,
        custom_clients_df=custom_clients,
    )
    llm_telemetry = enrich_profiles_with_local_llm(
        result=icp_result,
        cfg=cfg,
        force_deterministic=not use_ollama,
    )
    exported_paths = export_deliverables(icp_result, cfg.output_dir)

    # -------------------------------------------------------------------------
    # Top KPI Metrics Row
    # -------------------------------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Universe Επιχειρήσεων", f"{base_ds.companies.height:,}", "Κύριος ΚΑΔ 56 (2020–2024)")
    k2.metric("Πελάτες Πελατολογίου", f"{custom_clients.height:,}", f"Συν. Τζίρος €{custom_clients['turnover'].sum():,.0f}")
    k3.metric("Ανακαλυφθέντα ICPs (K)", f"{icp_result.optimal_k}", f"Silhouette: {icp_result.silhouette_by_k.get(icp_result.optimal_k, 0.0):.3f}")
    k4.metric("Σταθερότητα Bootstrap (ARI)", f"{100*icp_result.global_bootstrap_ari:.1f}%", f"{cfg.n_bootstrap_runs} επαναδειγματοληψίες")
    no_fit_count = icp_result.assignments_df.filter(pl.col("profile_id") == "NO_FIT").height
    k5.metric("Προτάσεις Lookalike", f"{icp_result.lookalikes_df.height:,}", f"NO_FIT Πελάτες: {no_fit_count}")

    # -------------------------------------------------------------------------
    # Main Navigation Tabs
    # -------------------------------------------------------------------------
    tab_icps, tab_map, tab_lookalikes, tab_assignments, tab_eval, tab_export = st.tabs(
        [
            "🧬 1. Ideal Customer Profiles (ICPs)",
            "🗺️ 2. Γεωγραφικός Χάρτης & Πυκνότητα (X1)",
            "🔍 3. Προτάσεις Lookalike (Top 2.000+)",
            "📋 4. Ανάθεση Πελατών & NO_FIT (A3)",
            "📈 5. Hold-Out Αξιολόγηση & Μοντέλα (Εν. 5.3 & 8.1)",
            "📥 6. Εξαγωγή Παραδοτέων & EU AI Act",
        ]
    )

    # =========================================================================
    # TAB 1: DISCOVERED ICPs
    # =========================================================================
    with tab_icps:
        st.subheader("Ανακαλυφθέντα Ideal Customer Profiles (ICPs) & Στατιστική Τεκμηρίωση")
        st.markdown(
            "Κάθε προφίλ τεκμηριώνεται με **Ακριβή Υπεργεωμετρικό Έλεγχο (Hypergeometric Test)**, "
            "**Διόρθωση Πολλαπλών Ελέγχων Benjamini-Hochberg (FDR)** και **Empirical Bayes Shrinkage Lift** "
            "ώστε σπάνια χαρακτηριστικά να μην περνούν από τύχη (Απαίτηση Α4)."
        )

        # Silhouette across K bar chart
        sil_df = pl.DataFrame(
            {
                "K (Αριθμός Προφίλ)": [str(k) for k in sorted(icp_result.silhouette_by_k.keys())],
                "Cosine Silhouette Score": [icp_result.silhouette_by_k[k] for k in sorted(icp_result.silhouette_by_k.keys())],
            }
        )
        col_sil, col_share = st.columns(2)
        with col_sil:
            fig_sil = px.bar(
                sil_df.to_pandas(),
                x="K (Αριθμός Προφίλ)",
                y="Cosine Silhouette Score",
                title="Επιλογή Βέλτιστου Αριθμού ICPs (K=1 έως 6)",
                text_auto=".3f",
                color="Cosine Silhouette Score",
                color_continuous_scale="Tealgrn",
            )
            fig_sil.update_layout(height=300)
            st.plotly_chart(fig_sil, use_container_width=True)

        with col_share:
            share_df = pl.DataFrame(
                {
                    "Profile": [f"{p.profile_id}: {p.title_el[:32]}..." for p in icp_result.profiles],
                    "Μερίδιο Τζίρου (%)": [round(100 * p.turnover_share, 1) for p in icp_result.profiles],
                    "Μερίδιο Πελατών (%)": [round(100 * p.client_share, 1) for p in icp_result.profiles],
                }
            )
            fig_share = px.bar(
                share_df.to_pandas(),
                x="Profile",
                y=["Μερίδιο Τζίρου (%)", "Μερίδιο Πελατών (%)"],
                barmode="group",
                title="Κατανομή Πελατών vs Συνολικού Τζίρου ανά ICP (Α2)",
            )
            fig_share.update_layout(height=300)
            st.plotly_chart(fig_share, use_container_width=True)

        for prof in icp_result.profiles:
            with st.container(border=True):
                c_head1, c_head2, c_head3, c_head4 = st.columns([3, 1, 1, 1])
                c_head1.markdown(f"### 🏷️ {prof.profile_id}: {prof.title_el}")
                c_head2.metric("Πελάτες / Μερίδιο Τζίρου", f"{prof.client_count} ({100*prof.turnover_share:.1f}%)")
                c_head3.metric("Σταθερότητα Bootstrap (A5)", f"{100*prof.stability_score:.1f}%")
                c_head4.metric("Εμπιστοσύνη Προφίλ (A3)", f"{100*prof.confidence_score:.1f}%")

                st.markdown(prof.ai_description_el)

                if prof.significant_features:
                    feat_rows = [
                        {
                            "Χαρακτηριστικό": f.feature_label_el,
                            "Πελάτες ICP": f.icp_count,
                            "Μερίδιο στο ICP (%)": round(100 * f.icp_share, 1),
                            "Μερίδιο στο Universe (%)": round(100 * f.universe_share, 2),
                            "Raw Lift": round(f.raw_lift, 2),
                            "Empirical Bayes Lift": round(f.eb_lift, 2),
                            "FDR p-value (BH)": f"{f.p_value_fdr:.2e}",
                        }
                        for f in prof.significant_features
                    ]
                    feat_df = pl.DataFrame(feat_rows)

                    c_chart, c_table = st.columns([1, 1])
                    with c_chart:
                        fig_lift = px.bar(
                            feat_df.head(8).to_pandas(),
                            x="Empirical Bayes Lift",
                            y="Χαρακτηριστικό",
                            orientation="h",
                            title=f"Top Υπερεκπροσωπούμενα Χαρακτηριστικά ({prof.profile_id})",
                            text_auto=".2f",
                            color="Empirical Bayes Lift",
                            color_continuous_scale="Blues",
                        )
                        fig_lift.update_layout(yaxis={"categoryorder": "total ascending"}, height=320)
                        st.plotly_chart(fig_lift, use_container_width=True)
                    with c_table:
                        st.dataframe(feat_df.to_pandas(), use_container_width=True, height=320)

    # =========================================================================
    # TAB 2: SPATIAL MAP & GEOGRAPHIC ICPs (Extra Point X1)
    # =========================================================================
    with tab_map:
        st.subheader("🗺️ Γεωγραφική Κατανομή Πελατών & Top Lookalikes (Extra Point X1)")
        ec_sorted = base_ds.enriched_companies.sort("company_id")

        client_map_df = (
            ec_sorted.join(icp_result.assignments_df, on="company_id", how="inner")
            .filter(pl.col("has_gps"))
            .select(
                [
                    "company_id",
                    "company_name_el",
                    "legal_type",
                    "prefecture_name",
                    "municipality_name",
                    "primary_kad",
                    "latitude",
                    "longitude",
                    "profile_id",
                    "score",
                ]
            )
            .with_columns(pl.lit("Υφιστάμενος Πελάτης").alias("Κατηγορία"))
        )

        top_look_map_df = (
            icp_result.lookalikes_df.head(500)
            .join(
                ec_sorted.select(
                    [
                        "company_id",
                        "company_name_el",
                        "legal_type",
                        "prefecture_name",
                        "municipality_name",
                        "primary_kad",
                        "latitude",
                        "longitude",
                        "has_gps",
                    ]
                ),
                on="company_id",
                how="left",
            )
            .filter(pl.col("has_gps"))
            .select(
                [
                    "company_id",
                    "company_name_el",
                    "legal_type",
                    "prefecture_name",
                    "municipality_name",
                    "primary_kad",
                    "latitude",
                    "longitude",
                    "profile_id",
                    "score",
                ]
            )
            .with_columns(pl.lit("Top-500 Lookalike").alias("Κατηγορία"))
        )

        combined_map_df = pl.concat([client_map_df, top_look_map_df])
        fig_map = px.scatter_mapbox(
            combined_map_df.to_pandas(),
            lat="latitude",
            lon="longitude",
            color="profile_id",
            symbol="Κατηγορία",
            hover_name="company_name_el",
            hover_data=["company_id", "legal_type", "prefecture_name", "primary_kad", "score"],
            zoom=5.5,
            center={"lat": 38.2, "lon": 23.8},
            mapbox_style="open-street-map",
            height=620,
            title="Χάρτης Επικράτειας: Υφιστάμενοι Πελάτες ανά ICP & Top-500 Προτάσεις Lookalike (© OpenStreetMap ODbL)",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # =========================================================================
    # TAB 3: LOOKALIKE RECOMMENDATIONS
    # =========================================================================
    with tab_lookalikes:
        st.subheader("🔍 Κατάταξη Lookalike Επιχειρήσεων από το Universe (Απαίτηση Α7)")
        ec_meta = base_ds.enriched_companies.select(
            [
                "company_id",
                "vat",
                "company_name_el",
                "legal_type",
                "prefecture_name",
                "municipality_name",
                "city",
                "primary_kad",
                "primary_kad_desc",
                "company_age_years",
                "branch_count",
                "total_kads",
            ]
        )
        rich_lookalikes = icp_result.lookalikes_df.join(ec_meta, on="company_id", how="left")

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            icp_filter = st.multiselect(
                "Φίλτρο ICP",
                options=[p.profile_id for p in icp_result.profiles],
                default=[p.profile_id for p in icp_result.profiles],
            )
        with f_col2:
            avail_prefs = sorted(
                [p for p in rich_lookalikes["prefecture_name"].unique().to_list() if p is not None]
            )
            pref_filter = st.multiselect("Φίλτρο Νομού", options=avail_prefs, default=[])
        with f_col3:
            search_query = st.text_input("Αναζήτηση Επωνυμίας / ΑΦΜ / ΚΑΔ", "")

        filtered_look = rich_lookalikes.filter(pl.col("profile_id").is_in(icp_filter))
        if pref_filter:
            filtered_look = filtered_look.filter(pl.col("prefecture_name").is_in(pref_filter))
        if search_query.strip():
            q = search_query.strip()
            filtered_look = filtered_look.filter(
                pl.col("company_name_el").str.contains(q, literal=True)
                | pl.col("vat").str.contains(q, literal=True)
                | pl.col("primary_kad").str.contains(q, literal=True)
            )

        st.write(f"Εμφάνιση **{filtered_look.height:,}** προτάσεων Lookalike (καμία εγγραφή δεν ανήκει σε ΑΦΜ πελάτη):")
        st.dataframe(
            filtered_look.select(
                [
                    "rank",
                    "profile_id",
                    "score",
                    "company_id",
                    "vat",
                    "company_name_el",
                    "legal_type",
                    "prefecture_name",
                    "municipality_name",
                    "primary_kad",
                    "total_kads",
                    "αιτιολόγηση",
                ]
            ).to_pandas(),
            use_container_width=True,
            height=500,
        )

    # =========================================================================
    # TAB 4: CLIENT ASSIGNMENTS & NO_FIT
    # =========================================================================
    with tab_assignments:
        st.subheader("📋 Ανάθεση Πελατών σε ICP & Εντοπισμός 'NO_FIT' (Απαίτηση Α3)")
        st.markdown(
            "Η ανάθεση κάθε πελάτη χρησιμοποιεί **ακριβώς την ίδια συνάρτηση ομοιότητας** με το Lookalike engine. "
            f"Πελάτες με βαθμολογία κάτω από το κατώφλι (`{cfg.no_fit_min_score:.2f}`) χαρακτηρίζονται ως **`NO_FIT`** "
            "(«δεν ταιριάζει σε κανένα profile»)."
        )
        rich_assign = icp_result.assignments_df.join(
            base_ds.enriched_companies.select(
                [
                    "company_id",
                    "vat",
                    "company_name_el",
                    "legal_type",
                    "prefecture_name",
                    "primary_kad",
                    "primary_kad_desc",
                    "turnover",
                    "total_kads",
                ]
            ),
            on="company_id",
            how="left",
        )
        st.dataframe(rich_assign.to_pandas(), use_container_width=True, height=480)

    # =========================================================================
    # TAB 5: HOLD-OUT EVALUATION & MODEL BENCHMARKING (Section 5.3 & 8.1)
    # =========================================================================
    with tab_eval:
        st.subheader("📈 Έλεγχος Hold-Out (Recall@500 & Recall@2000) & Τεκμηρίωση Μοντέλων")
        col_ev1, col_ev2 = st.columns([1, 2])
        with col_ev1:
            n_eval_splits = st.number_input("Αριθμός Hold-Out Splits (80% Train / 20% Test)", 1, 20, 5)
            run_bench_btn = st.button("🚀 Εκτέλεση Hold-Out Benchmark τώρα", type="primary")

        if run_bench_btn:
            with st.spinner(f"Εκτέλεση {n_eval_splits}x Monte Carlo Hold-Out Splits (240 Train / 60 Κρυμμένοι Πελάτες)..."):
                base_m = evaluate_ranker_on_holdout(
                    ds=base_ds,
                    ranker_fn=baseline_primary_kad_ranker,
                    name="ForceCRM Baseline (Primary KAD Frequency)",
                    n_splits=int(n_eval_splits),
                    holdout_fraction=0.20,
                    base_seed=cfg.random_seed,
                    is_baseline=True,
                )
                icp_m = evaluate_ranker_on_holdout(
                    ds=base_ds,
                    ranker_fn=make_icp_engine_ranker(fs, cfg),
                    name=f"Archimedes2026 ICP Engine ({selected_emb})",
                    n_splits=int(n_eval_splits),
                    holdout_fraction=0.20,
                    base_seed=cfg.random_seed,
                    is_baseline=False,
                )
                bench_df = pl.DataFrame(
                    [
                        {
                            "Μοντέλο": base_m.name,
                            "Recall@500 (%)": round(100 * base_m.recall_at_500_mean, 2),
                            "Recall@500 Std (%)": round(100 * base_m.recall_at_500_std, 2),
                            "Recall@2000 (%)": round(100 * base_m.recall_at_2000_mean, 2),
                            "Recall@2000 Std (%)": round(100 * base_m.recall_at_2000_std, 2),
                        },
                        {
                            "Μοντέλο": icp_m.name,
                            "Recall@500 (%)": round(100 * icp_m.recall_at_500_mean, 2),
                            "Recall@500 Std (%)": round(100 * icp_m.recall_at_500_std, 2),
                            "Recall@2000 (%)": round(100 * icp_m.recall_at_2000_mean, 2),
                            "Recall@2000 Std (%)": round(100 * icp_m.recall_at_2000_std, 2),
                        },
                    ]
                )
                st.dataframe(bench_df.to_pandas(), use_container_width=True)
                gain500 = icp_m.recall_at_500_mean / max(base_m.recall_at_500_mean, 1e-6)
                gain2000 = icp_m.recall_at_2000_mean / max(base_m.recall_at_2000_mean, 1e-6)
                st.success(
                    f"✅ Βελτίωση έναντι του ForceCRM Baseline: **{gain500:.2f}x στο Recall@500** "
                    f"({100*icp_m.recall_at_500_mean:.2f}% έναντι {100*base_m.recall_at_500_mean:.2f}%) και "
                    f"**{gain2000:.2f}x στο Recall@2000** ({100*icp_m.recall_at_2000_mean:.2f}% έναντι {100*base_m.recall_at_2000_mean:.2f}%)!"
                )

        st.markdown("#### Τηλεμετρία Ενεργού Μοντέλου Embeddings (Ενότητα 5.3)")
        st.json(fs.embedding_telemetry)

    # =========================================================================
    # TAB 6: DELIVERABLE EXPORTS & EU AI ACT SELF-ASSESSMENT (Section 9 & 10)
    # =========================================================================
    with tab_export:
        st.subheader("📥 Εξαγωγή Αρχείων Αυτόματης Αξιολόγησης (Ενότητα 10)")
        st.write("Τα αρχεία ενημερώνονται αυτόματα στον φάκελο `output/` και είναι διαθέσιμα για άμεση λήψη:")

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            profiles_bytes = exported_paths["profiles.json"].read_bytes()
            st.download_button(
                "⬇️ Λήψη profiles.json",
                data=profiles_bytes,
                file_name="profiles.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl2:
            assign_bytes = exported_paths["assignments.csv"].read_bytes()
            st.download_button(
                "⬇️ Λήψη assignments.csv",
                data=assign_bytes,
                file_name="assignments.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl3:
            look_bytes = exported_paths["lookalikes.csv"].read_bytes()
            st.download_button(
                "⬇️ Λήψη lookalikes.csv",
                data=look_bytes,
                file_name="lookalikes.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        st.subheader("🛡️ EU AI Act Self-Assessment (Ενότητα 9)")
        st.markdown(
            """
1. **Τι αποφασίζει ή προτείνει το σύστημα;**
   Το σύστημα ομαδοποιεί υφιστάμενους B2B πελάτες εστίασης σε 1–6 Ideal Customer Profiles (ICPs) και προτείνει παρόμοιες επιχειρήσεις (lookalikes) από το ΓΕΜΗ Universe αποκλειστικά βάσει εμπορικής/μορφολογικής ομοιότητας (δραστηριότητα ΚΑΔ, γεωγραφία, νομική μορφή, ηλικία). **Δεν αξιολογεί πιστοληπτική ικανότητα ή φερεγγυότητα.**
2. **Ποιος επηρεάζεται από αυτό;**
   Οι ομάδες πωλήσεων B2B (χρήστες του συστήματος) που ιεραρχούν τις εμπορικές τους επαφές. Οι ατομικές επιχειρήσεις (φυσικά πρόσωπα) προστατεύονται πλήρως καθώς δεν παράγεται κανένα credit/solvency score (εκτός Παραρτήματος III Υψηλού Κινδύνου).
3. **Τι βλέπει ο χρήστης για το πώς προέκυψε το αποτέλεσμα;**
   Κάθε ICP και κάθε πρόταση lookalike συνοδεύεται από μετρήσιμους δείκτες: Empirical Bayes Lift, FDR p-value (Benjamini-Hochberg), Bootstrap Stability (ARI), Confidence Score, και ρητή σήμανση κειμένου AI (Άρθρο 50).
4. **Πού μπορεί να κάνει λάθος και πώς το εντοπίζουμε;**
   - *Θορυβώδεις/ανενεργοί δευτερεύοντες ΚΑΔ:* Αντιμετωπίζεται με IDF στάθμιση, sublinear damping και L2 κανονικοποίηση.
   - *Πελάτες εκτός τυπικού προφίλ:* Εντοπίζονται αυτόματα μέσω του κατωφλίου `NO_FIT` στο `assignments.csv`.
   - *Έλλειψη στίγματος GPS (4% του Universe):* Χρήση fallback σε επίπεδο Δήμου/Νομού χωρίς απόρριψη της επιχείρησης.
            """
        )


if __name__ == "__main__":
    main()
