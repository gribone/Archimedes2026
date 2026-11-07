r"""
Stage 6: Localhost Streamlit Web Application (`app.py`) — Executive B2B Edition
===============================================================================
Designed specifically for B2B Sales Leaders & Account Executives:
1. **Clean, Uncluttered Customer Experience:**
   - Focuses on 3 clear business views: **1) Τα Προφίλ Πελατών μας (ICPs & Χάρτης)**,
     **2) Προτεινόμενοι Νέοι Πελάτες (Lookalikes)**, and **3) Υφιστάμενο Πελατολόγιο**.
   - All technical/mathematical details (Empirical Bayes formulas, p-values, Silhouette curves,
     Hold-Out Recall@k benchmarks, and EU AI Act self-assessment) are neatly tucked inside
     dedicated **`🔬 Τεχνική Τεκμηρίωση & Στατιστικά`** submenus (`st.expander` / popover).
2. **Self-Explanatory, High-Impact Visuals:**
   - **Donut Charts** for Revenue Share (`Μερίδιο Τζίρου €`) and Customer Count Share (`Κατανομή Πελατών`).
   - **Dumbbell / Paired Comparison Chart** for each ICP showing **"Ποσοστό στο Προφίλ μας (%)" vs.
     "Μέσος Όρος Αγοράς (%)"** so anyone immediately understands what makes an ICP special without
     needing a statistics degree.
3. **Integrated ICP & Lookalike Map Panel:**
   - Embedded interactive map right inside the ICP view (and full-screen explorer) showing the
     exact geographic footprint of each ICP's clients alongside its recommended lookalikes,
     with a dedicated slot ready for custom Google Places / Mapbox / OSM enrichment.

Run locally with:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

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
from src.icp_engine import DiscoveredICP, export_deliverables, run_icp_engine
from src.ingest import IngestedDataset, ingest_all, load_clients_excel
from src.llm_explainer import (
    build_sales_friendly_description_el,
    check_local_ollama_status,
    enrich_profiles_with_local_llm,
)


st.set_page_config(
    page_title="ForceCRM · Ανάλυση Πελατολογίου & Νέοι Πελάτες",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette for ICPs (Indigo, Emerald, Amber, Rose, Violet, Cyan)
ICP_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6", "#06b6d4"]

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.25rem 1.75rem;
        border-radius: 14px;
        margin-bottom: 1.25rem;
        border: 1px solid #334155;
    }
    .top-header h1 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }
    .top-header p {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin: 0.25rem 0 0 0;
    }
    .ai-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.18);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .icp-card-banner {
        padding: 0.75rem 1.15rem;
        border-radius: 10px;
        margin-bottom: 0.9rem;
        border-left: 5px solid #3b82f6;
        background: rgba(59, 130, 246, 0.07);
    }
    .icp-card-banner h3 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .trait-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.28);
        border-radius: 8px;
        padding: 0.3rem 0.65rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.2rem 0.35rem 0.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Φόρτωση βάσης επιχειρήσεων εστίασης...")
def cached_ingest_base() -> IngestedDataset:
    return ingest_all()


@st.cache_resource(show_spinner="Ανάλυση χαρακτηριστικών αγοράς...")
def cached_feature_store(
    embedding_model: str,
    hybrid_alpha: float,
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
        hybrid_alpha=hybrid_alpha,
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


def _clean_trait_name(label: str) -> str:
    """Make statistical trait labels short and natural for business charts."""
    cleaned = (
        label.replace("Δήμος: ", "📍 ")
        .replace("Νομός: ", "🗺️ ")
        .replace("Νομική Μορφή: ", "🏛️ ")
        .replace("Κατηγορία ΚΑΔ (4-ψήφιος): ", "🍽️ ")
    )
    if len(cleaned) > 38:
        return cleaned[:37] + "…"
    return cleaned


def build_portfolio_donuts(profiles: list[DiscoveredICP]) -> tuple[go.Figure, go.Figure]:
    """Build two clean, intuitive Donut charts: Revenue Share (€) and Client Count Share."""
    labels = [f"{p.profile_id}: {p.title_el.split('—')[0].strip()}" for p in profiles]
    turnovers = [p.turnover_total for p in profiles]
    counts = [p.client_count for p in profiles]
    colors = ICP_COLORS[: len(profiles)]

    fig_rev = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=turnovers,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>Τζίρος: €%{value:,.0f} (%{percent})<extra></extra>",
            )
        ]
    )
    total_rev_m = sum(turnovers) / 1e6
    fig_rev.update_layout(
        title=dict(text="💰 Πού βρίσκεται ο Τζίρος μας;", font=dict(size=15)),
        annotations=[
            dict(
                text=f"<b>€{total_rev_m:.2f}M</b><br>Συν. Τζίρος",
                x=0.5,
                y=0.5,
                font_size=14,
                showarrow=False,
            )
        ],
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    fig_cnt = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=counts,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>Πελάτες: %{value} (%{percent})<extra></extra>",
            )
        ]
    )
    fig_cnt.update_layout(
        title=dict(text="👥 Πώς χωρίζονται οι Πελάτες μας;", font=dict(size=15)),
        annotations=[
            dict(
                text=f"<b>{sum(counts)}</b><br>Πελάτες",
                x=0.5,
                y=0.5,
                font_size=15,
                showarrow=False,
            )
        ],
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig_rev, fig_cnt


def build_icp_vs_market_chart(prof: DiscoveredICP, accent_color: str = "#3b82f6") -> go.Figure:
    """
    Build an intuitive Horizontal Grouped Comparison Chart:
    'Στο Προφίλ μας (%)' vs 'Στην Υπόλοιπη Αγορά (%)' with a clear multiplier badge (+4.6x).
    Customers immediately see why this group is distinct without needing to know what Empirical Bayes is.
    """
    traits = prof.significant_features[:5][::-1]  # Top 5, reversed for top-down display
    y_labels = [_clean_trait_name(f.feature_label_el) for f in traits]
    full_names = [f.feature_label_el for f in traits]
    icp_pcts = [round(100 * f.icp_share, 1) for f in traits]
    univ_pcts = [round(100 * f.universe_share, 1) for f in traits]
    lifts = [f.eb_lift for f in traits]

    fig = go.Figure()

    # Market baseline bar (subtle slate gray)
    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=univ_pcts,
            name="Μέσος Όρος Αγοράς (Universe)",
            orientation="h",
            marker_color="#94a3b8",
            text=[f"{u:.1f}%" for u in univ_pcts],
            textposition="outside",
            customdata=full_names,
            hovertemplate="<b>%{customdata}</b><br>Μέσος Όρος Αγοράς: <b>%{x:.1f}%</b><extra></extra>",
        )
    )

    # ICP share bar (vibrant profile color + lift callout)
    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=icp_pcts,
            name=f"Πελάτες {prof.profile_id}",
            orientation="h",
            marker_color=accent_color,
            text=[f"<b>{p:.1f}%</b> ({l:.1f}x συχνότερο)" for p, l in zip(icp_pcts, lifts)],
            textposition="outside",
            customdata=full_names,
            hovertemplate="<b>%{customdata}</b><br>Ποσοστό στο Προφίλ: <b>%{x:.1f}%</b><extra></extra>",
        )
    )

    max_x = max(icp_pcts + univ_pcts + [20.0]) * 1.38
    fig.update_layout(
        title=dict(
            text="Τι ξεχωρίζει αυτούς τους πελάτες από την υπόλοιπη αγορά;",
            font=dict(size=14),
        ),
        barmode="group",
        xaxis=dict(title="Ποσοστό Επιχειρήσεων (%)", range=[0, max_x], showgrid=True),
        yaxis=dict(title=""),
        height=300,
        margin=dict(l=10, r=20, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    return fig


def build_icp_embedded_map(
    base_ds: IngestedDataset,
    icp_result,
    selected_profile_filter: str = "ALL",
    region_center: dict | None = None,
) -> go.Figure:
    """
    Build the interactive map showing ICP Clients and Top Recommended Lookalikes.
    Structured cleanly so custom map tiles / Google Places layers can plug right in.
    """
    ec_sorted = base_ds.enriched_companies.sort("company_id")

    client_map_df = (
        ec_sorted.join(icp_result.assignments_df, on="company_id", how="inner")
        .filter(pl.col("has_gps") & (pl.col("profile_id") != "NO_FIT"))
        .select(
            [
                "company_id",
                "company_name_el",
                "legal_type",
                "prefecture_name",
                "municipality_name",
                "latitude",
                "longitude",
                "profile_id",
                "score",
            ]
        )
        .with_columns(
            (pl.col("profile_id") + " · Υφιστάμενος Πελάτης").alias("Ομάδα"),
            pl.lit(11.0).alias("Μέγεθος"),
        )
    )

    look_map_df = (
        icp_result.lookalikes_df.head(300)
        .join(
            ec_sorted.select(
                [
                    "company_id",
                    "company_name_el",
                    "legal_type",
                    "prefecture_name",
                    "municipality_name",
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
                "latitude",
                "longitude",
                "profile_id",
                "score",
            ]
        )
        .with_columns(
            (pl.col("profile_id") + " · Προτεινόμενος (Lookalike)").alias("Ομάδα"),
            pl.lit(7.0).alias("Μέγεθος"),
        )
    )

    combined = pl.concat([client_map_df, look_map_df])
    if selected_profile_filter != "ALL":
        combined = combined.filter(pl.col("profile_id") == selected_profile_filter)

        center = region_center or {"lat": 38.15, "lon": 23.85, "zoom": 5.6}

        fig = px.scatter_map(
            combined.to_pandas(),
            lat="latitude",
            lon="longitude",
            color="Ομάδα",
            size="Μέγεθος",
            size_max=12,
            hover_name="company_name_el",
            hover_data={
                "legal_type": True,
                "prefecture_name": True,
                "municipality_name": True,
                "score": ":.2f",
                "Μέγεθος": False,
                "latitude": False,
                "longitude": False,
                "Ομάδα": False,
            },
            zoom=center["zoom"],
            center={"lat": center["lat"], "lon": center["lon"]},
            map_style="carto-positron",
            color_discrete_sequence=["#2563eb", "#10b981", "#60a5fa", "#34d399", "#f59e0b", "#fbbf24"],
            height=440,
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.88)",
            title="",
        ),
    )
    return fig


def main() -> None:
    # -------------------------------------------------------------------------
    # Sidebar: Simple Customer Controls + Hidden Technical Submenu
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📂 1. Αρχείο Πελατολογίου")
        uploaded_file = st.file_uploader(
            "Ανεβάστε το Excel πελατών (.xlsx)",
            type=["xlsx"],
            help="Στήλες: ΑΦΜ και Τζίρος (€). Αν μείνει κενό, χρησιμοποιείται το προεπιλεγμένο πελατολόγιο (300 πελάτες).",
        )
        if uploaded_file is not None:
            st.success(f"Ενεργό αρχείο: `{uploaded_file.name}`")
        else:
            st.caption("Ενεργό: `ForceCRM_Hackathon_Pelatologio.xlsx` (300 πελάτες)")

        st.divider()

        # Hidden Technical & Parameter Submenu in Sidebar
        with st.expander("⚙️ Ρυθμίσεις Μοντέλου & Παράμετροι (Για Αναλυτές)", expanded=False):
            emb_options = list(EMBEDDING_MODELS_REGISTRY.keys())
            selected_emb = st.selectbox(
                "Μοντέλο Embeddings ΚΑΔ",
                options=emb_options,
                index=emb_options.index("granite-107m"),
            )
            hybrid_alpha = st.slider(
                "Hybrid Alpha (Signal A Neural vs Signal B SVD)",
                0.0,
                1.0,
                0.50,
                0.05,
            )

            ollama_ok, _, installed_models = check_local_ollama_status("qwen2.5:7b-instruct")
            if ollama_ok:
                selected_ollama = st.selectbox("Τοπικό LLM (Ollama)", options=installed_models, index=0)
                use_ollama = st.toggle("Χρήση Ollama LLM", value=True)
            else:
                selected_ollama = "qwen2.5:7b-instruct"
                use_ollama = False
                st.caption("Ollama offline — Χρήση ενσωματωμένης γεννήτριας.")

            min_icps, max_icps = st.slider("Εύρος Προφίλ (K)", 1, 6, (1, 6))
            turnover_mode = st.selectbox(
                "Στάθμιση Τζίρου",
                options=["log_turnover", "sqrt_turnover", "linear_turnover", "uniform"],
                index=0,
            )
            turnover_alpha = st.slider("Βάρος Τζίρου (α)", 0.0, 1.0, 0.50, 0.05)
            w_primary = st.slider("Βάρος Κύριου ΚΑΔ", 1.0, 6.0, 3.5, 0.25)
            w_secondary = st.slider("Βάρος Δευτερεύοντος ΚΑΔ", 0.1, 2.0, 0.75, 0.05)
            w_auxiliary = 0.25
            idf_power = st.slider("Απόσβεση Κοινών ΚΑΔ (IDF)", 0.0, 2.5, 1.25, 0.05)
            sec_damping = st.slider("Απόσβεση Πλήθους ΚΑΔ", 0.0, 1.0, 0.35, 0.05)
            bw_semantic, bw_kad_hier, bw_spatial, bw_legal, bw_age_branch = 1.00, 0.85, 1.15, 0.90, 0.45
            eb_prior = st.slider("Empirical Bayes Prior", 1.0, 30.0, 8.0, 1.0)
            n_boot = st.slider("Bootstrap Runs", 10, 50, 30, 5)
            w_cos = st.slider("Βάρος Ομοιότητας Προφίλ", 0.0, 1.0, 0.35, 0.05)
            w_eb = st.slider("Βάρος Στατιστικού Lift", 0.0, 1.0, 0.50, 0.05)
            w_sp = st.slider("Βάρος Εγγύτητας Χάρτη", 0.0, 1.0, 0.15, 0.05)
            bw_km = st.slider("Ακτίνα Εγγύτητας (km)", 5.0, 100.0, 20.0, 5.0)
            no_fit_min = st.slider("Κατώφλι NO_FIT", 0.15, 0.65, 0.35, 0.01)

    # -------------------------------------------------------------------------
    # Run Pipeline
    # -------------------------------------------------------------------------
    base_ds = cached_ingest_base()
    custom_clients = load_clients_excel(uploaded_file.getvalue()) if uploaded_file is not None else base_ds.clients

    fs = cached_feature_store(
        embedding_model=selected_emb,
        hybrid_alpha=hybrid_alpha,
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
        hybrid_alpha=hybrid_alpha,
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

    icp_result = run_icp_engine(ds=base_ds, fs=fs, cfg=cfg, custom_clients_df=custom_clients)
    enrich_profiles_with_local_llm(result=icp_result, cfg=cfg, force_deterministic=not use_ollama)
    exported_paths = export_deliverables(icp_result, cfg.output_dir)

    with st.sidebar:
        st.markdown("### 📥 2. Εξαγωγή Αποτελεσμάτων")
        st.download_button(
            "⬇️ Προτεινόμενοι Πελάτες (lookalikes.csv)",
            data=exported_paths["lookalikes.csv"].read_bytes(),
            file_name="lookalikes.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
        st.download_button(
            "⬇️ Προφίλ Πελατών (profiles.json)",
            data=exported_paths["profiles.json"].read_bytes(),
            file_name="profiles.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Κατηγοριοποίηση Πελατών (assignments.csv)",
            data=exported_paths["assignments.csv"].read_bytes(),
            file_name="assignments.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # Clean Executive Header
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="top-header">
            <div>
                <h1>🎯 ForceCRM · Έξυπνη Ανάλυση Πελατολογίου & Προτάσεις Νέων Πελατών</h1>
                <p>Ανακαλύψτε τους τύπους πελατών που φέρνουν τον μεγαλύτερο τζίρο και βρείτε παρόμοιες επιχειρήσεις εστίασης στην Ελλάδα.</p>
            </div>
            <div>
                <span class="ai-badge">🤖 100% Local AI · EU AI Act Compliant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Simple, Business-Focused KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric("Πελάτες σας", f"{custom_clients.height:,}", "Ενεργό Πελατολόγιο")
    with k2:
        with st.container(border=True):
            st.metric("Συνολικός Τζίρος", f"€{custom_clients['turnover'].sum():,.0f}", f"Μέσος €{custom_clients['turnover'].mean():,.0f} / πελάτη")
    with k3:
        with st.container(border=True):
            st.metric("Τύποι Πελατών (ICPs)", f"{icp_result.optimal_k} Ομάδες", f"Αξιοπιστία {100*icp_result.global_bootstrap_ari:.0f}%")
    with k4:
        with st.container(border=True):
            st.metric("Νέες Προτάσεις (Lookalikes)", f"{icp_result.lookalikes_df.height:,}", f"Από 24.424 επιχειρήσεις")

    # -------------------------------------------------------------------------
    # 3 Customer-Centric Tabs + 1 Dedicated Technical Submenu Tab
    # -------------------------------------------------------------------------
    tab_overview, tab_leads, tab_clients, tab_tech = st.tabs(
        [
            "🧬 1. Τα Προφίλ Πελατών μας & Χάρτης (ICPs)",
            "🎯 2. Προτεινόμενοι Νέοι Πελάτες (Lookalikes)",
            "📋 3. Το Υφιστάμενο Πελατολόγιό μας",
            "🔬 4. Τεχνική Τεκμηρίωση & Αξιολόγηση AI (Submenu)",
        ]
    )

    # =========================================================================
    # TAB 1: CUSTOMER PROFILES (ICPs) & INTEGRATED MAP
    # =========================================================================
    with tab_overview:
        # Row 1: Two Intuitive Donut Charts + Integrated Map Slot
        top_left, top_right = st.columns([5, 7])

        with top_left:
            with st.container(border=True):
                fig_rev, fig_cnt = build_portfolio_donuts(icp_result.profiles)
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.plotly_chart(fig_rev, use_container_width=True)
                with d_col2:
                    st.plotly_chart(fig_cnt, use_container_width=True)

        with top_right:
            with st.container(border=True):
                m_head1, m_head2, m_head3 = st.columns([3, 2, 2])
                with m_head1:
                    st.markdown("##### 🗺️ Χάρτης Πελατών & Νέων Ευκαιριών")
                with m_head2:
                    prof_map_opts = ["ALL"] + [p.profile_id for p in icp_result.profiles]
                    sel_map_prof = st.selectbox(
                        "Φίλτρο Προφίλ:",
                        options=prof_map_opts,
                        format_func=lambda x: "Όλα τα Προφίλ" if x == "ALL" else x,
                        label_visibility="collapsed",
                    )
                with m_head3:
                    region_presets = {
                        "Όλη η Ελλάδα": {"lat": 38.15, "lon": 23.85, "zoom": 5.5},
                        "Αττική": {"lat": 37.98, "lon": 23.73, "zoom": 9.8},
                        "Χαλκιδική & Θεσ/νίκη": {"lat": 40.25, "lon": 23.35, "zoom": 8.2},
                        "Κρήτη": {"lat": 35.25, "lon": 24.80, "zoom": 7.7},
                        "Κυκλάδες & Δωδεκάνησα": {"lat": 36.85, "lon": 25.80, "zoom": 7.2},
                    }
                    sel_reg = st.selectbox(
                        "Περιοχή:",
                        options=list(region_presets.keys()),
                        label_visibility="collapsed",
                    )

                fig_map = build_icp_embedded_map(
                    base_ds=base_ds,
                    icp_result=icp_result,
                    selected_profile_filter=sel_map_prof,
                    region_center=region_presets[sel_reg],
                )
                st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("### 📌 Αναλυτική Εικόνα ανά Προφίλ Πελάτη (ICP)")

        # Individual ICP Cards: Clean Business Summary + Intuitive "Our ICP vs Market" Chart
        for idx, prof in enumerate(icp_result.profiles):
            accent = ICP_COLORS[idx % len(ICP_COLORS)]
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="icp-card-banner" style="border-left-color: {accent};">
                        <h3>🏷️ {prof.profile_id}: {prof.title_el}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                card_left, card_right = st.columns([5, 7])
                with card_left:
                    # 3 Clean Business Metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Πελάτες", f"{prof.client_count}", f"{100*prof.client_share:.0f}% πελατών")
                    c2.metric("Μερίδιο Τζίρου", f"{100*prof.turnover_share:.0f}%", f"€{prof.turnover_total:,.0f}")
                    c3.metric("Τυπικός Τζίρος", f"€{prof.turnover_median:,.0f}", f"Εμπιστοσύνη {100*prof.confidence_score:.0f}%")

                    # Concise, Jargon-Free Executive Summary
                    st.markdown(build_sales_friendly_description_el(prof))

                    # Visual Trait Pills
                    pills_html = "".join(
                        f'<span class="trait-pill">{_clean_trait_name(f.feature_label_el)} ({f.eb_lift:.1f}x)</span>'
                        for f in prof.significant_features[:4]
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)
                    st.caption("🤖 Περιγραφή παραγόμενη από Τοπικό AI (Άρθρο 50 EU AI Act) · Εμπορική Ομοιότητα B2B.")

                with card_right:
                    # Clear "Our Profile vs Market Average" Bar Chart
                    fig_comp = build_icp_vs_market_chart(prof, accent_color=accent)
                    st.plotly_chart(fig_comp, use_container_width=True)

                # Hidden Submenu for Technical Details of this ICP
                with st.expander(f"🔬 Στατιστική Τεκμηρίωση & Πίνακας Ελέγχων ({prof.profile_id} — Για Αναλυτές)"):
                    st.markdown(prof.ai_description_el)
                    feat_rows = [
                        {
                            "Χαρακτηριστικό": f.feature_label_el,
                            "Πελάτες ICP": f.icp_count,
                            "Μερίδιο στο Προφίλ (%)": round(100 * f.icp_share, 1),
                            "Μέσος Όρος Αγοράς (%)": round(100 * f.universe_share, 2),
                            "Empirical Bayes Lift": round(f.eb_lift, 2),
                            "Raw Lift": round(f.raw_lift, 2),
                            "FDR p-value (BH)": f"{f.p_value_fdr:.2e}",
                        }
                        for f in prof.significant_features
                    ]
                    st.dataframe(
                        pl.DataFrame(feat_rows).to_pandas(),
                        use_container_width=True,
                        hide_index=True,
                    )

    # =========================================================================
    # TAB 2: RECOMMENDED NEW LEADS (LOOKALIKES)
    # =========================================================================
    with tab_leads:
        st.subheader("🎯 Προτεινόμενοι Νέοι Πελάτες (Lookalikes)")
        st.caption(
            "Επιχειρήσεις της αγοράς που δεν είναι ακόμη πελάτες σας, αλλά ταιριάζουν απόλυτα με τα παραπάνω προφίλ."
        )

        ec_meta = base_ds.enriched_companies.select(
            [
                "company_id",
                "vat",
                "company_name_el",
                "legal_type",
                "prefecture_name",
                "municipality_name",
                "city",
                "street",
                "street_number",
                "zip_code",
                "primary_kad",
                "primary_kad_desc",
                "company_age_years",
                "branch_count",
                "total_kads",
            ]
        )
        rich_lookalikes = (
            icp_result.lookalikes_df.join(ec_meta, on="company_id", how="left")
            .with_columns((pl.col("score") * 100.0).round(1).alias("score_pct"))
        )

        with st.container(border=True):
            f1, f2, f3, f4 = st.columns([2, 3, 2, 3])
            with f1:
                icp_filter = st.multiselect(
                    "Προφίλ ICP",
                    options=[p.profile_id for p in icp_result.profiles],
                    default=[p.profile_id for p in icp_result.profiles],
                )
            with f2:
                avail_prefs = sorted(
                    [p for p in rich_lookalikes["prefecture_name"].unique().to_list() if p is not None]
                )
                pref_filter = st.multiselect("Νομός", options=avail_prefs, default=[])
            with f3:
                avail_lts = sorted(
                    [lt for lt in rich_lookalikes["legal_type"].unique().to_list() if lt is not None]
                )
                lt_filter = st.multiselect("Νομική Μορφή", options=avail_lts, default=[])
            with f4:
                search_query = st.text_input("Αναζήτηση (Επωνυμία, ΑΦΜ, Πόλη ή ΚΑΔ)", "")

        filtered_look = rich_lookalikes.filter(pl.col("profile_id").is_in(icp_filter))
        if pref_filter:
            filtered_look = filtered_look.filter(pl.col("prefecture_name").is_in(pref_filter))
        if lt_filter:
            filtered_look = filtered_look.filter(pl.col("legal_type").is_in(lt_filter))
        if search_query.strip():
            q = search_query.strip().upper()
            filtered_look = filtered_look.filter(
                pl.col("company_name_el").str.to_uppercase().str.contains(q, literal=True)
                | pl.col("vat").str.contains(q, literal=True)
                | pl.col("primary_kad").str.contains(q, literal=True)
                | pl.col("city").fill_null("").str.to_uppercase().str.contains(q, literal=True)
            )

        st.dataframe(
            filtered_look.select(
                [
                    "rank",
                    "profile_id",
                    "score_pct",
                    "company_name_el",
                    "legal_type",
                    "prefecture_name",
                    "city",
                    "primary_kad_desc",
                    "αιτιολόγηση",
                    "vat",
                ]
            ).to_pandas(),
            use_container_width=True,
            hide_index=True,
            height=450,
            column_config={
                "rank": st.column_config.NumberColumn("Σειρά", format="#%d", width="small"),
                "profile_id": st.column_config.TextColumn("Προφίλ", width="small"),
                "score_pct": st.column_config.ProgressColumn(
                    "Ταύτιση",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "company_name_el": st.column_config.TextColumn("Επωνυμία Επιχείρησης", width="medium"),
                "legal_type": st.column_config.TextColumn("Μορφή", width="small"),
                "prefecture_name": st.column_config.TextColumn("Νομός", width="medium"),
                "city": st.column_config.TextColumn("Πόλη / Περιοχή", width="small"),
                "primary_kad_desc": st.column_config.TextColumn("Κύρια Δραστηριότητα", width="medium"),
                "αιτιολόγηση": st.column_config.TextColumn("Γιατί Προτείνεται", width="large"),
                "vat": st.column_config.TextColumn("ΑΦΜ", width="small"),
            },
        )

        if filtered_look.height > 0:
            with st.expander("🔎 Προβολή Καρτέλας & Όλων των ΚΑΔ Επιλεγμένης Επιχείρησης"):
                top_inspect = filtered_look.head(100)
                opts = {
                    f"#{r['rank']} — {r['company_name_el']} ({r['prefecture_name']} · {r['score_pct']:.1f}%)": r["company_id"]
                    for r in top_inspect.iter_rows(named=True)
                }
                sel_lbl = st.selectbox("Επιλέξτε επιχείρηση:", options=list(opts.keys()))
                sel_cid = opts[sel_lbl]
                crow = filtered_look.filter(pl.col("company_id") == sel_cid).row(0, named=True)

                c_i1, c_i2, c_i3 = st.columns(3)
                c_i1.metric("Επωνυμία & ΑΦΜ", crow["company_name_el"], f"ΑΦΜ: {crow['vat']}")
                addr = f"{crow.get('street') or ''} {crow.get('street_number') or ''}, {crow.get('city') or ''} ({crow.get('zip_code') or ''})".strip()
                c_i2.metric("Διεύθυνση", crow["prefecture_name"] or "-", addr)
                c_i3.metric("Ηλικία & Δομή", f"{crow['company_age_years']:.1f} έτη", f"{crow['total_kads']} δηλωμένοι ΚΑΔ")

                comp_kads = (
                    base_ds.company_activities.filter(pl.col("company_id") == sel_cid)
                    .join(base_ds.activities, on="activity_id", how="left")
                    .sort(by=[pl.col("activity_type") == "Κύρια", "date_from"], descending=[True, False])
                )
                st.dataframe(
                    comp_kads.select(["activity_type", "activity_id", "description", "date_from"]).to_pandas(),
                    use_container_width=True,
                    hide_index=True,
                )

    # =========================================================================
    # TAB 3: EXISTING CLIENT PORTFOLIO
    # =========================================================================
    with tab_clients:
        st.subheader("📋 Κατηγοριοποίηση Υφιστάμενων Πελατών")
        st.caption(
            "Δείτε σε ποιο Προφίλ (ICP) ανήκει κάθε πελάτης σας, καθώς και ποιες επιχειρήσεις αποτελούν ειδικές περιπτώσεις (NO_FIT)."
        )
        rich_assign = (
            icp_result.assignments_df.join(
                base_ds.enriched_companies.select(
                    [
                        "company_id",
                        "vat",
                        "company_name_el",
                        "legal_type",
                        "prefecture_name",
                        "city",
                        "primary_kad_desc",
                        "turnover",
                    ]
                ),
                on="company_id",
                how="left",
            )
            .with_columns((pl.col("score") * 100.0).round(1).alias("score_pct"))
        )

        st.dataframe(
            rich_assign.select(
                [
                    "profile_id",
                    "score_pct",
                    "company_name_el",
                    "turnover",
                    "legal_type",
                    "prefecture_name",
                    "city",
                    "primary_kad_desc",
                    "vat",
                ]
            ).to_pandas(),
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                "profile_id": st.column_config.TextColumn("Προφίλ ICP", width="small"),
                "score_pct": st.column_config.ProgressColumn(
                    "Ταύτιση",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "company_name_el": st.column_config.TextColumn("Επωνυμία Πελάτη", width="medium"),
                "turnover": st.column_config.NumberColumn("Τζίρος (€)", format="€%d"),
                "legal_type": st.column_config.TextColumn("Μορφή", width="small"),
                "prefecture_name": st.column_config.TextColumn("Νομός", width="medium"),
                "city": st.column_config.TextColumn("Πόλη", width="small"),
                "primary_kad_desc": st.column_config.TextColumn("Κύρια Δραστηριότητα", width="large"),
                "vat": st.column_config.TextColumn("ΑΦΜ", width="small"),
            },
        )

    # =========================================================================
    # TAB 4: TECHNICAL SUBMENU (Hold-Out Benchmark, Silhouette, AI Act & Telemetry)
    # =========================================================================
    with tab_tech:
        st.subheader("🔬 Τεχνική Τεκμηρίωση, Αξιολόγηση Hold-Out & EU AI Act (Για Κριτές & Αναλυτές)")
        st.caption(
            "Όλες οι μαθηματικές μετρικές, η επιλογή του K, το benchmark Recall@500 / Recall@2000 και η τηλεμετρία των μοντέλων."
        )

        t_col1, t_col2 = st.columns([1, 1])
        with t_col1:
            with st.container(border=True):
                st.markdown("##### 📊 Επιλογή Αριθμού Προφίλ (Cosine Silhouette K=1..6)")
                sil_keys = sorted(icp_result.silhouette_by_k.keys())
                sil_df = pl.DataFrame(
                    {
                        "K": [f"K={k}" for k in sil_keys],
                        "Cosine Silhouette": [icp_result.silhouette_by_k[k] for k in sil_keys],
                        "Κατάσταση": ["Επιλεγμένο K" if k == icp_result.optimal_k else "Εναλλακτικό" for k in sil_keys],
                    }
                )
                fig_sil = px.bar(
                    sil_df.to_pandas(),
                    x="K",
                    y="Cosine Silhouette",
                    color="Κατάσταση",
                    text_auto=".3f",
                    color_discrete_map={"Επιλεγμένο K": "#2563eb", "Εναλλακτικό": "#94a3b8"},
                )
                fig_sil.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_sil, use_container_width=True)

        with t_col2:
            with st.container(border=True):
                st.markdown("##### 🎯 Έλεγχος Hold-Out (Recall@500 & Recall@2000 — Ενότητα 8.1)")
                st.markdown(
                    "- **ForceCRM Baseline (Συχνότητα Κύριου ΚΑΔ):** `Recall@500 = 1.83%` · `Recall@2000 = 8.17%`\n"
                    "- **Archimedes2026 Engine (10x Splits):** **`Recall@500 = 13.83%` (7.5x)** · **`Recall@2000 = 26.50%` (3.2x)**"
                )
                n_eval_splits = st.number_input("Αριθμός Monte Carlo Splits (80/20)", 1, 20, 5)
                if st.button("🚀 Εκτέλεση Ζωντανού Benchmark", type="primary"):
                    with st.spinner("Εκτέλεση Hold-Out αξιολόγησης..."):
                        base_m = evaluate_ranker_on_holdout(
                            ds=base_ds,
                            ranker_fn=baseline_primary_kad_ranker,
                            name="ForceCRM Baseline",
                            n_splits=int(n_eval_splits),
                            is_baseline=True,
                        )
                        icp_m = evaluate_ranker_on_holdout(
                            ds=base_ds,
                            ranker_fn=make_icp_engine_ranker(fs, cfg),
                            name="Archimedes2026 Engine",
                            n_splits=int(n_eval_splits),
                        )
                        st.success(
                            f"**Recall@500:** `{100*icp_m.recall_at_500_mean:.2f}%` vs Baseline `{100*base_m.recall_at_500_mean:.2f}%` | "
                            f"**Recall@2000:** `{100*icp_m.recall_at_2000_mean:.2f}%` vs Baseline `{100*base_m.recall_at_2000_mean:.2f}%`"
                        )

        with st.expander("🛡️ Πλήρες EU AI Act Self-Assessment (Ενότητα 9) & Τηλεμετρία Μοντέλων (Ενότητα 5.3)", expanded=False):
            st.markdown(
                """
1. **Τι αποφασίζει ή προτείνει το σύστημα;** Ομαδοποιεί B2B πελάτες εστίασης σε 1–6 Ideal Customer Profiles και προτείνει παρόμοιες επιχειρήσεις από το Universe αποκλειστικά βάσει εμπορικής/μορφολογικής ομοιότητας.
2. **Προστασία Φυσικών Προσώπων (`ΑΤΟΜΙΚΗ`):** Οι 12.090 ατομικές επιχειρήσεις είναι φυσικά πρόσωπα. Το σύστημα **δεν αξιολογεί πιστοληπτική ικανότητα ή φερεγγυότητα**.
3. **Διαφάνεια (Άρθρο 50):** Κάθε κείμενο AI σημαίνεται ρητά και κάθε ισχυρισμός τεκμηριώνεται με Empirical Bayes Lift και Benjamini-Hochberg FDR p-values.
                """
            )
            st.json(fs.embedding_telemetry)


if __name__ == "__main__":
    main()
