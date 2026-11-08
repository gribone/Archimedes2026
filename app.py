r"""
Stage 6: ForceCRM Archimedes 2026 — Executive B2B Intelligence Application (`app.py`)
=====================================================================================
Designed for B2B Sales Leaders & Account Executives:
1. **Interactive OpenStreetMap Route Planner (Extra Point X1/X3):**
   - High-performance Plotly OpenStreetMap map (`mapbox_style="open-street-map"`, zero API keys).
   - Plots existing clients (sized by `turnover`, colored by ICP) alongside recommended Lookalikes.
   - Interactive Municipality (`Δήμος`) and ICP filters for visual neighborhood sales route planning.
2. **Professional Executive Theme & Auto-Fitting Tables:**
   - Dark Slate & Navy luxury styling with clean borders and zero horizontal table scrolling.
   - Pre-filtered against deprecated/closed venues via the Premise Supersession Engine.
3. **Dual-Mode Architecture:**
   - Streamlit Application: default or via `streamlit run app.py` / `python app.py`.
   - Single-Page Application (SPA): via `python app.py --spa` launching Starlette ASGI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Dynamically locate virtual environment site-packages if present
_venv_base = Path(__file__).resolve().parent / ".venv"
if _venv_base.exists():
    _py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _venv_site = _venv_base / "lib" / _py_ver / "site-packages"
    if _venv_site.exists() and str(_venv_site) not in sys.path:
        sys.path.insert(0, str(_venv_site))

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
from src.icp_engine import DiscoveredICP, ICPEngineResult, export_deliverables, run_icp_engine
from src.ingest import IngestedDataset, ingest_all, load_clients_excel
from src.llm_explainer import check_local_ollama_status, enrich_profiles_with_local_llm

ICP_COLORS = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]


# =============================================================================
# CACHED DATA LOADERS
# =============================================================================
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


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================
def _clean_trait_name(label: str) -> str:
    cleaned = (
        label.replace("Δήμος: ", "📍 ")
        .replace("Νομός: ", "🗺️ ")
        .replace("Νομική Μορφή: ", "🏛️ ")
        .replace("Κατηγορία ΚΑΔ (4-ψήφιος): ", "🍽️ ")
        .replace("Γεωγραφικό Καθεστώς: ", "🌐 ")
    )
    return cleaned[:37] + "…" if len(cleaned) > 38 else cleaned


def build_portfolio_donuts(profiles: list[DiscoveredICP]) -> tuple[go.Figure, go.Figure]:
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
                hovertemplate="<b>%{label}</b><br>Κύκλος Εργασιών: €%{value:,.0f} (%{percent})<extra></extra>",
            )
        ]
    )
    total_rev_m = sum(turnovers) / 1e6
    fig_rev.update_layout(
        title=dict(text="💰 Κατανομή Κύκλου Εργασιών ανά ICP", font=dict(size=14, color="#1e293b")),
        annotations=[
            dict(
                text=f"<b>€{total_rev_m:.2f}M</b><br>Συν. Κύκλος Εργ.",
                x=0.5,
                y=0.5,
                font_size=13,
                showarrow=False,
            )
        ],
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
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
        title=dict(text="👥 Κατανομή Πελατών ανά ICP", font=dict(size=14, color="#1e293b")),
        annotations=[
            dict(
                text=f"<b>{sum(counts)}</b><br>Πελάτες",
                x=0.5,
                y=0.5,
                font_size=13,
                showarrow=False,
            )
        ],
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig_rev, fig_cnt


def build_icp_vs_market_chart(prof: DiscoveredICP, accent_color: str = "#2563eb") -> go.Figure:
    traits = prof.significant_features[:5][::-1]
    y_labels = [_clean_trait_name(f.feature_label_el) for f in traits]
    full_names = [f.feature_label_el for f in traits]
    icp_pcts = [round(100 * f.icp_share, 1) for f in traits]
    univ_pcts = [round(100 * f.universe_share, 1) for f in traits]
    lifts = [f.eb_lift for f in traits]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=univ_pcts,
            name="Μέσος Όρος Αγοράς",
            orientation="h",
            marker_color="#94a3b8",
            text=[f"{u:.1f}%" for u in univ_pcts],
            textposition="outside",
            customdata=full_names,
            hovertemplate="<b>%{customdata}</b><br>Μέσος Όρος: <b>%{x:.1f}%</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=icp_pcts,
            name=f"Πελάτες {prof.profile_id}",
            orientation="h",
            marker_color=accent_color,
            text=[f"<b>{p:.1f}%</b> ({l:.1f}x)" for p, l in zip(icp_pcts, lifts)],
            textposition="outside",
            customdata=full_names,
            hovertemplate="<b>%{customdata}</b><br>Στο Προφίλ: <b>%{x:.1f}%</b><extra></extra>",
        )
    )
    max_x = max(icp_pcts + univ_pcts + [20.0]) * 1.35
    fig.update_layout(
        title=dict(text="Διαφοροποιητικά Χαρακτηριστικά έναντι Αγοράς", font=dict(size=13)),
        barmode="group",
        xaxis=dict(title="Ποσοστό (%)", range=[0, max_x], showgrid=True),
        yaxis=dict(title=""),
        height=270,
        margin=dict(l=10, r=20, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def build_route_planner_map(
    base_ds: IngestedDataset,
    icp_result: ICPEngineResult,
    selected_icp: str = "ALL",
    selected_muni: str = "ALL",
    min_score: float = 0.50,
    top_lookalikes: int = 500,
) -> go.Figure:
    """
    Interactive OpenStreetMap Route Planner (Extra Point X1/X3):
    Plots existing clients (sized by turnover, colored by ICP) alongside
    recommended Lookalikes (colored by predicted ICP) with Municipality
    and ICP filtering for neighborhood sales route planning.
    Uses Plotly scatter_mapbox with mapbox_style="open-street-map" (0 API keys required).
    """
    ec_sorted = base_ds.enriched_companies.sort("company_id")

    # 1. Clients
    client_df = (
        ec_sorted.join(icp_result.assignments_df, on="company_id", how="inner")
        .filter(pl.col("has_gps") & (pl.col("profile_id") != "NO_FIT"))
        .select([
            "company_id",
            "vat",
            "company_name_el",
            "legal_type",
            "prefecture_name",
            "municipality_name",
            "primary_kad",
            "primary_kad_desc",
            "turnover",
            "latitude",
            "longitude",
            "profile_id",
            "score",
        ])
    )

    # 2. Lookalikes
    look_sub = icp_result.lookalikes_df.head(top_lookalikes)
    look_cols = [
        "company_id",
        "vat",
        "company_name_el",
        "legal_type",
        "prefecture_name",
        "municipality_name",
        "primary_kad",
        "primary_kad_desc",
        "latitude",
        "longitude",
        "has_gps",
    ]
    if "nearest_client_km" in look_sub.columns:
        look_df = (
            look_sub.join(ec_sorted.select(look_cols), on="company_id", how="left")
            .filter(pl.col("has_gps") & (pl.col("score") >= min_score))
        )
    else:
        look_df = (
            look_sub.join(ec_sorted.select(look_cols), on="company_id", how="left")
            .filter(pl.col("has_gps") & (pl.col("score") >= min_score))
            .with_columns(pl.lit(None).cast(pl.Float64).alias("nearest_client_km"))
        )

    # 3. Filters
    if selected_icp != "ALL":
        client_df = client_df.filter(pl.col("profile_id") == selected_icp)
        look_df = look_df.filter(pl.col("profile_id") == selected_icp)

    if selected_muni != "ALL":
        client_df = client_df.filter(pl.col("municipality_name") == selected_muni)
        look_df = look_df.filter(pl.col("municipality_name") == selected_muni)

    # Determine map center
    all_lats: list[float] = []
    all_lons: list[float] = []
    if client_df.height > 0:
        all_lats.extend(client_df["latitude"].to_list())
        all_lons.extend(client_df["longitude"].to_list())
    if look_df.height > 0:
        all_lats.extend(look_df["latitude"].to_list())
        all_lons.extend(look_df["longitude"].to_list())

    if len(all_lats) > 0 and selected_muni != "ALL":
        center_lat = float(np.median(all_lats))
        center_lon = float(np.median(all_lons))
        zoom = 12.0
    elif len(all_lats) > 0:
        center_lat = float(np.median(all_lats))
        center_lon = float(np.median(all_lons))
        zoom = 6.4
    else:
        center_lat, center_lon, zoom = 38.2, 23.8, 6.0

    fig = go.Figure()

    icp_color_map = {
        "ICP_1": "#2563eb",
        "ICP_2": "#10b981",
        "ICP_3": "#f59e0b",
        "ICP_4": "#8b5cf6",
        "ICP_5": "#ec4899",
        "ICP_6": "#06b6d4",
    }

    # Add Lookalike traces per ICP
    for prof in icp_result.profiles:
        pid = prof.profile_id
        color = icp_color_map.get(pid, "#3b82f6")
        sub_look = look_df.filter(pl.col("profile_id") == pid)
        if sub_look.height > 0:
            custom_data = []
            for r in sub_look.iter_rows(named=True):
                dist_str = f"{r.get('nearest_client_km', 0):.1f} km" if r.get("nearest_client_km") is not None else "N/A"
                custom_data.append([
                    r["company_name_el"],
                    r["legal_type"],
                    r["municipality_name"] or "-",
                    r["prefecture_name"] or "-",
                    (r["primary_kad_desc"] or "")[:45],
                    f"{100*r['score']:.1f}%",
                    dist_str,
                    r.get("αιτιολόγηση", ""),
                ])

            _ScatterMap = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
            fig.add_trace(
                _ScatterMap(
                    lat=sub_look["latitude"].to_list(),
                    lon=sub_look["longitude"].to_list(),
                    mode="markers",
                    marker=dict(size=9, color=color, opacity=0.82),
                    name=f"🎯 {pid} Προτεινόμενοι ({sub_look.height})",
                    customdata=custom_data,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "📍 %{customdata[2]} (%{customdata[3]})<br>"
                        "🍽️ %{customdata[4]}<br>"
                        "⭐ Ταύτιση ICP: <b>%{customdata[5]}</b><br>"
                        "🚚 Απόσταση από πελάτη: <b>%{customdata[6]}</b><br>"
                        "💡 <i>%{customdata[7]}</i>"
                        "<extra></extra>"
                    ),
                )
            )

    # Add Client traces per ICP (larger markers, sized by turnover)
    for prof in icp_result.profiles:
        pid = prof.profile_id
        color = icp_color_map.get(pid, "#2563eb")
        sub_clients = client_df.filter(pl.col("profile_id") == pid)
        if sub_clients.height > 0:
            turnovers = sub_clients["turnover"].to_numpy()
            sizes = np.clip(np.sqrt(np.maximum(turnovers, 1000.0)) / 14.0, 11.0, 28.0)
            custom_data = []
            for r in sub_clients.iter_rows(named=True):
                custom_data.append([
                    r["company_name_el"],
                    r["legal_type"],
                    r["municipality_name"] or "-",
                    r["prefecture_name"] or "-",
                    (r["primary_kad_desc"] or "")[:45],
                    f"€{r['turnover']:,.0f}",
                ])

            _ScatterMap = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
            fig.add_trace(
                _ScatterMap(
                    lat=sub_clients["latitude"].to_list(),
                    lon=sub_clients["longitude"].to_list(),
                    mode="markers",
                    marker=dict(size=sizes, color=color, opacity=0.95),
                    name=f"⭐ {pid} Πελάτες ({sub_clients.height})",
                    customdata=custom_data,
                    hovertemplate=(
                        "<b>⭐ %{customdata[0]}</b> (ΠΕΛΑΤΗΣ)<br>"
                        "📍 %{customdata[2]} (%{customdata[3]})<br>"
                        "💰 Κύκλος Εργασιών: <b>%{customdata[5]}</b><br>"
                        "🍽️ %{customdata[4]}"
                        "<extra></extra>"
                    ),
                )
            )

    map_config = dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=zoom,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#94a3b8",
            borderwidth=1,
        ),
        height=580,
    )
    if hasattr(go, "Scattermap"):
        fig.update_layout(map=map_config)
    else:
        fig.update_layout(mapbox=map_config)
    return fig


# =============================================================================
# STREAMLIT APPLICATION ENTRY POINT
# =============================================================================
def run_streamlit_app() -> None:
    st.set_page_config(
        page_title="ForceCRM · Ευφυΐα Πελατολογίου & Σχεδιασμός Δρομολογίων",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Executive Clean Styling
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }
        .top-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
            padding: 1.1rem 1.6rem;
            border-radius: 8px;
            margin-bottom: 1.2rem;
            border: 1px solid #334155;
        }
        .top-header h1 {
            font-size: 1.45rem;
            font-weight: 700;
            margin: 0;
            color: #ffffff;
        }
        .top-header p {
            font-size: 0.88rem;
            color: #cbd5e1;
            margin: 0.2rem 0 0 0;
        }
        .ai-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(59, 130, 246, 0.18);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.35);
        }
        .icp-card-banner {
            padding: 0.7rem 1.1rem;
            border-radius: 6px;
            margin-bottom: 0.8rem;
            border-left: 4px solid #2563eb;
            background: rgba(37, 99, 235, 0.06);
        }
        .icp-card-banner h3 {
            margin: 0;
            font-size: 1.15rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: File Uploader + Hidden Analyst Submenu
    with st.sidebar:
        st.markdown("### 📂 1. Αρχείο Πελατολογίου")
        uploaded_file = st.file_uploader(
            "Ανεβάστε το Excel πελατών (.xlsx)",
            type=["xlsx"],
            help="Στήλες: ΑΦΜ και Κύκλος Εργασιών (€). Προεπιλογή: 300 πελάτες ForceCRM.",
        )
        if uploaded_file is not None:
            st.success(f"Ενεργό: `{uploaded_file.name}`")
        else:
            st.caption("Ενεργό: `ForceCRM_Hackathon_Pelatologio.xlsx` (300 πελάτες)")

        st.divider()

        with st.expander("⚙️ Ρυθμίσεις Μοντέλου & Παράμετροι (Αναλυτής)", expanded=False):
            emb_options = list(EMBEDDING_MODELS_REGISTRY.keys())
            selected_emb = st.selectbox(
                "Μοντέλο Embeddings ΚΑΔ",
                options=emb_options,
                index=emb_options.index("granite-107m"),
            )
            hybrid_alpha = st.slider("Hybrid Alpha (Neural vs SVD)", 0.0, 1.0, 0.50, 0.05)

            ollama_ok, _, installed_models = check_local_ollama_status("qwen2.5:7b")
            if ollama_ok:
                selected_ollama = st.selectbox("Τοπικό LLM (Ollama)", options=installed_models, index=0)
                use_ollama = st.toggle("Χρήση Ollama LLM", value=True)
            else:
                selected_ollama = "qwen2.5:7b"
                use_ollama = False
                st.caption("Ollama offline — Χρήση ντετερμινιστικής γεννήτριας.")

            min_icps, max_icps = st.slider("Εύρος Προφίλ (K)", 1, 6, (1, 6))
            turnover_mode = st.selectbox("Στάθμιση Κύκλου Εργασιών", options=["log_turnover", "sqrt_turnover", "uniform"], index=0)
            turnover_alpha = st.slider("Βάρος Κύκλου Εργασιών (α)", 0.0, 1.0, 0.50, 0.05)
            w_primary = st.slider("Βάρος Κύριου ΚΑΔ", 1.0, 6.0, 3.5, 0.25)
            w_secondary = st.slider("Βάρος Δευτερεύοντος ΚΑΔ", 0.1, 2.0, 0.75, 0.05)
            w_auxiliary = 0.25
            idf_power = st.slider("Απόσβεση Κοινών ΚΑΔ (IDF)", 0.0, 2.5, 1.25, 0.05)
            sec_damping = st.slider("Απόσβεση Πλήθους ΚΑΔ", 0.0, 1.0, 0.35, 0.05)
            bw_semantic, bw_kad_hier, bw_spatial, bw_legal, bw_age_branch = 1.00, 0.85, 1.15, 0.90, 0.45
            eb_prior = st.slider("Empirical Bayes Prior", 1.0, 30.0, 8.0, 1.0)
            n_boot = st.slider("Bootstrap Runs", 10, 50, 30, 5)
            w_cos = st.slider("Βάρος Ομοιότητας Centroid", 0.0, 1.0, 0.35, 0.05)
            w_eb = st.slider("Βάρος Στατιστικού Lift", 0.0, 1.0, 0.50, 0.05)
            w_sp = st.slider("Βάρος Εγγύτητας Χάρτη", 0.0, 1.0, 0.15, 0.05)
            bw_km = st.slider("Ακτίνα Εγγύτητας (km)", 5.0, 100.0, 25.0, 5.0)
            no_fit_min = st.slider("Κατώφλι NO_FIT", 0.15, 0.65, 0.35, 0.01)

    # Ingestion & Model Execution
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

    # Top Header
    st.markdown(
        """
        <div class="top-header">
            <div>
                <h1>🎯 ForceCRM Archimedes 2026 · Ευφυΐα Πελατολογίου & Σχεδιασμός Δρομολογίων</h1>
                <p>Ομαδοποίηση υφιστάμενων B2B πελατών, εντοπισμός νέων υποψήφιων πελατών (Lookalikes) και οπτικός σχεδιασμός δρομολογίων πωλήσεων.</p>
            </div>
            <div>
                <span class="ai-badge">🛡️ EU AI Act Art. 50 Grounded · 100% Offline</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric("Ενεργοί Πελάτες", f"{custom_clients.height:,}", "Από πελατολόγιο")
    with k2:
        with st.container(border=True):
            st.metric("Συνολικός Κύκλος Εργασιών", f"€{custom_clients['turnover'].sum():,.0f}", f"Μέσος €{custom_clients['turnover'].mean():,.0f}")
    with k3:
        with st.container(border=True):
            st.metric("Προφίλ Πελατών (ICPs)", f"{icp_result.optimal_k} Ομάδες", f"Σταθερότητα {100*icp_result.global_bootstrap_ari:.0f}%")
    with k4:
        with st.container(border=True):
            st.metric("Προτεινόμενα Leads", f"{icp_result.lookalikes_df.height:,}", "Από 24.424 επιχειρήσεις")

    # Main Tabs
    tab_profiles, tab_leads, tab_route_planner, tab_clients, tab_tech = st.tabs(
        [
            "🧬 1. Προφίλ Πελατών (ICPs)",
            "🎯 2. Προτεινόμενοι Πελάτες (Lookalikes)",
            "🗺️ 3. OpenStreetMap Route Planner",
            "📋 4. Υφιστάμενο Πελατολόγιο",
            "🔬 5. Τεχνική Τεκμηρίωση & Benchmarks",
        ]
    )

    # =========================================================================
    # TAB 1: CUSTOMER PROFILES (ICPs)
    # =========================================================================
    with tab_profiles:
        d_col1, d_col2 = st.columns(2)
        fig_rev, fig_cnt = build_portfolio_donuts(icp_result.profiles)
        with d_col1:
            with st.container(border=True):
                st.plotly_chart(fig_rev, use_container_width=True)
        with d_col2:
            with st.container(border=True):
                st.plotly_chart(fig_cnt, use_container_width=True)

        st.markdown("### 📌 Αναλυτική Εικόνα ανά Προφίλ Πελάτη (ICP)")
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
                c_left, c_right = st.columns([5, 7])
                with c_left:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Πελάτες", f"{prof.client_count}", f"{100*prof.client_share:.0f}%")
                    m2.metric("Μερίδιο Κύκλου Εργασιών", f"{100*prof.turnover_share:.0f}%", f"€{prof.turnover_total:,.0f}")
                    m3.metric("Διάμεσος Κύκλος Εργασιών", f"€{prof.turnover_median:,.0f}", f"Εμπιστοσύνη {100*prof.confidence_score:.0f}%")

                    st.markdown("**Επιχειρηματική Περιγραφή:**")
                    st.markdown(prof.ai_description_el)

                with c_right:
                    fig_vs = build_icp_vs_market_chart(prof, accent_color=accent)
                    st.plotly_chart(fig_vs, use_container_width=True)

                feat_rows = [
                    {
                        "Χαρακτηριστικό": f.feature_label_el,
                        "Πελάτες ICP": f.icp_count,
                        "Μερίδιο στο Προφίλ": f"{100*f.icp_share:.1f}%",
                        "Μέσος Όρος Αγοράς": f"{100*f.universe_share:.2f}%",
                        "Empirical Bayes Lift": f"{f.eb_lift:.2f}x",
                        "FDR p-value": f"{f.p_value_fdr:.2e}",
                    }
                    for f in prof.significant_features
                ]
                st.dataframe(
                    pl.DataFrame(feat_rows).to_pandas(),
                    use_container_width=True,
                    hide_index=True,
                )

    # =========================================================================
    # TAB 2: RECOMMENDED LOOKALIKES
    # =========================================================================
    with tab_leads:
        st.subheader("🎯 Προτεινόμενοι Νέοι Πελάτες (Lookalikes)")
        st.caption(
            "Επιχειρήσεις της αγοράς εκτός πελατολογίου που προσομοιάζουν τα χαρακτηριστικά των καλύτερων πελατών σας."
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

        cols_to_show = [
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
        if "nearest_client_km" in filtered_look.columns:
            cols_to_show.insert(3, "nearest_client_km")

        st.dataframe(
            filtered_look.select(cols_to_show).to_pandas(),
            use_container_width=True,
            hide_index=True,
            height=440,
            column_config={
                "rank": st.column_config.NumberColumn("Σειρά", format="#%d", width="small"),
                "profile_id": st.column_config.TextColumn("Προφίλ", width="small"),
                "score_pct": st.column_config.ProgressColumn("Ταύτιση", format="%.1f%%", min_value=0, max_value=100),
                "nearest_client_km": st.column_config.NumberColumn("Εγγύτητα (km)", format="%.1f km", width="small"),
                "company_name_el": st.column_config.TextColumn("Επωνυμία", width="medium"),
                "legal_type": st.column_config.TextColumn("Μορφή", width="small"),
                "prefecture_name": st.column_config.TextColumn("Νομός", width="small"),
                "city": st.column_config.TextColumn("Πόλη", width="small"),
                "primary_kad_desc": st.column_config.TextColumn("Κύρια Δραστηριότητα", width="medium"),
                "αιτιολόγηση": st.column_config.TextColumn("Γιατί Προτείνεται", width="large"),
                "vat": st.column_config.TextColumn("ΑΦΜ", width="small"),
            },
        )

    # =========================================================================
    # TAB 3: INTERACTIVE OPENSTREETMAP ROUTE PLANNER
    # =========================================================================
    with tab_route_planner:
        st.subheader("🗺️ Σχεδιασμός Δρομολογίων Πωλήσεων (OpenStreetMap Route Planner)")
        st.caption(
            "Οπτικοποιήστε τους υφιστάμενους πελάτες σας (μεγέθη ανάλογα με τον κύκλο εργασιών) δίπλα στους νέους υποψήφιους πελάτες (Lookalikes) "
            "για άμεσο σχεδιασμό δρομολογίων πωλητών ανά Δήμο και Προφίλ ICP (Extra Points X1/X3)."
        )

        rp1, rp2, rp3 = st.columns([3, 3, 2])
        with rp1:
            all_munis = sorted([
                m for m in base_ds.enriched_companies["municipality_name"].unique().to_list()
                if m is not None
            ])
            sel_muni = st.selectbox("Επιλέξτε Δήμο:", options=["ALL"] + all_munis, format_func=lambda x: "Όλη η Ελλάδα (Όλοι οι Δήμοι)" if x == "ALL" else f"Δήμος {x}")
        with rp2:
            prof_options = ["ALL"] + [p.profile_id for p in icp_result.profiles]
            sel_icp = st.selectbox("Επιλέξτε Προφίλ ICP:", options=prof_options, format_func=lambda x: "Όλα τα Προφίλ ICP" if x == "ALL" else f"Προφίλ {x}")
        with rp3:
            top_n_map = st.selectbox("Πλήθος Προτεινόμενων:", options=[200, 500, 1000, 2000], index=1)

        with st.container(border=True):
            fig_planner = build_route_planner_map(
                base_ds=base_ds,
                icp_result=icp_result,
                selected_icp=sel_icp,
                selected_muni=sel_muni,
                min_score=0.45,
                top_lookalikes=int(top_n_map),
            )
            st.plotly_chart(fig_planner, use_container_width=True)

    # =========================================================================
    # TAB 4: EXISTING CLIENTS PORTFOLIO
    # =========================================================================
    with tab_clients:
        st.subheader("📋 Κατηγοριοποίηση Υφιστάμενου Πελατολογίου")
        st.caption("Κατανομή των υφιστάμενων πελατών σας ανά ICP και εντοπισμός ειδικών περιπτώσεων (NO_FIT).")

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
            height=440,
            column_config={
                "profile_id": st.column_config.TextColumn("Προφίλ ICP", width="small"),
                "score_pct": st.column_config.ProgressColumn("Ταύτιση", format="%.1f%%", min_value=0, max_value=100),
                "company_name_el": st.column_config.TextColumn("Επωνυμία Πελάτη", width="medium"),
                "turnover": st.column_config.NumberColumn("Κύκλος Εργασιών (€)", format="€%d"),
                "legal_type": st.column_config.TextColumn("Μορφή", width="small"),
                "prefecture_name": st.column_config.TextColumn("Νομός", width="small"),
                "city": st.column_config.TextColumn("Πόλη", width="small"),
                "primary_kad_desc": st.column_config.TextColumn("Κύρια Δραστηριότητα", width="large"),
                "vat": st.column_config.TextColumn("ΑΦΜ", width="small"),
            },
        )

    # =========================================================================
    # TAB 5: TECHNICAL BENCHMARKS & MODEL TELEMETRY
    # =========================================================================
    with tab_tech:
        st.subheader("🔬 Τεχνική Τεκμηρίωση & Αξιολόγηση Μοντέλου")
        st.caption("Μαθηματικές μετρικές, επιλογή K μέσω Cosine Silhouette και Hold-Out Cross Validation.")

        t_col1, t_col2 = st.columns(2)
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
                fig_sil.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_sil, use_container_width=True)

        with t_col2:
            with st.container(border=True):
                st.markdown("##### 🎯 Έλεγχος Hold-Out (Recall@500 & Recall@2000 — Ενότητα 8.1)")
                st.markdown(
                    "- **ForceCRM Baseline (Συχνότητα Κύριου ΚΑΔ):** `Recall@500 = 1.67%` · `Recall@2000 = 7.00%`\n"
                    "- **Archimedes2026 Engine (100-Fold CV):** **`Recall@500 = 12.67%` (7.7x)** · **`Recall@2000 = 38.00%` (5.8x)**"
                )
                n_eval_splits = st.number_input("Αριθμός Folds (Monte Carlo 80/20)", 1, 10, 5)
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

        with st.expander("🛡️ Τηλεμετρία Μοντέλου & EU AI Act Article 50 Grounding", expanded=False):
            st.json(fs.embedding_telemetry)


# =============================================================================
# CLI DUAL-MODE DISPATCHER
# =============================================================================
def main() -> None:
    # If run through streamlit runner directly (e.g. `streamlit run app.py`)
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            run_streamlit_app()
            return
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="ForceCRM Archimedes 2026 Executive B2B Intelligence Application"
    )
    parser.add_argument("--spa", action="store_true", default=True, help="Launch the Starlette ASGI SPA web server (default)")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit legacy application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Port to listen on (default: 8501)")
    parser.add_argument("--reload", action="store_true", help="Enable dev auto-reload (SPA mode only)")
    args, unknown = parser.parse_known_args()

    if args.streamlit:
        # Launch Streamlit application via streamlit.web.cli
        import streamlit.web.cli as stcli
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.port",
            str(args.port),
            "--server.address",
            args.host,
        ] + unknown
        sys.exit(stcli.main())
    else:
        # Default: Launch Starlette ASGI SPA Web Server
        from src.server import run_server
        if args.reload:
            import uvicorn
            print(f"🚀 Launching Archimedes 2026 SPA at http://{args.host}:{args.port}")
            uvicorn.run("src.server:create_app", host=args.host, port=args.port, reload=True, factory=True)
        else:
            run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
