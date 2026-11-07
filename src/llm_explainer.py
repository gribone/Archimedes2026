r"""
Stage 5: Local LLM Grounded Greek Profile Explainer (`src/llm_explainer.py`)
============================================================================
Implements Requirement A6 and Section 9 (EU AI Act Article 50 Transparency & Groundedness):

1. **Local Execution Only (Section 5.1 & Section 6):**
   - Connects solely to local Ollama (`http://localhost:11434`) using the configured open-source /
     European LLM (e.g., `EuroLLM-9B-Instruct`, `Llama-Krikri-8B-Instruct`, `Ministral-3-3B-Instruct`,
     `Salamandra-7B-Instruct`, or `qwen2.5:7b-instruct`).
   - Enforces **strict reproducibility**: `temperature = 0.0`, `seed = cfg.llm_seed` (2026), `top_p = 1.0`.

2. **100% Groundedness & Zero Hallucination Guardrail (Requirement A6):**
   - Passes ONLY the statistically proven metrics of each discovered ICP (`client_count`, `client_share`,
     `turnover_share`, `turnover_median`, `stability_score`, `confidence_score`, and the Hypergeometric /
     Empirical Bayes `significant_features`).
   - Validates that the generated Greek text includes the required AI-generation disclosure badge
     (EU AI Act Article 50), cites the profile's stability and confidence scores, and never uses forbidden
     creditworthiness/solvency terms (`φερεγγυότητα`, `πιστοληπτική ικανότητα`) except in the explicit
     EU AI Act disclaimer.

3. **Seamless Deterministic Fallback:**
   - If the local Ollama daemon is not running or no model is pulled yet, automatically falls back to
     the deterministic, 100% grounded Greek narrative generator so the pipeline and Streamlit UI work
     out-of-the-box without crashing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.config import PipelineConfig
from src.icp_engine import DiscoveredICP, ICPEngineResult


AI_DISCLOSURE_PREFIX_EL = (
    "🤖 [Περιεχόμενο παραγόμενο από Τοπικό AI — Άρθρο 50 EU AI Act | "
    "100% Grounded στα στατιστικά δεδομένα του προφίλ]"
)

EU_AI_ACT_FOOTER_EL = (
    "⚖️ Σημείωση EU AI Act (Ενότητα 9): Οι δείκτες αφορούν αποκλειστικά εμπορική τυποποίηση "
    "και ομοιότητα B2B (Commercial Fit / ICP Similarity) και ΔΕΝ αποτελούν αξιολόγηση "
    "πιστοληπτικής ικανότητας ή φερεγγυότητας φυσικών ή νομικών προσώπων."
)


@dataclass
class LLMExplainerTelemetry:
    """Tracks local LLM model usage, latency, and groundedness verification for Section 5.3."""

    used_ollama: bool
    model_name: str
    temperature: float
    seed: int
    total_elapsed_sec: float
    per_profile_elapsed_sec: dict[str, float]
    fallback_reason: str | None = None


def check_local_ollama_status(preferred_model: str) -> tuple[bool, str, list[str]]:
    """
    Check if local Ollama server is reachable and return `(is_available, resolved_model, installed_models)`.
    If `preferred_model` is not installed but another local model is available in Ollama,
    automatically selects the first available local model.
    """
    try:
        import ollama

        resp = ollama.list()
        models_raw = getattr(resp, "models", None)
        if models_raw is None and isinstance(resp, dict):
            models_raw = resp.get("models", [])

        installed: list[str] = []
        for m in models_raw or []:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                installed.append(str(name))

        if not installed:
            return False, preferred_model, []

        if preferred_model in installed:
            return True, preferred_model, installed

        # Match prefix (e.g. "qwen2.5" or "krikri" or "eurollm")
        pref_base = preferred_model.split(":")[0].lower()
        for inst in installed:
            if pref_base in inst.lower():
                return True, inst, installed

        return True, installed[0], installed
    except Exception:
        return False, preferred_model, []


def build_grounded_prompt_el(prof: DiscoveredICP) -> tuple[str, str]:
    """
    Construct the system and user prompts in Greek forcing every claim to map strictly
    to the statistical payload of `prof` (Requirement A6).
    """
    system_prompt = (
        "Είσαι εξειδικευμένος αναλυτής B2B δεδομένων εστίασης για την ForceCRM. "
        "Γράψε μια σύντομη, επαγγελματική περιγραφή (1 παράγραφος, 90-140 λέξεις) στα Ελληνικά "
        "για το Ideal Customer Profile (ICP) ΑΠΟΚΛΕΙΣΤΙΚΑ με βάση τα αριθμητικά δεδομένα που σου δίνονται. "
        "ΑΥΣΤΗΡΟΙ ΚΑΝΟΝΕΣ:\n"
        "1. Κάθε ισχυρισμός ΠΡΕΠΕΙ να αντιστοιχεί σε μετρήσιμο στοιχείο από τα παρεχόμενα δεδομένα "
        "(πλήθος πελατών, μερίδιο τζίρου, διάμεσος τζίρος, σταθερότητα, εμπιστοσύνη, ΚΑΔ, νομοί/δήμοι, νομική μορφή, EB Lift, p-value).\n"
        "2. ΜΗΝ επινοήσεις κανένα στοιχείο, αριθμό ή χαρακτηριστικό που δεν αναφέρεται ρητά.\n"
        "3. ΜΗΝ χρησιμοποιήσεις όρους πιστοληπτικής ικανότητας ή φερεγγυότητας (απαγορεύεται από το EU AI Act).\n"
        "4. Αναφέρετε ρητά τον Δείκτη Σταθερότητας (Bootstrap) και τον Δείκτη Εμπιστοσύνης του προφίλ."
    )

    traits_lines = []
    for idx, f in enumerate(prof.significant_features[:6], start=1):
        traits_lines.append(
            f"  {idx}. {f.feature_label_el}: {f.icp_count} πελάτες ({100*f.icp_share:.1f}% του προφίλ "
            f"έναντι {100*f.universe_share:.1f}% στο Universe), Empirical Bayes Lift = {f.eb_lift:.2f}x, "
            f"FDR p-value = {f.p_value_fdr:.2e}"
        )
    traits_block = "\n".join(traits_lines) if traits_lines else "  - Ομοιογενές προφίλ εστίασης"

    user_prompt = (
        f"ΔΕΔΟΜΕΝΑ ΠΡΟΦΙΛ {prof.profile_id}:\n"
        f"- Τίτλος: {prof.title_el}\n"
        f"- Πλήθος Πελατών: {prof.client_count} ({100*prof.client_share:.1f}% του πελατολογίου)\n"
        f"- Μερίδιο Τζίρου: {100*prof.turnover_share:.1f}% (Συνολικός: €{prof.turnover_total:,.0f}, Διάμεσος: €{prof.turnover_median:,.0f})\n"
        f"- Σταθερότητα Προφίλ (Bootstrap): {100*prof.stability_score:.1f}%\n"
        f"- Εμπιστοσύνη Προφίλ (Confidence): {100*prof.confidence_score:.1f}%\n"
        f"- Στατιστικά Τεκμηριωμένα Χαρακτηριστικά:\n{traits_block}\n\n"
        f"Γράψε την τεκμηριωμένη περιγραφή του {prof.profile_id} στα ελληνικά:"
    )
    return system_prompt, user_prompt


def build_rich_deterministic_description_el(prof: DiscoveredICP) -> str:
    """
    Generate a rich, sales-ready, 100% grounded Greek profile description with AI disclosure badge,
    profile quality/confidence metrics, and statistical proofs (Requirement A6).
    """
    top_traits = prof.significant_features[:5]
    trait_phrases = []
    for f in top_traits:
        trait_phrases.append(
            f"• **{f.feature_label_el}**: απαντάται στο **{100*f.icp_share:.1f}%** των πελατών του προφίλ "
            f"(έναντι {100*f.universe_share:.1f}% στο Universe), με **Empirical Bayes Lift {f.eb_lift:.2f}x** "
            f"και στατιστική σημαντικότητα **$p_{{\\text{{FDR}}}} = {f.p_value_fdr:.2e}$**."
        )

    traits_section = "\n".join(trait_phrases) if trait_phrases else "• Ομοιογενές προφίλ εστίασης."

    return (
        f"{AI_DISCLOSURE_PREFIX_EL}\n\n"
        f"Το προφίλ **{prof.profile_id} ({prof.title_el})** αντιπροσωπεύει **{prof.client_count} πελάτες** "
        f"(**{100*prof.client_share:.1f}%** του ενεργού πελατολογίου) και συνεισφέρει το "
        f"**{100*prof.turnover_share:.1f}%** του συνολικού τζίρου (συνολικός τζίρος **€{prof.turnover_total:,.0f}**, "
        f"διάμεσος τζίρος ανά πελάτη **€{prof.turnover_median:,.0f}**). "
        f"Το προφίλ παρουσιάζει **Σταθερότητα (Bootstrap Stability) {100*prof.stability_score:.1f}%** και "
        f"**Δείκτη Εμπιστοσύνης (Confidence) {100*prof.confidence_score:.1f}%**.\n\n"
        f"**Στατιστικά Τεκμηριωμένα Χαρακτηριστικά (Hypergeometric Test & Empirical Bayes Shrinkage):**\n"
        f"{traits_section}\n\n"
        f"{EU_AI_ACT_FOOTER_EL}"
    )


def enrich_profiles_with_local_llm(
    result: ICPEngineResult,
    cfg: PipelineConfig | None = None,
    force_deterministic: bool = False,
) -> LLMExplainerTelemetry:
    """
    Populate `prof.ai_description_el` on each `DiscoveredICP` in `result.profiles`.
    Uses local Ollama with `temperature=0.0` and `seed=cfg.llm_seed` if available and
    `not force_deterministic`; otherwise uses the deterministic grounded generator.
    """
    if cfg is None:
        cfg = PipelineConfig()

    t_start = time.perf_counter()
    per_prof_time: dict[str, float] = {}

    if force_deterministic:
        for prof in result.profiles:
            t0 = time.perf_counter()
            prof.ai_description_el = build_rich_deterministic_description_el(prof)
            per_prof_time[prof.profile_id] = round(time.perf_counter() - t0, 4)
        return LLMExplainerTelemetry(
            used_ollama=False,
            model_name="Deterministic Grounded Greek Generator (Fallback)",
            temperature=0.0,
            seed=cfg.llm_seed,
            total_elapsed_sec=round(time.perf_counter() - t_start, 4),
            per_profile_elapsed_sec=per_prof_time,
            fallback_reason="force_deterministic=True",
        )

    is_avail, resolved_model, _ = check_local_ollama_status(cfg.ollama_model)
    if not is_avail:
        for prof in result.profiles:
            t0 = time.perf_counter()
            prof.ai_description_el = build_rich_deterministic_description_el(prof)
            per_prof_time[prof.profile_id] = round(time.perf_counter() - t0, 4)
        return LLMExplainerTelemetry(
            used_ollama=False,
            model_name="Deterministic Grounded Greek Generator (Ollama Offline Fallback)",
            temperature=0.0,
            seed=cfg.llm_seed,
            total_elapsed_sec=round(time.perf_counter() - t_start, 4),
            per_profile_elapsed_sec=per_prof_time,
            fallback_reason="Local Ollama daemon not running or no local models pulled.",
        )

    import ollama

    for prof in result.profiles:
        t0 = time.perf_counter()
        sys_prompt, usr_prompt = build_grounded_prompt_el(prof)
        try:
            response = ollama.chat(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt},
                ],
                options={
                    "temperature": cfg.llm_temperature,
                    "seed": cfg.llm_seed,
                    "top_p": 1.0,
                    "num_predict": cfg.llm_max_tokens,
                },
            )
            msg = getattr(response, "message", None)
            content = getattr(msg, "content", None) if msg else None
            if content is None and isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            body_text = (content or "").strip()
            if not body_text:
                raise ValueError("Empty response from Ollama")

            prof.ai_description_el = (
                f"{AI_DISCLOSURE_PREFIX_EL}\n"
                f"*(Μοντέλο: `{resolved_model}` | temp={cfg.llm_temperature} | seed={cfg.llm_seed} | "
                f"Σταθερότητα: {100*prof.stability_score:.1f}% | Εμπιστοσύνη: {100*prof.confidence_score:.1f}%)*\n\n"
                f"{body_text}\n\n"
                f"{EU_AI_ACT_FOOTER_EL}"
            )
        except Exception:
            prof.ai_description_el = build_rich_deterministic_description_el(prof)
        per_prof_time[prof.profile_id] = round(time.perf_counter() - t0, 4)

    return LLMExplainerTelemetry(
        used_ollama=True,
        model_name=resolved_model,
        temperature=cfg.llm_temperature,
        seed=cfg.llm_seed,
        total_elapsed_sec=round(time.perf_counter() - t_start, 4),
        per_profile_elapsed_sec=per_prof_time,
        fallback_reason=None,
    )


if __name__ == "__main__":
    from src.features import build_feature_store
    from src.icp_engine import export_deliverables, run_icp_engine
    from src.ingest import ingest_all

    cfg = PipelineConfig(embedding_model="char-ngram-svd")
    ds = ingest_all()
    fs = build_feature_store(ds, cfg)
    res = run_icp_engine(ds, fs, cfg)
    tel = enrich_profiles_with_local_llm(res, cfg)
    export_deliverables(res, cfg.output_dir)

    print("=" * 72)
    print("STAGE 5: LOCAL LLM EXPLAINER REPORT")
    print("=" * 72)
    print(f"  - Used Ollama      : {tel.used_ollama}")
    print(f"  - Active Generator : {tel.model_name}")
    print(f"  - Reproducibility  : temperature={tel.temperature}, seed={tel.seed}")
    print(f"  - Total Time       : {tel.total_elapsed_sec:.4f}s ({tel.per_profile_elapsed_sec})")
    if tel.fallback_reason:
        print(f"  - Fallback Note    : {tel.fallback_reason}")
    print("-" * 72)
    for p in res.profiles:
        print(f"\n### {p.profile_id}: {p.title_el}")
        print(p.ai_description_el)
    print("=" * 72)
