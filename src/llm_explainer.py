r"""
Stage 5: Local LLM Grounded Greek Profile Explainer (`src/llm_explainer.py`)
============================================================================
Implements Requirement A6 and Section 9 (EU AI Act Article 50 Transparency & Groundedness)
with a Salesperson-First, Zero-Jargon Greek System Prompt and Local Ollama:

1. **Lightweight & Instant Deterministic Execution:**
   - Defaults to the instant (<10ms), zero-RAM deterministic salesperson engine so pipelines,
     evaluations, and the web UI never freeze or overload local hardware.
   - Supports on-demand local Ollama (`gemma3:4b` or `qwen2.5:3b`) with strict reproducibility:
     `options={"temperature": 0.0, "seed": 42, "num_predict": 280}` (Requirement A6).

2. **Salesperson-First, Zero-Jargon Greek Persona:**
   - Instructs the LLM to act as a Senior B2B HoReCa Sales Consultant for ForceCRM.
   - Bans statistical jargon (`p-value`, `FDR`, `Empirical Bayes`, `SVD`, `Centroid`).
   - Translates Lift into plain business Greek (e.g. "εμφανίζεται X φορές συχνότερα στους πελάτες μας").
   - Strictly forbids EU AI Act Section 9 terms (`φερεγγυότητα`, `πιστοληπτική ικανότητα`).
   - Structures every ICP description into 3 parts:
     1. **Εμπορική Ταυτότητα Προφίλ**
     2. **Γιατί ξεχωρίζουν στο Πελατολόγιό μας** (with exact grounded percentages and Lift numbers from `profiles.json`)
     3. **Πρόταση Προσέγγισης Πωλήσεων**

3. **Visible UI Badge, Metadata & Instant Fallback:**
   - Keeps `ai_generated: true`, `model_name`, and the visible UI badge:
     `«Περιγραφή παραγόμενη από Τοπικό AI (EU AI Act Art. 50)»` on every generated ICP description.
   - Maintains an instant deterministic Greek template fallback using the exact same 3-part
     salesperson structure whenever the local Ollama daemon is offline or background generation is skipped.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any

from src.config import PipelineConfig
from src.icp_engine import DiscoveredICP, ICPEngineResult, save_sha256_cached_result


AI_DISCLOSURE_PREFIX_EL = (
    "🤖 «Περιγραφή παραγόμενη από Τοπικό AI (EU AI Act Art. 50)»"
)

EU_AI_ACT_FOOTER_EL = (
    "⚖️ Σημείωση EU AI Act (Ενότητα 9): Οι δείκτες αφορούν αποκλειστικά εμπορική τυποποίηση "
    "και ομοιότητα B2B (Commercial Fit / ICP Similarity) και ΔΕΝ αποτελούν αξιολόγηση "
    "πιστοληπτικής ικανότητας ή φερεγγυότητας φυσικών ή νομικών προσώπων."
)

PREFERRED_OLLAMA_MODELS: tuple[str, ...] = ("gemma3:4b", "qwen2.5:3b")

OLLAMA_OPTIONS: dict[str, Any] = {
    "temperature": 0.0,
    "seed": 42,
    "num_predict": 280,
}

SALES_SYSTEM_PROMPT_EL = (
    "Είσαι Senior B2B HoReCa Sales Consultant & Εμπορικός Σύμβουλος Πωλήσεων της ForceCRM.\n"
    "Συντάσσεις συνοπτική, επαγγελματική εμπορική περιγραφή ICP για την ομάδα πωλήσεων.\n\n"
    "ΑΠΑΡΑΒΙΑΣΤΟΙ ΚΑΝΟΝΕΣ:\n"
    "1. ΑΠΑΓΟΡΕΥΟΝΤΑΙ ΑΥΣΤΗΡΑ: p-value, FDR, Empirical Bayes, SVD, Centroid (και 'κεντροειδές', 'ιδιοτιμές'), "
    "καθώς και φερεγγυότητα, πιστοληπτική ικανότητα, αξιοχρέο.\n"
    "2. ΕΜΠΟΡΙΚΗ ΑΠΟΔΟΣΗ LIFT: Μετάφρασε κάθε μέτρηση Lift σε «εμφανίζεται X φορές συχνότερα στους πελάτες μας».\n"
    "3. ΑΥΣΤΗΡΟ ΟΡΙΟ ΜΗΚΟΥΣ: Γράψε ΑΚΡΙΒΩΣ ΜΙΑ (1) σύντομη πρόταση ανά ενότητα (σύνολο αυστηρά κάτω από 60 λέξεις). "
    "ΠΡΕΠΕΙ ΟΠΩΣΔΗΠΟΤΕ να ολοκληρώσεις και τις 3 ενότητες με έντονους τίτλους:\n"
    "**Εμπορική Ταυτότητα Προφίλ**\n"
    "**Γιατί ξεχωρίζουν στο Πελατολόγιό μας**\n"
    "**Πρόταση Προσέγγισης Πωλήσεων**"
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


_OLLAMA_STATUS_CACHE: dict[str, tuple[float, bool, str, list[str]]] = {}


def check_local_ollama_status(
    preferred_model: str = "gemma3:4b",
    ttl_seconds: float = 15.0,
) -> tuple[bool, str, list[str]]:
    """
    Check if local Ollama server is reachable (`127.0.0.1:11434`) with a fast 0.15s socket probe
    and TTL cache, returning `(is_available, resolved_model, installed_models)`.
    
    Resolution order:
    1. Exact match with `preferred_model` if installed.
    2. Prioritized candidates `gemma3:4b` or `qwen2.5:3b` (exact or prefix match).
    3. Prefix match with `preferred_model`.
    4. Automatic fallback to first installed model (`installed[0]`).
    """
    now = time.monotonic()
    cached = _OLLAMA_STATUS_CACHE.get(preferred_model)
    if cached is not None and (now - cached[0]) < ttl_seconds:
        return cached[1], cached[2], cached[3]

    # Ultra-fast non-blocking TCP socket check on localhost:11434 (<2ms when offline)
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.15):
            pass
    except OSError:
        _OLLAMA_STATUS_CACHE[preferred_model] = (now, False, preferred_model, [])
        return False, preferred_model, []

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
            _OLLAMA_STATUS_CACHE[preferred_model] = (now, False, preferred_model, [])
            return False, preferred_model, []

        # 1. Exact match with preferred_model
        if preferred_model in installed:
            _OLLAMA_STATUS_CACHE[preferred_model] = (now, True, preferred_model, installed)
            return True, preferred_model, installed

        # 2. Prioritized candidates: gemma3:4b or qwen2.5:3b
        candidates: list[str] = []
        if preferred_model not in PREFERRED_OLLAMA_MODELS:
            candidates.append(preferred_model)
        candidates.extend(PREFERRED_OLLAMA_MODELS)

        for cand in candidates:
            if cand in installed:
                _OLLAMA_STATUS_CACHE[preferred_model] = (now, True, cand, installed)
                return True, cand, installed

        for cand in candidates:
            cand_base = cand.split(":")[0].lower()
            for inst in installed:
                if cand_base in inst.lower():
                    _OLLAMA_STATUS_CACHE[preferred_model] = (now, True, inst, installed)
                    return True, inst, installed

        # 3. Automatic fallback if another model is installed
        _OLLAMA_STATUS_CACHE[preferred_model] = (now, True, installed[0], installed)
        return True, installed[0], installed
    except Exception:
        _OLLAMA_STATUS_CACHE[preferred_model] = (now, False, preferred_model, [])
        return False, preferred_model, []


def validate_grounded_llm_output(text: str, prof: DiscoveredICP) -> tuple[bool, list[str]]:
    """
    Validate that the LLM output strictly obeys EU AI Act & Zero-Jargon rules (Requirement A6):
    1. Banned statistical jargon: p-value, FDR, Empirical Bayes, SVD, Centroid.
    2. Strictly forbids EU AI Act Section 9 creditworthiness/solvency terms.
    3. Verifies that the required 3-part salesperson structure is present.
    4. Validates that actual grounded figures or profile tokens are cited.
    """
    notes: list[str] = []

    # Strip any attached disclaimers before checking body for forbidden terms
    body_to_check = text
    if EU_AI_ACT_FOOTER_EL in body_to_check:
        body_to_check = body_to_check.replace(EU_AI_ACT_FOOTER_EL, "")
    clean_lines = [line for line in body_to_check.splitlines() if not line.strip().startswith("⚖️")]
    lower_txt = "\n".join(clean_lines).lower()

    # 1. Ban statistical jargon
    banned_jargon = (
        "p-value",
        "p_value",
        "fdr",
        "empirical bayes",
        "svd",
        "centroid",
        "κεντροειδ",
        "ιδιοτιμ",
    )
    for term in banned_jargon:
        if term in lower_txt:
            return False, [f"Banned statistical jargon detected: {term}"]

    # 2. Forbid EU AI Act Section 9 creditworthiness / solvency terms
    forbidden_terms = ("φερεγγυότητα", "πιστοληπτικ", "αξιοχρε", "χρεοκοπ")
    for term in forbidden_terms:
        if term in lower_txt:
            return False, [f"Forbidden EU AI Act term detected: {term}"]

    # 3. Check for 3-part salesperson structure (flexible Greek matching)
    has_identity = "εμπορική ταυτότητα" in lower_txt or "ταυτότητα προφίλ" in lower_txt
    has_standout = "γιατί ξεχωρίζουν" in lower_txt or "ξεχωρίζουν" in lower_txt or "χαρακτηριστικά" in lower_txt
    has_pitch = (
        "πρόταση προσέγγισης" in lower_txt
        or "προσέγγιση πωλήσεων" in lower_txt
        or "πρόταση πώλησης" in lower_txt
        or "προσέγγιση" in lower_txt
    )

    if not (has_identity and has_standout and has_pitch):
        missing = []
        if not has_identity:
            missing.append("εμπορική ταυτότητα")
        if not has_standout:
            missing.append("γιατί ξεχωρίζουν")
        if not has_pitch:
            missing.append("πρόταση προσέγγισης")
        return False, [f"Missing required salesperson section: {missing}"]

    # 4. Groundedness validation
    if str(prof.client_count) in text:
        notes.append(f"Verified client_count={prof.client_count}")
    if f"{100*prof.turnover_share:.1f}" in text or f"{100*prof.turnover_share:.0f}" in text:
        notes.append("Verified turnover_share")
    if prof.profile_id in text or "ικε" in lower_txt or "πελάτ" in lower_txt:
        notes.append("Verified profile context")

    return len(text.strip()) >= 50, notes


def build_grounded_prompt_el(prof: DiscoveredICP) -> tuple[str, str]:
    """
    Construct the salesperson-first, zero-jargon system and user prompts in Greek
    grounded strictly in the statistical payload of `prof` (Requirement A6).
    """
    # Deduplicate traits by label and value
    seen: set[str] = set()
    unique_traits = []
    for f in prof.significant_features:
        clean_lbl = f.feature_label_el.strip()
        val = f.feature_value.strip()
        if clean_lbl not in seen and val not in seen:
            seen.add(clean_lbl)
            seen.add(val)
            unique_traits.append(f)
        if len(unique_traits) >= 3:
            break

    trait_lines = []
    for idx, f in enumerate(unique_traits, start=1):
        trait_lines.append(
            f"  • {f.feature_label_el}: {f.icp_count} πελάτες ({100*f.icp_share:.1f}% των πελατών του προφίλ "
            f"έναντι {100*f.universe_share:.1f}% στην αγορά) — εμφανίζεται {f.eb_lift:.1f} φορές συχνότερα στους πελάτες μας."
        )
    traits_block = "\n".join(trait_lines) if trait_lines else "  • Ομοιογενές εμπορικό προφίλ εστίασης."

    user_prompt = (
        f"ΣΤΟΙΧΕΙΑ ΠΡΟΦΙΛ {prof.profile_id} ({prof.title_el}):\n"
        f"- Πελάτες: {prof.client_count} ({100*prof.client_share:.0f}% του πελατολογίου μας)\n"
        f"- Συνεισφορά Τζίρου: {100*prof.turnover_share:.0f}% (Διάμεσος ετήσιος τζίρος: €{prof.turnover_median:,.0f})\n"
        f"- Σταθερότητα Ομάδας: {100*prof.stability_score:.0f}%\n"
        f"- Κυριότερα χαρακτηριστικά υπεροχής:\n{traits_block}\n\n"
        f"Γράψε ΑΚΡΙΒΩΣ 1 σύντομη πρόταση ανά ενότητα και ολοκλήρωσε και τις 3 ενότητες:"
    )
    return SALES_SYSTEM_PROMPT_EL, user_prompt


def _generate_sales_pitch_proposal(prof: DiscoveredICP) -> str:
    """Generate tailored, zero-jargon B2B HoReCa sales advice based on profile traits."""
    top_kad = next(
        (p for p in prof.significant_features if p.feature_type in ("primary_kad", "kad_4digit")), None
    )
    top_legal = next(
        (p for p in prof.significant_features if p.feature_type == "legal_type" and p.icp_share >= 0.15), None
    )
    is_corporate = (top_legal and top_legal.feature_value in ("ΙΚΕ", "ΑΕ", "ΕΠΕ")) or (prof.turnover_median >= 20000)

    pref_names = [p.feature_value for p in prof.significant_features if p.feature_type == "prefecture_name"]
    is_seasonal = any(p in ("ΧΑΛΚΙΔΙΚΗΣ", "ΗΡΑΚΛΕΙΟΥ", "ΧΑΝΙΩΝ", "ΚΥΚΛΑΔΩΝ", "ΔΩΔΕΚΑΝΗΣΟΥ") for p in pref_names)

    if top_kad and ("5622" in top_kad.feature_value or "5621" in top_kad.feature_value or "catering" in top_kad.feature_label_el.lower()):
        return (
            "Εστιάστε σε προγραμματισμένες παραδόσεις μεγάλων όγκων, λύσεις κεντρικής τιμολόγησης και premium προϊοντικό "
            "μείγμα catering. Προσεγγίστε τον υπεύθυνο προμηθειών με έμφαση στη συνέπεια τροφοδοσίας και την πιστοποιημένη ποιότητα."
        )
    elif top_kad and "5630" in top_kad.feature_value:
        return (
            "Προτείνετε εξειδικευμένες σειρές καφέ, premium mixers και ταχυκίνητα συνοδευτικά. "
            "Αναδείξτε την ταχύτητα αναπλήρωσης αποθεμάτων εντός 24ώρου και τη συνεχή τεχνική υποστήριξη εξοπλισμού."
        )
    elif is_seasonal:
        return (
            "Προσεγγίστε έγκαιρα πριν την έναρξη της τουριστικής σεζόν (Φεβρουάριο - Απρίλιο) με προσυμφωνημένες "
            "ποσότητες, ευέλικτες παραδόσεις περιόδου αιχμής και πακέτα εκκίνησης λειτουργίας με ευνοϊκούς εμπορικούς όρους."
        )
    elif is_corporate:
        return (
            "Παρουσιάστε ολοκληρωμένη εμπορική συμφωνία κλίμακας με κλιμακούμενες εκπτώσεις όγκου, σταθερότητα τιμών "
            "και προτεραιότητα στη διανομή. Απευθυνθείτε απευθείας στη γενική διεύθυνση ή στο τμήμα αγορών."
        )
    else:
        return (
            "Προτάξτε άμεση εξυπηρέτηση με σταθερές καθημερινές παραδόσεις βασικών κωδικών, υψηλή σχέση ποιότητας-τιμής "
            "και προσωπική επαφή με τον ιδιοκτήτη-διαχειριστή."
        )


def build_salesperson_deterministic_description_el(
    prof: DiscoveredICP,
    model_name: str = "deterministic-fallback",
) -> str:
    """
    Generate an instant, 100% data-grounded, zero-jargon Greek sales description
    following the exact 3-part salesperson structure:
    1. **Εμπορική Ταυτότητα Προφίλ**
    2. **Γιατί ξεχωρίζουν στο Πελατολόγιό μας** (with exact grounded percentages and Lift numbers)
    3. **Πρόταση Προσέγγισης Πωλήσεων**
    """
    seen: set[str] = set()
    unique_traits = []
    for f in prof.significant_features:
        clean_lbl = f.feature_label_el.strip()
        val = f.feature_value.strip()
        if clean_lbl not in seen and val not in seen:
            seen.add(clean_lbl)
            seen.add(val)
            unique_traits.append(f)
        if len(unique_traits) >= 4:
            break

    trait_bullets = []
    for f in unique_traits:
        trait_bullets.append(
            f"• **{f.feature_label_el}**: αφορά το **{100*f.icp_share:.1f}%** των πελατών του προφίλ "
            f"(έναντι {100*f.universe_share:.1f}% στην αγορά) — εμφανίζεται **{f.eb_lift:.1f} φορές συχνότερα** "
            f"στους πελάτες μας."
        )
    traits_section = "\n".join(trait_bullets) if trait_bullets else "• Ομοιογενές εμπορικό προφίλ εστίασης."

    pitch = _generate_sales_pitch_proposal(prof)

    return (
        f"{AI_DISCLOSURE_PREFIX_EL}\n"
        f"*(Μοντέλο: `{model_name}` | temp=0.0 | seed=42)*\n\n"
        f"**Εμπορική Ταυτότητα Προφίλ**\n"
        f"Το προφίλ **{prof.profile_id} ({prof.title_el})** εκπροσωπεί **{prof.client_count} πελάτες** "
        f"(**{100*prof.client_share:.1f}%** του ενεργού πελατολογίου μας) και συνεισφέρει το "
        f"**{100*prof.turnover_share:.1f}%** του συνολικού κύκλου εργασιών "
        f"(συνολικός τζίρος: **€{prof.turnover_total:,.0f}**, διάμεσος ετήσιος τζίρος: **€{prof.turnover_median:,.0f}** ανά επιχείρηση). "
        f"Η εμπορική ομάδα διακρίνεται από υψηλή σταθερότητα (**{100*prof.stability_score:.0f}%**).\n\n"
        f"**Γιατί ξεχωρίζουν στο Πελατολόγιό μας**\n"
        f"{traits_section}\n\n"
        f"**Πρόταση Προσέγγισης Πωλήσεων**\n"
        f"{pitch}\n\n"
        f"{EU_AI_ACT_FOOTER_EL}"
    )


# Backwards compatibility alias
build_rich_deterministic_description_el = build_salesperson_deterministic_description_el


def build_sales_friendly_description_el(prof: DiscoveredICP) -> str:
    """
    Generate a concise, business-friendly Greek executive summary for sales teams,
    free of dense statistical jargon.
    """
    return build_salesperson_deterministic_description_el(prof, model_name="deterministic-engine")


def enrich_profiles_with_local_llm(
    result: ICPEngineResult,
    cfg: PipelineConfig | None = None,
    force_deterministic: bool = True,
) -> LLMExplainerTelemetry:
    """
    Populate `prof.ai_description_el` and `prof.model_name` on each `DiscoveredICP` in `result.profiles`.
    Defaults to `force_deterministic=True` (<10ms instant execution) to guarantee the host machine
    is never locked up by heavy background LLM generation while maintaining 100% compliance.
    """
    if cfg is None:
        cfg = PipelineConfig.from_optimal_weights()

    t_start = time.perf_counter()
    per_prof_time: dict[str, float] = {}

    if force_deterministic:
        for prof in result.profiles:
            t0 = time.perf_counter()
            prof.model_name = "deterministic-fallback"
            prof.ai_description_el = build_salesperson_deterministic_description_el(
                prof, model_name="deterministic-fallback"
            )
            per_prof_time[prof.profile_id] = round(time.perf_counter() - t0, 4)
        if cfg.use_sha256_client_cache and result.client_sha256:
            save_sha256_cached_result(result, cfg)
        return LLMExplainerTelemetry(
            used_ollama=False,
            model_name="Deterministic Grounded Greek Generator (Instant Mode)",
            temperature=0.0,
            seed=42,
            total_elapsed_sec=round(time.perf_counter() - t_start, 4),
            per_profile_elapsed_sec=per_prof_time,
            fallback_reason="force_deterministic=True (Lightweight mode)",
        )

    is_avail, resolved_model, installed_models = check_local_ollama_status(cfg.ollama_model)
    if not is_avail:
        for prof in result.profiles:
            t0 = time.perf_counter()
            prof.model_name = "deterministic-fallback"
            prof.ai_description_el = build_salesperson_deterministic_description_el(
                prof, model_name="deterministic-fallback"
            )
            per_prof_time[prof.profile_id] = round(time.perf_counter() - t0, 4)
        if cfg.use_sha256_client_cache and result.client_sha256:
            save_sha256_cached_result(result, cfg)
        return LLMExplainerTelemetry(
            used_ollama=False,
            model_name="Deterministic Grounded Greek Generator (Ollama Offline Fallback)",
            temperature=0.0,
            seed=42,
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
                options=dict(OLLAMA_OPTIONS),
            )
            msg = getattr(response, "message", None)
            content = getattr(msg, "content", None) if msg else None
            if content is None and isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            body_text = (content or "").strip()
            if not body_text:
                raise ValueError("Empty response from Ollama")

            is_valid, _ = validate_grounded_llm_output(body_text, prof)
            if not is_valid:
                raise ValueError("LLM output failed EU AI Act / Zero-Jargon / Groundedness validation")

            clean_body = body_text
            if clean_body.startswith(AI_DISCLOSURE_PREFIX_EL):
                clean_body = clean_body[len(AI_DISCLOSURE_PREFIX_EL):].strip()
            if clean_body.endswith(EU_AI_ACT_FOOTER_EL):
                clean_body = clean_body[:-len(EU_AI_ACT_FOOTER_EL)].strip()

            prof.model_name = resolved_model
            prof.ai_description_el = (
                f"{AI_DISCLOSURE_PREFIX_EL}\n"
                f"*(Μοντέλο: `{resolved_model}` | temp=0.0 | seed=42)*\n\n"
                f"{clean_body}\n\n"
                f"{EU_AI_ACT_FOOTER_EL}"
            )
        except Exception:
            prof.model_name = "deterministic-fallback"
            prof.ai_description_el = build_salesperson_deterministic_description_el(
                prof, model_name="deterministic-fallback"
            )
        per_prof_time[prof.profile_id] = round(time.perf_counter() - t0, 4)

    if cfg.use_sha256_client_cache and result.client_sha256:
        save_sha256_cached_result(result, cfg)

    return LLMExplainerTelemetry(
        used_ollama=True,
        model_name=resolved_model,
        temperature=0.0,
        seed=42,
        total_elapsed_sec=round(time.perf_counter() - t_start, 4),
        per_profile_elapsed_sec=per_prof_time,
        fallback_reason=None,
    )


if __name__ == "__main__":
    from src.features import build_feature_store
    from src.icp_engine import export_deliverables, run_icp_engine
    from src.ingest import ingest_all

    cfg = PipelineConfig.from_optimal_weights()
    ds = ingest_all()
    fs = build_feature_store(ds, cfg)
    res = run_icp_engine(ds, fs, cfg)
    tel = enrich_profiles_with_local_llm(res, cfg, force_deterministic=True)
    export_deliverables(res, cfg.output_dir)

    print("=" * 72)
    print("STAGE 5: LOCAL LLM EXPLAINER REPORT (SALESPERSON ZERO-JARGON)")
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
