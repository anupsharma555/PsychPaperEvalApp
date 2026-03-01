from __future__ import annotations

import json
import multiprocessing as mp
import os
from queue import Empty as QueueEmpty
import re
from typing import Any, Callable

from app.core.config import settings
from app.services.author_utils import sanitize_author_list
from app.services.analysis.llm import chat_text_deep
from app.services.analysis.schemas import StructuredDossierV2
from app.services.analysis.utils import (
    clamp_confidence,
    extract_json,
    max_chars_for_ctx,
    summarize_packet_statements,
    truncate_text,
)

SYNTHESIS_SYSTEM = """
You are compiling a multimodal peer-review dossier from normalized evidence packets.
The executive_summary must be one concise paragraph that explicitly includes:
introduction/objective, methods, key results, discussion/limitations, and conclusion.
Return JSON only with this schema:
{
  "executive_summary": "single paragraph summary",
  "methods_strengths": ["bullet"],
  "methods_weaknesses": ["bullet"],
  "reproducibility_ethics": ["bullet"],
  "uncertainty_gaps": ["bullet"]
}
""".strip()

VERIFIER_SYSTEM = """
You are verifier mode for an executive research report.
Review the draft narrative against provided evidence summaries and cross-modal checks.
Correct any unsupported, exaggerated, or internally inconsistent statements.
Do not invent claims; if evidence is weak, make wording more cautious.
Ensure executive_summary remains concise and explicitly covers:
introduction/objective, methods, key results, discussion/limitations, and conclusion.
Return JSON only with this schema:
{
  "executive_summary": "single paragraph summary",
  "methods_strengths": ["bullet"],
  "methods_weaknesses": ["bullet"],
  "reproducibility_ethics": ["bullet"],
  "uncertainty_gaps": ["bullet"]
}
""".strip()

SECTION_EXTRACTION_SYSTEM = """
You are a structured, high-fidelity scientific paper extraction engine.
You must extract only explicit statements from provided evidence rows.
Do not summarize across sections, do not critique, and do not invent details.
Keep section fidelity strict: Introduction, Methods, Results, Discussion, Conclusion.
Use only evidence refs that appear in input rows.
Output JSON only with this schema:
{
  "introduction": [{"statement": "one sentence", "evidence_refs": ["anchor"], "kind": "objective|gap|rationale|hypothesis|other"}],
  "methods": [{"statement": "one sentence", "evidence_refs": ["anchor"], "kind": "participants|design|measure|acquisition|preprocessing|statistics|correction|sensitivity|supplementary|other"}],
  "results": [{"statement": "one sentence", "evidence_refs": ["anchor"], "kind": "primary|effect|statistical_support|null|sensitivity|secondary|supplementary|other"}],
  "discussion": [{"statement": "one sentence", "evidence_refs": ["anchor"], "kind": "interpretation|mechanism|limitations|implications|other"}],
  "conclusion": [{"statement": "one sentence", "evidence_refs": ["anchor"], "kind": "takeaway|claim_strength|future|clinical|other"}]
}
Rules:
- Keep statistics/values exactly as stated if present.
- Do not include bullets without evidence refs.
- If uncertain about section membership, omit the bullet.
- Do not emit placeholders like "not reported" unless explicitly stated in evidence rows.
""".strip()

METHOD_ANCHOR_RE = re.compile(
    r"\b(method|methods|methodology|materials|protocol|study design|design|participants?)\b",
    re.IGNORECASE,
)
METHOD_SIGNAL_RE = re.compile(
    r"\b("
    r"randomi[sz]ed|cohort|case[- ]control|cross[- ]sectional|pragmatic trial|"
    r"double[- ]blind|single[- ]blind|allocation|sample size|power analys(?:is|es)|"
    r"inclusion criteria|exclusion criteria|participant|intervention|comparator|control arm|"
    r"endpoint|outcome measure|instrument|scale|questionnaire|follow[- ]up|"
    r"regression|mixed[- ]effects|bayesian|covariate|confound|imputation|sensitivity analys(?:is|es)|"
    r"pre[- ]register|preregistration|code availability|data availability"
    r")\b",
    re.IGNORECASE,
)
OUTCOME_SUMMARY_RE = re.compile(
    r"\b("
    r"results?\s+(?:show|indicat|suggest)|"
    r"we found|the study found|key finding|"
    r"improv(?:ed|ement)|reduc(?:ed|tion)|increas(?:ed|e)"
    r")\b",
    re.IGNORECASE,
)
RESULTS_ANCHOR_RE = re.compile(r"\b(results?|discussion|conclusion)\b", re.IGNORECASE)
ACCESS_LIMITED_NOTE_RE = re.compile(r"\b(access[- ]limited|paywall|subscription|publisher)\b", re.IGNORECASE)
INTRO_SECTION_RE = re.compile(r"\b(intro|introduction|background|rationale|objective|aim|hypothesis)\b", re.IGNORECASE)
METHODS_SECTION_RE = re.compile(
    r"\b(method|methods|methodology|materials|protocol|design|participants?|analysis plan|statistics?)\b",
    re.IGNORECASE,
)
RESULTS_SECTION_RE = re.compile(r"\b(result|results|finding|outcome|effect|association|improvement)\b", re.IGNORECASE)
DISCUSSION_SECTION_RE = re.compile(
    r"\b(discussion|interpretation|implication|limitation|limitations|generalizability|context|to date|in our study|consistent with|may reflect)\b",
    re.IGNORECASE,
)
CONCLUSION_SECTION_RE = re.compile(
    r"\b(conclusion|conclusions|conclude|summary|takeaway|overall|future research|longitudinal)\b",
    re.IGNORECASE,
)
SECTION_LABELS = {"introduction", "methods", "results", "discussion", "conclusion", "unknown"}
METHOD_LIKE_RE = re.compile(
    r"\b(participants?|scanner|acquisition|processing|covariates?|post hoc|seed-based|"
    r"was used to test|model(?:ing)?|protocol|procedure|inclusion|exclusion|measured using|analysis was conducted)\b",
    re.IGNORECASE,
)
CONCRETE_RESULT_RE = re.compile(
    r"\b("
    r"p\s*[<=>]\s*0?\.\d+|"
    r"t\s*=\s*-?\d+(?:\.\d+)?|f\s*=\s*-?\d+(?:\.\d+)?|z\s*=\s*-?\d+(?:\.\d+)?|"
    r"\d+(?:\.\d+)?\s*(?:%|ms|mm|sd|ci|or|hr)\b|"
    r"increased|decreased|higher|lower|hyperconnectivity|hypoconnectivity|dysconnectivity|"
    r"associated with|correlated with|significant|found|"
    r"identified\s+(?:foci|clusters|regions|differences)"
    r")\b",
    re.IGNORECASE,
)
HIGH_SIGNAL_RESULT_RE = re.compile(
    r"\b("
    r"identified|revealed|demonstrated|showed|stratif(?:y|ication)|biotype|cluster(?:ing)?|subtype|"
    r"replicat(?:ed|ion)|validation cohort|normative model(?:ing)?|deviation pattern|network pattern"
    r")\b",
    re.IGNORECASE,
)
GENERIC_VISUAL_RE = re.compile(
    r"^(figure|table|supplement(?:ary)?\s+figure|supplement(?:ary)?\s+table)\b.*\b(shows?|displays?|illustrates?|depicts?|presents?)\b",
    re.IGNORECASE,
)
VISUAL_ANNOTATION_RE = re.compile(
    r"\b("
    r"solid dots indicate|outer circles?|distance from center|each axis represents|"
    r"panel [a-h]|axis represents|color bar|heatmap scale|"
    r"nodes? exhibiting|rings? indicate|circle size indicates"
    r")\b",
    re.IGNORECASE,
)
INTRO_FALLBACK_RE = re.compile(
    r"\b(background|objective|aim|rationale|hypothesis|motivation|transdiagnostic)\b",
    re.IGNORECASE,
)
CONCLUSION_FALLBACK_RE = re.compile(
    r"\b(in conclusion|overall|these findings|these results (?:suggest|support|emphasize)|this study suggests|conclude|conclusion|future research|longitudinal|replicat(?:ed|ion)|generalizab(?:ility|le))\b",
    re.IGNORECASE,
)
CONCLUSION_RECOVERY_RE = re.compile(
    r"\b("
    r"in conclusion|overall|main takeaway|the study (?:suggests|supports|indicates)|"
    r"these (?:findings|results) (?:suggest|support|indicate|emphasize)|"
    r"clinical (?:implication|relevance)|implications?|"
    r"future (?:work|research)|longitudinal|intervention(?:s)?|target(?:ing)?|"
    r"caution|interpret(?:ation|ive)?|may reflect|shared (?:neural|network)|transdiagnostic|"
    r"replicat(?:ed|ion)|generalizab(?:ility|le)"
    r")\b",
    re.IGNORECASE,
)
DISCUSSION_FALLBACK_RE = re.compile(
    r"\b(to date|in our study|in contrast|consistent with|may reflect|limitations?|interpret|implication)\b",
    re.IGNORECASE,
)
CONCLUSION_SIGNAL_RE = re.compile(
    r"\b("
    r"overall|in conclusion|these findings|these results|taken together|collectively|"
    r"suggest|support|indicat|implication|clinical|future|generalizab|replicat|"
    r"highlight|underscore|caution|warrant"
    r")\b",
    re.IGNORECASE,
)
NOISE_STATEMENT_RE = re.compile(
    r"\b("
    r"ajp\.psychiatryonline\.org|doi:|received [a-z]+ \d{1,2}, \d{4}|revision received|accepted [a-z]+ \d{1,2}, \d{4}|"
    r"published online|address correspondence|presented at|supported by nih|financial relationships|"
    r"^articles$|^sharma et al\.?$|^reward deficits across mood and psychotic disorders$|table\s+\d+|figure\s+\d+"
    r")\b",
    re.IGNORECASE,
)
LAYOUT_ARTIFACT_RE = re.compile(
    r"\b("
    r"left right|right left|cortical projection|brain (?:surface|map)|color bar|"
    r"axial|coronal|sagittal|panel [a-z]|projection displaying|surface rendering"
    r")\b",
    re.IGNORECASE,
)
INTRO_EXCLUDE_RE = re.compile(
    r"\b(conclusion|conclude|discussion|limitation|in conclusion|we found|results showed|"
    r"significant|p\s*[<=>]\s*0?\.\d+)\b",
    re.IGNORECASE,
)
CONFIDENCE_INLINE_RE = re.compile(
    r"\s*(?:\((?:model\s+)?confidence[:\s]*\d{1,3}(?:\.\d+)?%?\)|(?:model\s+)?confidence[:\s]+\d{1,3}(?:\.\d+)?%|\(\d{1,3}(?:\.\d+)?%\))\s*",
    re.IGNORECASE,
)
LEADING_CITATION_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?:
        [\(\[\{]?\s*
        \d{1,3}(?:\s*[-–]\s*\d{1,3})?
        (?:\s*(?:,|;)\s*\d{1,3}(?:\s*[-–]\s*\d{1,3})?)*
        \s*[\)\]\}]?
        (?:\s*(?:,|;|:))?
        \s*
    )
    (?=[A-Z])
    """,
    re.VERBOSE,
)
CANONICAL_STATEMENT_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
FRAGMENT_START_RE = re.compile(
    r"^\s*(?:was|were|is|are|been|being|as in|for a list|finally|next\b|and\b|or\b|but\b)\b",
    re.IGNORECASE,
)

METHODS_COMPACT_VERSION = 1
SECTIONS_COMPACT_VERSION = 1
METHODS_FOUND_CONFIDENCE_THRESHOLD = 0.55
SECTIONS_FOUND_CONFIDENCE_THRESHOLD = 0.55
RESULT_SUPPORT_FOUND_CONFIDENCE_THRESHOLD = 0.45
MAX_COVERAGE_REFS = 256
MAX_COVERAGE_REF_LEN = 96
MAX_META_STR_LEN = 512
METHOD_STATEMENT_MAX_CHARS = 160
METHOD_NOT_FOUND_STATEMENT = "N/A - not found in parsed text."
METHOD_ACCESS_LIMITED_STATEMENT = "N/A - source appears access-limited; upload full PDF for reliable extraction."
SECTION_NOT_FOUND_STATEMENT = "N/A - not found in parsed text."
SECTION_ACCESS_LIMITED_STATEMENT = "N/A - source appears access-limited; upload full PDF for reliable extraction."
SECTION_EXTRACTION_LIMITS: dict[str, int] = {
    "introduction": 8,
    "methods": 16,
    "results": 20,
    "discussion": 10,
    "conclusion": 8,
}
INCOMPLETE_TAIL_TOKENS = {
    "and",
    "or",
    "with",
    "without",
    "to",
    "of",
    "in",
    "on",
    "for",
    "by",
    "from",
    "as",
    "than",
    "that",
    "which",
    "whose",
    "while",
    "when",
    "where",
    "because",
    "therefore",
    "however",
    "thus",
    "including",
    "beginning",
    "using",
    "via",
    "consistent",
}
ALLOWED_SHORT_FINAL_TOKENS = {"adhd", "fdr", "ci", "sd", "msn", "auc", "sex", "age"}

METHODS_COMPACT_SLOTS: list[dict[str, Any]] = [
    {
        "slot_key": "study_design",
        "label": "Study Design",
        "keywords": ["study design", "randomized", "trial", "cohort", "case-control", "cross-sectional", "cwas"],
        "category_hints": {"methods", "stats"},
    },
    {
        "slot_key": "sample_population",
        "label": "Sample/Population",
        "keywords": ["participant", "sample", "subjects", "cohort", "diagnostic groups", "n="],
        "category_hints": {"methods", "clinical"},
    },
    {
        "slot_key": "inclusion_criteria",
        "label": "Inclusion Criteria",
        "keywords": ["inclusion criteria", "included", "eligib", "enrolled"],
        "category_hints": {"methods", "clinical"},
    },
    {
        "slot_key": "exclusion_criteria",
        "label": "Exclusion Criteria",
        "keywords": ["exclusion criteria", "excluded", "omit", "dropout"],
        "category_hints": {"methods", "clinical"},
    },
    {
        "slot_key": "intervention_or_exposure",
        "label": "Intervention/Exposure",
        "keywords": ["intervention", "exposure", "comparator", "control", "placebo", "treatment"],
        "category_hints": {"methods", "clinical"},
    },
    {
        "slot_key": "outcomes_measures",
        "label": "Outcomes/Measures",
        "keywords": ["outcome", "endpoint", "measure", "scale", "questionnaire", "bas", "instrument"],
        "category_hints": {"methods", "clinical", "stats"},
    },
    {
        "slot_key": "data_acquisition_protocol",
        "label": "Data Acquisition Protocol",
        "keywords": ["scanner", "mri", "sequence", "acquisition", "protocol", "imaging"],
        "category_hints": {"methods"},
    },
    {
        "slot_key": "statistical_model",
        "label": "Statistical Model",
        "keywords": ["regression", "model", "mdmr", "mixed-effects", "bayesian", "analysis"],
        "category_hints": {"stats", "methods"},
    },
    {
        "slot_key": "covariates",
        "label": "Covariates",
        "keywords": ["covariate", "adjusted", "controlling", "nuisance covariates", "confound"],
        "category_hints": {"stats", "methods"},
    },
    {
        "slot_key": "missing_data_and_sensitivity",
        "label": "Missing Data/Sensitivity",
        "keywords": ["missing data", "imputation", "sensitivity", "robustness", "complete-case"],
        "category_hints": {"stats", "methods", "limitations"},
    },
    {
        "slot_key": "reproducibility_transparency",
        "label": "Reproducibility/Transparency",
        "keywords": ["preregister", "pre-register", "code availability", "data availability", "open materials"],
        "category_hints": {"reproducibility", "methods"},
    },
    {
        "slot_key": "method_limitations",
        "label": "Method Limitations",
        "keywords": ["limitation", "bias", "underpowered", "confounding", "generalizability"],
        "category_hints": {"limitations", "methods"},
    },
]

METHOD_SLOT_CATEGORY: dict[str, str] = {
    "study_design": "methods",
    "sample_population": "methods",
    "inclusion_criteria": "methods",
    "exclusion_criteria": "methods",
    "intervention_or_exposure": "clinical",
    "outcomes_measures": "methods",
    "data_acquisition_protocol": "methods",
    "statistical_model": "stats",
    "covariates": "stats",
    "missing_data_and_sensitivity": "stats",
    "reproducibility_transparency": "reproducibility",
    "method_limitations": "limitations",
}
STRICT_METHOD_SLOT_KEYWORDS = {
    "inclusion_criteria",
    "exclusion_criteria",
    "intervention_or_exposure",
    "covariates",
    "missing_data_and_sensitivity",
    "reproducibility_transparency",
}

SECTION_COMPACT_SLOTS: dict[str, list[dict[str, Any]]] = {
    "introduction": [
        {
            "slot_key": "research_problem",
            "label": "Research Problem",
            "keywords": ["anhedonia", "reward", "problem", "motivation", "disability", "transdiagnostic"],
        },
        {
            "slot_key": "knowledge_gap",
            "label": "Knowledge Gap",
            "keywords": ["gap", "limited", "few studies", "unclear", "not fully resolved", "diminishes ability"],
        },
        {
            "slot_key": "objective_hypothesis",
            "label": "Objective/Hypothesis",
            "keywords": ["objective", "aim", "hypothesized", "we hypothesized", "goal"],
        },
    ],
    "results": [
        {
            "slot_key": "primary_findings",
            "label": "Primary Findings",
            "keywords": ["identified", "revealed", "found", "dysconnectivity", "associated"],
        },
        {
            "slot_key": "effect_direction_magnitude",
            "label": "Effect Direction/Magnitude",
            "keywords": ["increased", "decreased", "higher", "lower", "percent", "effect"],
        },
        {
            "slot_key": "statistical_support",
            "label": "Statistical Support",
            "keywords": ["p<", "p <", "p=", "significant", "confidence interval", "pseudo-f"],
        },
        {
            "slot_key": "subgroup_sensitivity",
            "label": "Subgroup/Sensitivity",
            "keywords": ["within-group", "subgroup", "diagnostic category", "sensitivity", "specificity"],
        },
        {
            "slot_key": "null_adverse_findings",
            "label": "Null/Adverse Findings",
            "keywords": ["did not", "no", "not identify", "underpowered", "null"],
        },
    ],
    "discussion": [
        {
            "slot_key": "interpretation",
            "label": "Interpretation",
            "keywords": ["interpret", "suggest", "supports", "implicate", "mechanism"],
        },
        {
            "slot_key": "limitations",
            "label": "Limitations",
            "keywords": ["limitation", "cross-sectional", "causal", "underpowered", "bias"],
        },
        {
            "slot_key": "generalizability",
            "label": "Generalizability",
            "keywords": ["generalizability", "heterogeneous", "across diagnostic", "population"],
        },
        {
            "slot_key": "implications",
            "label": "Implications",
            "keywords": ["implication", "future work", "intervention", "clinical relevance", "rdoc"],
        },
    ],
    "conclusion": [
        {
            "slot_key": "main_takeaway",
            "label": "Main Takeaway",
            "keywords": ["conclusion", "overall", "takeaway", "corroborate", "emphasize"],
        },
        {
            "slot_key": "claim_strength_caution",
            "label": "Claim Strength/Caution",
            "keywords": ["suggest", "may", "caution", "should", "provisional"],
        },
    ],
}

SECTION_COMPACT_MIN_FOUND_FOR_RESCUE: dict[str, int] = {
    "introduction": 1,
    "results": 2,
    "discussion": 3,
    "conclusion": 2,
}


def synthesize_report(
    text_report: dict[str, Any],
    table_report: dict[str, Any],
    figure_report: dict[str, Any],
    supp_report: dict[str, Any],
    reconcile_report: dict[str, Any],
    paper_meta: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    text_chunk_records: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    _emit_synthesis_progress(
        progress_callback,
        0.05,
        "Synthesizing executive report: preparing payload",
    )
    _trace_synthesis_step("synthesize:start")
    safe_paper_meta = _sanitize_paper_meta(paper_meta or {})
    safe_coverage = _sanitize_coverage(coverage or {})
    _trace_synthesis_step("synthesize:sanitize_done")
    payload = {
        "paper_meta": safe_paper_meta,
        "coverage": safe_coverage,
        "text_packets": list(text_report.get("evidence_packets", [])),
        "table_packets": list(table_report.get("evidence_packets", [])),
        "figure_packets": list(figure_report.get("evidence_packets", [])),
        "supp_packets": list(supp_report.get("evidence_packets", [])),
        "analysis_notes": _unique_strings(
            _as_str_list(text_report.get("analysis_notes"))
            + _as_str_list(table_report.get("analysis_notes"))
            + _as_str_list(figure_report.get("analysis_notes"))
            + _as_str_list(supp_report.get("analysis_notes")),
            max_items=8,
        ),
        "cross_modal_claims": list(reconcile_report.get("cross_modal_claims", [])),
        "discrepancies": list(reconcile_report.get("discrepancies", [])),
        "text_chunk_records": list(text_chunk_records or []),
    }
    _trace_synthesis_step("synthesize:payload_ready")
    _emit_synthesis_progress(
        progress_callback,
        0.22,
        "Synthesizing executive report: building evidence payload",
    )

    _trace_synthesis_step("synthesize:assemble_begin")
    _emit_synthesis_progress(
        progress_callback,
        0.38,
        "Synthesizing executive report: assembling section dossier",
    )
    draft = _assemble_structured_dossier(payload)
    _trace_synthesis_step("synthesize:assemble_done")
    _emit_synthesis_progress(
        progress_callback,
        0.58,
        "Synthesizing executive report: dossier assembled",
    )
    if settings.analysis_narrative_overrides_enabled:
        _emit_synthesis_progress(
            progress_callback,
            0.67,
            "Synthesizing executive report: narrative override pass",
        )
        llm_overrides = _llm_synthesis_overrides(payload)
        if llm_overrides:
            _apply_narrative_overrides(draft, llm_overrides)

        if settings.analysis_verifier_enabled:
            _emit_synthesis_progress(
                progress_callback,
                0.77,
                "Synthesizing executive report: verifier pass",
            )
            verifier_overrides = _llm_verifier_overrides(payload, draft)
            if verifier_overrides:
                _apply_narrative_overrides(draft, verifier_overrides)
    _emit_synthesis_progress(
        progress_callback,
        0.86,
        "Synthesizing executive report: finalizing summary",
    )

    if settings.analysis_exec_summary_second_pass_enabled:
        _ensure_executive_summary_components(draft, payload)
        _trace_synthesis_step("synthesize:summary_ensured")
    else:
        _trace_synthesis_step("synthesize:summary_second_pass_skipped")
    draft["coverage_snapshot_line"] = _media_counts_line(payload.get("coverage", {}) or {})
    report_payload = draft
    if settings.analysis_schema_validation_enabled:
        _emit_synthesis_progress(
            progress_callback,
            0.94,
            "Synthesizing executive report: validating report schema",
        )
        _trace_synthesis_step("synthesize:validate_begin")
        report_payload = StructuredDossierV2.model_validate(draft).model_dump()
        _trace_synthesis_step("synthesize:validate_done")
    _trace_synthesis_step("synthesize:return")
    _emit_synthesis_progress(
        progress_callback,
        1.0,
        "Synthesizing executive report: complete",
    )
    return _attach_v1_compat_fields(report_payload)


def _emit_synthesis_progress(
    progress_callback: Callable[[float, str], None] | None,
    progress: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    try:
        bounded = max(0.0, min(float(progress), 1.0))
    except Exception:
        bounded = 0.0
    try:
        progress_callback(bounded, str(message or "Synthesizing executive report"))
    except Exception:
        return


def _trace_synthesis_step(step: str) -> None:
    trace_path = str(os.getenv("SYNTHESIS_TRACE_FILE", "")).strip()
    if not trace_path:
        return
    try:
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(f"{step}\n")
    except Exception:
        return


def _deep_json_prompt_worker(prompt: str, system_prompt: str, out_queue: Any) -> None:
    try:
        response = chat_text_deep(prompt, system=system_prompt)
        out_queue.put({"ok": True, "data": extract_json(response)})
    except Exception as exc:
        out_queue.put({"ok": False, "error": str(exc)})


def _run_deep_json_prompt(
    *,
    prompt: str,
    system_prompt: str,
    timeout_seconds: int,
    guard_enabled: bool,
) -> Any:
    if not guard_enabled:
        try:
            response = chat_text_deep(prompt, system=system_prompt)
            return extract_json(response)
        except Exception:
            return {}

    timeout = max(15, int(timeout_seconds or 0))
    context = mp.get_context("spawn")
    out_queue = context.Queue(maxsize=1)
    proc = context.Process(target=_deep_json_prompt_worker, args=(prompt, system_prompt, out_queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {}
    if int(proc.exitcode or 0) != 0:
        return {}
    try:
        payload = out_queue.get_nowait()
    except QueueEmpty:
        return {}
    except Exception:
        return {}
    if not isinstance(payload, dict) or not payload.get("ok"):
        return {}
    return payload.get("data", {})


def _sanitize_text_atom(value: Any, *, max_len: int = MAX_META_STR_LEN) -> str:
    text = str(value or "")
    text = text.replace("\u0000", " ").replace("\u0001", " ").replace("\u0002", " ").replace("\u0003", " ")
    text = re.sub(r"[\x04-\x08\x0b-\x1f\x7f]", " ", text)
    text = " ".join(text.split()).strip()
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _sanitize_paper_meta(meta: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in meta.items():
        clean_key = _sanitize_text_atom(key, max_len=64)
        if not clean_key:
            continue
        if isinstance(value, list):
            clean_list: list[Any] = []
            for item in value[:64]:
                if isinstance(item, (str, int, float, bool)):
                    clean_list.append(_sanitize_text_atom(item))
                elif isinstance(item, dict):
                    clean_list.append(
                        {
                            _sanitize_text_atom(k, max_len=64): _sanitize_text_atom(v)
                            for k, v in list(item.items())[:16]
                        }
                    )
            out[clean_key] = clean_list
            continue
        if isinstance(value, dict):
            out[clean_key] = {
                _sanitize_text_atom(k, max_len=64): _sanitize_text_atom(v)
                for k, v in list(value.items())[:32]
            }
            continue
        if isinstance(value, (str, int, float, bool)):
            out[clean_key] = _sanitize_text_atom(value)
    authors_raw = out.get("authors")
    if isinstance(authors_raw, list):
        authors_display, authors_extracted_count = sanitize_author_list(authors_raw, max_items=24)
        out["authors"] = authors_display
        out["authors_extracted_count"] = authors_extracted_count
        out["authors_display_count"] = len(authors_display)
    return out


def _sanitize_coverage_block(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}

    def _as_int(key: str) -> int:
        try:
            return max(0, int(block.get(key, 0) or 0))
        except Exception:
            return 0

    def _clean_ref_list(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            ref = _sanitize_text_atom(raw, max_len=MAX_COVERAGE_REF_LEN)
            if not ref:
                continue
            if ref in seen:
                continue
            seen.add(ref)
            out.append(ref)
            if len(out) >= MAX_COVERAGE_REFS:
                break
        return out

    return {
        "expected": _as_int("expected"),
        "extracted": _as_int("extracted"),
        "expected_refs": _clean_ref_list(block.get("expected_refs", [])),
        "extracted_refs": _clean_ref_list(block.get("extracted_refs", [])),
        "missing_refs": _clean_ref_list(block.get("missing_refs", [])),
    }


def _sanitize_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        return {}
    return {
        "figures": _sanitize_coverage_block(coverage.get("figures", {})),
        "tables": _sanitize_coverage_block(coverage.get("tables", {})),
        "supp_figures": _sanitize_coverage_block(coverage.get("supp_figures", {})),
        "supp_tables": _sanitize_coverage_block(coverage.get("supp_tables", {})),
    }


def _llm_synthesis_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    prompt_payload = {
        "paper_meta": payload.get("paper_meta", {}),
        "coverage": payload.get("coverage", {}),
        "text_findings": summarize_packet_statements(payload.get("text_packets", []), max_items=8),
        "table_findings": summarize_packet_statements(payload.get("table_packets", []), max_items=6),
        "figure_findings": summarize_packet_statements(payload.get("figure_packets", []), max_items=6),
        "supp_findings": summarize_packet_statements(payload.get("supp_packets", []), max_items=6),
        "discrepancies": payload.get("discrepancies", [])[:8],
        "cross_modal_claims": payload.get("cross_modal_claims", [])[:8],
    }
    prompt = truncate_text(
        "Create a concise, evidence-grounded dossier narrative summary.\n\n" + json.dumps(prompt_payload, indent=2),
        max_chars_for_ctx(settings.llm_n_ctx),
    )
    parsed = _run_deep_json_prompt(
        prompt=prompt,
        system_prompt=SYNTHESIS_SYSTEM,
        timeout_seconds=int(settings.analysis_narrative_overrides_subprocess_timeout_sec or 0),
        guard_enabled=bool(settings.analysis_narrative_overrides_subprocess_guard_enabled),
    )
    return _coerce_narrative_overrides(parsed)


def _llm_verifier_overrides(payload: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    prompt_payload = {
        "paper_meta": payload.get("paper_meta", {}),
        "coverage": payload.get("coverage", {}),
        "evidence_digest": {
            "text_findings": summarize_packet_statements(payload.get("text_packets", []), max_items=10),
            "table_findings": summarize_packet_statements(payload.get("table_packets", []), max_items=8),
            "figure_findings": summarize_packet_statements(payload.get("figure_packets", []), max_items=8),
            "supp_findings": summarize_packet_statements(payload.get("supp_packets", []), max_items=8),
        },
        "cross_modal_claims": payload.get("cross_modal_claims", [])[:10],
        "discrepancies": payload.get("discrepancies", [])[:10],
        "draft_report": {
            "executive_summary": draft.get("executive_summary", ""),
            "methods_strengths": draft.get("methods_strengths", []),
            "methods_weaknesses": draft.get("methods_weaknesses", []),
            "reproducibility_ethics": draft.get("reproducibility_ethics", []),
            "uncertainty_gaps": draft.get("uncertainty_gaps", []),
        },
    }
    prompt = truncate_text(
        "Verifier pass: fix only what is inaccurate or unsupported in this draft report.\n\n"
        + json.dumps(prompt_payload, indent=2),
        max_chars_for_ctx(settings.llm_n_ctx),
    )
    parsed = _run_deep_json_prompt(
        prompt=prompt,
        system_prompt=VERIFIER_SYSTEM,
        timeout_seconds=int(settings.analysis_narrative_overrides_subprocess_timeout_sec or 0),
        guard_enabled=bool(settings.analysis_narrative_overrides_subprocess_guard_enabled),
    )
    return _coerce_narrative_overrides(parsed)


def _llm_section_extraction_worker(payload: dict[str, Any], out_queue: Any) -> None:
    try:
        out_queue.put({"ok": True, "data": _llm_section_extraction_direct(payload)})
    except Exception as exc:
        out_queue.put({"ok": False, "error": str(exc)})


def _section_extraction_safe_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    def _clean_packets(raw_packets: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not isinstance(raw_packets, list):
            return out
        for packet in raw_packets:
            if not isinstance(packet, dict):
                continue
            refs = [str(ref).strip() for ref in packet.get("evidence_refs", []) if str(ref).strip()][:8]
            out.append(
                {
                    "finding_id": str(packet.get("finding_id", "") or ""),
                    "anchor": str(packet.get("anchor", "") or ""),
                    "statement": str(packet.get("statement", "") or ""),
                    "evidence_refs": refs,
                    "confidence": clamp_confidence(packet.get("confidence", 0.0)),
                    "category": str(packet.get("category", "") or ""),
                    "section_label": str(packet.get("section_label", "") or ""),
                }
            )
        return out

    return {
        "text_packets": _clean_packets(payload.get("text_packets", [])),
        "table_packets": _clean_packets(payload.get("table_packets", [])),
        "figure_packets": _clean_packets(payload.get("figure_packets", [])),
        "supp_packets": _clean_packets(payload.get("supp_packets", [])),
    }


def _section_extract_has_content(extracted: dict[str, list[dict[str, Any]]]) -> bool:
    if not isinstance(extracted, dict):
        return False
    for key in ("introduction", "methods", "results", "discussion", "conclusion"):
        rows = extracted.get(key, [])
        if isinstance(rows, list) and rows:
            return True
    return False


def _section_rows_for_extraction(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    text_packets = list(payload.get("text_packets", []))
    table_packets = list(payload.get("table_packets", []))
    figure_packets = list(payload.get("figure_packets", []))
    supp_packets = list(payload.get("supp_packets", []))
    rows_by_section: dict[str, list[dict[str, Any]]] = {
        "introduction": [],
        "methods": [],
        "results": [],
        "discussion": [],
        "conclusion": [],
    }
    for packet in text_packets:
        section = _packet_section_label(packet)
        if section not in rows_by_section:
            continue
        statement = _strip_confidence_annotations(str(packet.get("statement", "")).strip())
        if not statement:
            continue
        refs = _packet_refs(packet, max_items=6)
        if not refs:
            continue
        rows_by_section[section].append(
            {
                "statement": statement,
                "evidence_refs": refs,
                "anchor": str(packet.get("anchor", "")).strip(),
                "category": str(packet.get("category", "")).strip(),
                "confidence": clamp_confidence(packet.get("confidence", 0.0)),
            }
        )

    # Allow results extraction to include figure/table/supp evidence with concrete outcomes.
    for packet in table_packets + figure_packets + supp_packets:
        statement = _strip_confidence_annotations(str(packet.get("statement", "")).strip())
        if not statement:
            continue
        if not _has_concrete_result_outcome(statement) or _is_method_like_statement(statement):
            continue
        refs = _packet_refs(packet, max_items=6)
        if not refs:
            continue
        rows_by_section["results"].append(
            {
                "statement": statement,
                "evidence_refs": refs,
                "anchor": str(packet.get("anchor", "")).strip(),
                "category": str(packet.get("category", "")).strip(),
                "confidence": clamp_confidence(packet.get("confidence", 0.0)),
            }
        )
    for section, rows in rows_by_section.items():
        rows.sort(key=lambda row: (-float(row.get("confidence", 0.0)), len(str(row.get("statement", "")))))
        rows_by_section[section] = rows[: max(30, SECTION_EXTRACTION_LIMITS.get(section, 8) * 4)]
    return rows_by_section


def _fallback_section_row_score(section: str, row: dict[str, Any]) -> float:
    statement = str(row.get("statement", "")).strip()
    if not statement:
        return 0.0
    score = 1.1 * clamp_confidence(row.get("confidence", 0.0))
    score += _section_statement_alignment_score(section, statement)
    if section == "results" and _has_concrete_result_outcome(statement):
        score += 0.22
    if _is_fragment_like_statement(statement):
        score -= 0.30
    return float(score)


def _fallback_section_extraction_from_rows(
    rows_by_section: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    global_cap = max(1, int(settings.analysis_section_extraction_max_points_per_section or 8))
    for section in ("introduction", "methods", "results", "discussion", "conclusion"):
        rows = rows_by_section.get(section, []) if isinstance(rows_by_section, dict) else []
        if not isinstance(rows, list):
            out[section] = []
            continue
        limit = min(
            SECTION_EXTRACTION_LIMITS.get(section, 8),
            global_cap if section != "results" else max(global_cap, 12),
        )
        ranked = sorted(
            [row for row in rows if isinstance(row, dict)],
            key=lambda row: (-_fallback_section_row_score(section, row), len(str(row.get("statement", "")))),
        )
        selected: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for row in ranked:
            statement = _compact_section_statement(_strip_confidence_annotations(str(row.get("statement", "")).strip()), max_chars=220)
            refs = [str(ref).strip() for ref in row.get("evidence_refs", []) if str(ref).strip()][:6]
            if not statement or not refs:
                continue
            key = _canonical_statement_text(statement)
            if not key or _is_redundant_statement_key(key, seen_keys):
                continue
            seen_keys.add(key)
            selected.append(
                {
                    "statement": statement,
                    "evidence_refs": refs,
                    "kind": "fallback",
                }
            )
            if len(selected) >= limit:
                break
        out[section] = selected
    return out


def _llm_section_extraction(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    safe_payload = _section_extraction_safe_payload(payload)
    if not safe_payload.get("text_packets"):
        return {}
    deterministic_fallback = _fallback_section_extraction_from_rows(_section_rows_for_extraction(safe_payload))
    if not settings.analysis_section_extraction_subprocess_guard_enabled:
        return _llm_section_extraction_direct(safe_payload)

    timeout_seconds = max(15, int(settings.analysis_section_extraction_subprocess_timeout_sec or 0))
    context = mp.get_context("spawn")
    out_queue = context.Queue(maxsize=1)
    proc = context.Process(target=_llm_section_extraction_worker, args=(safe_payload, out_queue))
    proc.start()
    proc.join(timeout_seconds)

    timed_out = proc.is_alive()
    if timed_out:
        proc.terminate()
        proc.join(5)
        return deterministic_fallback

    if int(proc.exitcode or 0) != 0:
        return deterministic_fallback

    try:
        result = out_queue.get_nowait()
    except QueueEmpty:
        return deterministic_fallback
    except Exception:
        return deterministic_fallback
    if not isinstance(result, dict) or not result.get("ok"):
        return deterministic_fallback
    data = result.get("data")
    if isinstance(data, dict) and _section_extract_has_content(data):
        return data
    return deterministic_fallback


def _llm_section_extraction_direct(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not list(payload.get("text_packets", [])):
        return {}

    rows_by_section = _section_rows_for_extraction(payload)
    fallback_extraction = _fallback_section_extraction_from_rows(rows_by_section)
    prompt_payload = {"section_rows": rows_by_section}
    prompt = truncate_text(
        "Extract high-fidelity section bullets from these evidence rows.\n\n" + json.dumps(prompt_payload, indent=2),
        max_chars_for_ctx(settings.llm_n_ctx),
    )
    try:
        response = chat_text_deep(prompt, system=SECTION_EXTRACTION_SYSTEM)
    except Exception:
        return fallback_extraction
    normalized = _normalize_llm_section_extraction(extract_json(response), payload)
    if _section_extract_has_content(normalized):
        return normalized
    return fallback_extraction


def _normalize_llm_section_extraction(
    raw: Any,
    payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        return {}

    text_packets = list(payload.get("text_packets", []))
    table_packets = list(payload.get("table_packets", []))
    figure_packets = list(payload.get("figure_packets", []))
    supp_packets = list(payload.get("supp_packets", []))

    text_ref_section: dict[str, str] = {}
    all_valid_refs: set[str] = set()
    for packet in text_packets:
        section = _packet_section_label(packet)
        for ref in _packet_refs(packet, max_items=8):
            all_valid_refs.add(ref)
            if ref not in text_ref_section and section in SECTION_LABELS:
                text_ref_section[ref] = section
    for packet in table_packets + figure_packets + supp_packets:
        for ref in _packet_refs(packet, max_items=8):
            all_valid_refs.add(ref)

    global_cap = max(1, int(settings.analysis_section_extraction_max_points_per_section or 8))
    out: dict[str, list[dict[str, Any]]] = {}
    for section in ("introduction", "methods", "results", "discussion", "conclusion"):
        section_items = raw.get(section, [])
        if not isinstance(section_items, list):
            out[section] = []
            continue
        limit = min(SECTION_EXTRACTION_LIMITS.get(section, 8), global_cap if section != "results" else max(global_cap, 12))
        normalized: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for item in section_items:
            if isinstance(item, dict):
                statement = _strip_confidence_annotations(str(item.get("statement", "")).strip())
                raw_refs = item.get("evidence_refs", [])
                kind = str(item.get("kind", "other") or "other").strip().lower()
            else:
                statement = _strip_confidence_annotations(str(item or "").strip())
                raw_refs = []
                kind = "other"
            if not statement:
                continue
            refs = [str(ref).strip() for ref in (raw_refs if isinstance(raw_refs, list) else []) if str(ref).strip()]
            refs = [ref for ref in refs if ref in all_valid_refs][:6]
            if not refs:
                continue
            if section == "results":
                if not any(_is_results_ref(ref, text_ref_section) for ref in refs):
                    continue
            else:
                if not any(text_ref_section.get(ref) == section for ref in refs):
                    continue
            statement = _compact_section_statement(statement, max_chars=220)
            key = _canonical_text(statement)
            if not key or _is_redundant_statement_key(key, seen_keys):
                continue
            seen_keys.add(key)
            normalized.append(
                {
                    "statement": statement,
                    "evidence_refs": refs,
                    "kind": kind or "other",
                }
            )
            if len(normalized) >= limit:
                break
        out[section] = normalized
    return out


def _assemble_structured_dossier(payload: dict[str, Any]) -> dict[str, Any]:
    paper_meta = payload.get("paper_meta", {}) or {}
    coverage = payload.get("coverage", {}) or {}
    text_packets = list(payload.get("text_packets", []))
    table_packets = list(payload.get("table_packets", []))
    figure_packets = list(payload.get("figure_packets", []))
    supp_packets = list(payload.get("supp_packets", []))
    cross_modal_claims = list(payload.get("cross_modal_claims", []))
    discrepancies = list(payload.get("discrepancies", []))
    analysis_notes = _as_str_list(payload.get("analysis_notes"))
    text_chunk_records = list(payload.get("text_chunk_records", []))

    methods_compact = _methods_compact(text_packets, analysis_notes=analysis_notes)
    sections_extracted = (
        _llm_section_extraction(payload) if settings.analysis_section_extraction_enabled else {}
    )
    sections_compact = _sections_compact(
        text_packets=text_packets,
        methods_compact=methods_compact,
        analysis_notes=analysis_notes,
        text_chunk_records=text_chunk_records,
        result_support_packets=table_packets + figure_packets + supp_packets,
        sections_extracted=sections_extracted,
    )
    sections_compact, sections_compact_dedupe = _dedupe_sections_compact_rows(
        sections_compact,
        access_limited=any(ACCESS_LIMITED_NOTE_RE.search(str(note or "")) for note in analysis_notes),
    )

    executive_summary = _build_executive_summary(
        sections_compact=sections_compact,
        existing_summary="",
        sections_extracted=sections_extracted,
    )
    if settings.analysis_summary_polish_enabled:
        executive_summary = _constrained_summary_polish(executive_summary, sections_compact)

    detailed_sections, section_diagnostics, sections_fallback_notes = _build_detailed_sections(
        text_packets=text_packets,
        table_packets=table_packets,
        figure_packets=figure_packets,
        supp_packets=supp_packets,
        methods_compact=methods_compact,
        analysis_notes=analysis_notes,
        text_chunk_records=text_chunk_records,
        sections_extracted=sections_extracted,
    )
    extractive_evidence = _build_extractive_evidence(
        sections_payload=detailed_sections,
        text_chunk_records=text_chunk_records,
    )
    presentation_evidence = _build_presentation_evidence(extractive_evidence=extractive_evidence)
    executive_report = _build_executive_report(
        extractive_evidence=extractive_evidence,
        fallback_summary=executive_summary,
    )
    grounded_overview = str(executive_report.get("overview", "")).strip()
    if grounded_overview:
        executive_summary = grounded_overview
    section_diagnostics.update(
        {
            "section_packet_counts": _section_packet_counts(text_packets),
            "section_conflict_count": _section_conflict_count(text_packets),
            "unknown_section_count": _unknown_section_count(text_packets),
            "slot_fill_rates": _slot_fill_rates(sections_compact),
            "sections_compact_cross_section_dedupe": sections_compact_dedupe,
            "cross_section_rejections": _cross_section_rejections(text_packets),
            "section_extraction_enabled": bool(settings.analysis_section_extraction_enabled),
            "section_extraction_counts": {
                key: len(value) for key, value in sections_extracted.items() if isinstance(value, list)
            },
            "extractive_evidence_counts": {
                key: len(value) for key, value in extractive_evidence.items() if isinstance(value, list)
            },
            "presentation_evidence_counts": {
                key: len(value) for key, value in presentation_evidence.items() if isinstance(value, list)
            },
            "executive_report_style": str(executive_report.get("style", "")),
        }
    )
    strengths = _methods_strengths_from_compact(methods_compact, max_items=5)
    weaknesses = _methods_weaknesses_from_compact(methods_compact, max_items=5)
    if not weaknesses:
        weaknesses = ["No explicit methodological weaknesses were extracted; manual appraisal is recommended."]

    reproducibility_ethics: list[str] = []
    for packet in text_packets:
        tokens = _category_tokens(str(packet.get("category", "")))
        if set(tokens) & {"ethics", "reproducibility"}:
            statement = str(packet.get("statement", "")).strip()
            if statement:
                reproducibility_ethics.append(statement)
    reproducibility_ethics = _unique_strings(reproducibility_ethics, max_items=6)

    uncertainty_gaps = _coverage_gaps(coverage)
    uncertainty_gaps += analysis_notes
    uncertainty_gaps += [
        _discrepancy_uncertainty_line(item)
        for item in discrepancies[:5]
        if isinstance(item, dict)
    ]
    uncertainty_gaps = _unique_strings([item for item in uncertainty_gaps if item], max_items=8)
    methodology_details = _methodology_details_from_compact(methods_compact, max_items=20)
    if not methodology_details:
        methodology_details = _methodology_details(text_packets)

    all_packets = text_packets + table_packets + figure_packets + supp_packets
    confidence_values = [clamp_confidence(packet.get("confidence", 0.0)) for packet in all_packets]
    avg_conf = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    penalty = min(0.5, 0.08 * len(discrepancies))
    overall_confidence = max(0.0, min(1.0, avg_conf - penalty))

    sectioned_report_version = 3 if settings.sectioned_report_v3_enabled else 2
    report = {
        "schema_version": 2,
        "sectioned_report_version": sectioned_report_version,
        "paper_meta": paper_meta,
        "coverage": coverage,
        "modalities": {
            "text": {
                "findings": text_packets,
                "highlights": _unique_strings(summarize_packet_statements(text_packets, max_items=12), max_items=8),
                "coverage_gaps": [],
            },
            "table": {
                "findings": table_packets,
                "highlights": _unique_strings(summarize_packet_statements(table_packets, max_items=12), max_items=8),
                "coverage_gaps": _coverage_block_gaps(coverage.get("tables", {}), "table"),
            },
            "figure": {
                "findings": figure_packets,
                "highlights": _unique_strings(summarize_packet_statements(figure_packets, max_items=12), max_items=8),
                "coverage_gaps": _coverage_block_gaps(coverage.get("figures", {}), "figure"),
            },
            "supplement": {
                "findings": supp_packets,
                "highlights": _unique_strings(summarize_packet_statements(supp_packets, max_items=12), max_items=8),
                "coverage_gaps": (
                    _coverage_block_gaps(coverage.get("supp_tables", {}), "supplement table")
                    + _coverage_block_gaps(coverage.get("supp_figures", {}), "supplement figure")
                ),
            },
        },
        "cross_modal_claims": cross_modal_claims,
        "discrepancies": discrepancies,
        "executive_summary": executive_summary,
        "extractive_evidence_version": 1,
        "extractive_evidence": extractive_evidence,
        "presentation_evidence_version": 1,
        "presentation_evidence": presentation_evidence,
        "executive_report_version": 1,
        "executive_report": executive_report,
        "methods_strengths": strengths,
        "methods_weaknesses": weaknesses,
        "methods_compact_version": METHODS_COMPACT_VERSION,
        "methods_compact": methods_compact,
        "sections_compact_version": SECTIONS_COMPACT_VERSION,
        "sections_compact": sections_compact,
        "sections_extracted_version": 1 if sections_extracted else 0,
        "sections_extracted": sections_extracted,
        "methodology_details": methodology_details,
        "sections": detailed_sections,
        "section_diagnostics": section_diagnostics,
        "sections_fallback_used": bool(sections_fallback_notes),
        "sections_fallback_notes": sections_fallback_notes,
        "coverage_snapshot_line": _media_counts_line(coverage),
        "reproducibility_ethics": reproducibility_ethics,
        "uncertainty_gaps": uncertainty_gaps,
        "overall_confidence": overall_confidence,
    }
    return report


def _attach_v1_compat_fields(v2_report: dict[str, Any]) -> dict[str, Any]:
    modalities = v2_report.get("modalities", {})
    figure_packets = modalities.get("figure", {}).get("findings", [])
    table_packets = modalities.get("table", {}).get("findings", [])
    supp_packets = modalities.get("supplement", {}).get("findings", [])
    text_packets = modalities.get("text", {}).get("findings", [])
    discrepancies = v2_report.get("discrepancies", [])
    coverage = v2_report.get("coverage", {})

    key_findings = summarize_packet_statements(text_packets, 5)
    key_findings += summarize_packet_statements(table_packets, 3)
    key_findings += summarize_packet_statements(figure_packets, 3)
    key_findings += summarize_packet_statements(supp_packets, 3)
    key_findings = _unique_strings(key_findings, max_items=10)

    v2_report["summary_schema_version"] = int(v2_report.get("schema_version", 2) or 2)
    v2_report["sectioned_report_version"] = int(v2_report.get("sectioned_report_version", 2) or 2)
    v2_report["figure_coverage"] = coverage.get("figures", {})
    v2_report["table_coverage"] = coverage.get("tables", {})
    v2_report["supp_figure_coverage"] = coverage.get("supp_figures", {})
    v2_report["supp_table_coverage"] = coverage.get("supp_tables", {})
    v2_report["figure_results"] = _packets_to_results(figure_packets)
    v2_report["table_results"] = _packets_to_results(table_packets)
    v2_report["supp_results"] = _packets_to_results(supp_packets)
    v2_report["definitions"] = []
    v2_report["key_findings"] = key_findings
    v2_report["strengths"] = v2_report.get("methods_strengths", [])
    v2_report["weaknesses"] = v2_report.get("methods_weaknesses", [])
    v2_report["scores"] = {
        "methods": 0,
        "statistics": 0,
        "clinical_relevance": 0,
        "ethics": 0,
        "reproducibility": 0,
        "clarity": 0,
    }
    v2_report["key_discrepancies"] = [
        {
            "claim": item.get("claim", ""),
            "severity": item.get("severity", "medium"),
            "evidence": item.get("evidence", []),
        }
        for item in discrepancies[:8]
        if isinstance(item, dict)
    ]
    return v2_report


def _packets_to_results(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for packet in packets:
        statement = str(packet.get("statement", "")).strip()
        if not statement:
            continue
        out.append(
            {
                "result": statement,
                "evidence": packet.get("evidence_refs", []),
                "confidence": clamp_confidence(packet.get("confidence", 0.0)),
            }
        )
    return out


def _methods_compact(text_packets: list[dict[str, Any]], *, analysis_notes: list[str]) -> list[dict[str, Any]]:
    candidates_by_slot: dict[str, list[dict[str, Any]]] = {slot["slot_key"]: [] for slot in METHODS_COMPACT_SLOTS}
    access_limited = any(ACCESS_LIMITED_NOTE_RE.search(str(note or "")) for note in analysis_notes)
    for idx, packet in enumerate(text_packets):
        candidate = _method_candidate(packet, idx=idx)
        if not candidate:
            continue
        for slot in METHODS_COMPACT_SLOTS:
            score = _method_slot_score(slot, candidate)
            if score <= 0:
                continue
            candidates_by_slot[slot["slot_key"]].append(
                {
                    "score": score,
                    "anchor_specificity": _anchor_specificity_score(candidate["anchor"]),
                    "confidence": candidate["confidence"],
                    "idx": idx,
                    "candidate": candidate,
                }
            )

    for slot_key, rows in candidates_by_slot.items():
        rows.sort(
            key=lambda item: (
                -int(item["score"]),
                -int(item["anchor_specificity"]),
                -float(item["confidence"]),
                int(item["idx"]),
            )
        )
        candidates_by_slot[slot_key] = rows

    used_packet_keys: set[str] = set()
    used_anchor_keys: set[str] = set()
    used_statement_keys: set[str] = set()
    used_statement_stems: set[str] = set()
    methods_compact: list[dict[str, Any]] = []
    for slot in METHODS_COMPACT_SLOTS:
        slot_key = str(slot["slot_key"])
        label = str(slot["label"])

        def _pick_candidate(*, allow_anchor_reuse: bool) -> dict[str, Any] | None:
            for row in candidates_by_slot.get(slot_key, []):
                candidate = row["candidate"]
                packet_key = str(candidate["packet_key"])
                if packet_key in used_packet_keys:
                    continue
                anchor_key = str(candidate.get("anchor_key", "")).strip()
                if not allow_anchor_reuse and anchor_key and anchor_key in used_anchor_keys:
                    continue
                statement_key = _canonical_statement_text(str(candidate.get("statement", "")))
                if statement_key and _is_redundant_statement_key(statement_key, used_statement_keys):
                    continue
                stem_key = _statement_stem_key(str(candidate.get("statement", "")))
                if stem_key and stem_key in used_statement_stems:
                    continue
                return candidate
            return None

        chosen = _pick_candidate(allow_anchor_reuse=False)
        if chosen is None:
            chosen = _pick_candidate(allow_anchor_reuse=True)

        if chosen is not None:
            chosen_statement = _compact_method_statement(str(chosen["statement"]))
            chosen_statement_key = _canonical_statement_text(chosen_statement)
            chosen_stem_key = _statement_stem_key(chosen_statement)
            chosen_anchor_key = str(chosen.get("anchor_key", "")).strip()
            used_packet_keys.add(str(chosen.get("packet_key", "")))
            if chosen_anchor_key:
                used_anchor_keys.add(chosen_anchor_key)
            if chosen_statement_key:
                used_statement_keys.add(chosen_statement_key)
            if chosen_stem_key:
                used_statement_stems.add(chosen_stem_key)
            methods_compact.append(
                {
                    "slot_key": slot_key,
                    "label": label,
                    "statement": chosen_statement,
                    "status": "found",
                    "evidence_refs": list(chosen["evidence_refs"])[:5],
                    "confidence": clamp_confidence(chosen["confidence"]),
                }
            )
            continue

        methods_compact.append(
            {
                "slot_key": slot_key,
                "label": label,
                "statement": METHOD_ACCESS_LIMITED_STATEMENT if access_limited else METHOD_NOT_FOUND_STATEMENT,
                "status": "access_limited" if access_limited else "not_found",
                "evidence_refs": [],
                "confidence": 0.0,
            }
        )
    return methods_compact


def _sections_compact(
    *,
    text_packets: list[dict[str, Any]],
    methods_compact: list[dict[str, Any]],
    analysis_notes: list[str],
    text_chunk_records: list[dict[str, Any]] | None = None,
    result_support_packets: list[dict[str, Any]] | None = None,
    sections_extracted: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    access_limited = any(ACCESS_LIMITED_NOTE_RE.search(str(note or "")) for note in analysis_notes)
    out: dict[str, list[dict[str, Any]]] = {
        "introduction": [],
        "methods": [],
        "results": [],
        "discussion": [],
        "conclusion": [],
    }
    out["methods"] = [
        {
            "section_key": "methods",
            "slot_key": str(slot.get("slot_key", "")),
            "label": str(slot.get("label", "")),
            "statement": str(slot.get("statement", "")),
            "status": str(slot.get("status", "not_found")),
            "evidence_refs": [str(ref).strip() for ref in slot.get("evidence_refs", []) if str(ref).strip()][:5],
            "confidence": clamp_confidence(slot.get("confidence", 0.0)),
        }
        for slot in methods_compact
    ]

    by_section: dict[str, list[dict[str, Any]]] = {
        section_key: [packet for packet in text_packets if _packet_section_label(packet) == section_key]
        for section_key in SECTION_COMPACT_SLOTS
    }
    extracted = sections_extracted or {}
    for section_key, slot_specs in SECTION_COMPACT_SLOTS.items():
        candidates = _section_compact_candidates(section_key, by_section.get(section_key, []))
        if section_key in {"discussion", "conclusion"}:
            candidates.extend(_section_compact_candidates_from_extracted(section_key, extracted.get(section_key, [])))
        if section_key == "results":
            candidates = candidates + _result_support_compact_candidates(result_support_packets or [])
        used_packet_keys: set[str] = set()
        rows: list[dict[str, Any]] = []
        for slot_spec in slot_specs:
            slot_key = str(slot_spec.get("slot_key", ""))
            slot_label = str(slot_spec.get("label", slot_key))
            best: dict[str, Any] | None = None
            best_score = -1
            for candidate in candidates:
                packet_key = str(candidate.get("packet_key", ""))
                if not packet_key or packet_key in used_packet_keys:
                    continue
                score = _section_slot_score(section_key, slot_spec, candidate)
                if score <= best_score:
                    continue
                best_score = score
                best = candidate
            if best is not None:
                used_packet_keys.add(str(best.get("packet_key", "")))
                rows.append(
                    {
                        "section_key": section_key,
                        "slot_key": slot_key,
                        "label": slot_label,
                        "statement": _compact_section_statement(str(best.get("statement", ""))),
                        "status": "found",
                        "evidence_refs": [str(ref) for ref in best.get("evidence_refs", [])][:5],
                        "confidence": clamp_confidence(best.get("confidence", 0.0)),
                    }
                )
                continue
            rows.append(
                {
                    "section_key": section_key,
                    "slot_key": slot_key,
                    "label": slot_label,
                    "statement": SECTION_ACCESS_LIMITED_STATEMENT if access_limited else SECTION_NOT_FOUND_STATEMENT,
                    "status": "access_limited" if access_limited else "not_found",
                    "evidence_refs": [],
                    "confidence": 0.0,
                }
            )
        out[section_key] = rows
    for section_key in ("introduction", "results", "discussion", "conclusion"):
        out[section_key] = _augment_section_compact_from_raw_chunks(
            section_key=section_key,
            rows=out.get(section_key, []),
            slot_specs=SECTION_COMPACT_SLOTS.get(section_key, []),
            text_chunk_records=text_chunk_records or [],
            access_limited=access_limited,
        )
    return out


def _compact_not_found_row_template(section_key: str, *, access_limited: bool) -> tuple[str, str]:
    key = str(section_key or "").strip().lower()
    if key == "methods":
        return (
            "access_limited" if access_limited else "not_found",
            METHOD_ACCESS_LIMITED_STATEMENT if access_limited else METHOD_NOT_FOUND_STATEMENT,
        )
    return (
        "access_limited" if access_limited else "not_found",
        SECTION_ACCESS_LIMITED_STATEMENT if access_limited else SECTION_NOT_FOUND_STATEMENT,
    )


def _score_compact_row_for_section(section_key: str, row: dict[str, Any]) -> float:
    statement = str(row.get("statement", "")).strip()
    score = 1.2 * clamp_confidence(row.get("confidence", 0.0))
    score += _section_statement_alignment_score(section_key, statement)
    lowered = statement.lower()
    if section_key == "methods":
        if RESULTS_SECTION_RE.search(statement):
            score -= 0.45
        if re.search(r"\b(our results|these results|results?\s+(?:show|suggest|indicat|corroborate|emphasize))\b", lowered):
            score -= 0.45
    elif section_key == "introduction":
        if RESULTS_SECTION_RE.search(statement) and not INTRO_SECTION_RE.search(statement):
            score -= 0.30
    elif section_key in {"discussion", "conclusion"}:
        if METHODS_SECTION_RE.search(statement) and not RESULTS_SECTION_RE.search(statement):
            score -= 0.18
    refs = [str(ref).strip() for ref in row.get("evidence_refs", []) if str(ref).strip()]
    anchor = refs[0] if refs else ""
    anchor_section = _anchor_section_label(anchor)
    if anchor_section == section_key:
        score += 0.30
    elif anchor_section != "unknown" and anchor_section != section_key:
        score -= 0.25
    return float(score)


def _dedupe_sections_compact_rows(
    sections_compact: dict[str, list[dict[str, Any]]],
    *,
    access_limited: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not isinstance(sections_compact, dict):
        return sections_compact, {"removed_count": 0, "removed_by_section": {}}

    section_order = [key for key in EXEC_REPORT_SECTION_ORDER if isinstance(sections_compact.get(key), list)]
    entries: list[dict[str, Any]] = []
    for section in section_order:
        rows = sections_compact.get(section, [])
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if str(row.get("status", "")).strip().lower() != "found":
                continue
            statement = str(row.get("statement", "")).strip()
            if not statement:
                continue
            key = _canonical_statement_text(statement)
            if not key:
                continue
            entries.append(
                {
                    "section": section,
                    "idx": idx,
                    "key": key,
                    "score": _score_compact_row_for_section(section, row),
                }
            )

    if len(entries) <= 1:
        return sections_compact, {"removed_count": 0, "removed_by_section": {}}

    # Keep compact dedupe conservative to avoid suppressing distinct section points.
    # Cross-section removal here only applies to exact canonicalized statement matches.
    clusters: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        cluster_key = str(entry["key"])
        clusters.setdefault(cluster_key, []).append(entry)

    keep: set[tuple[str, int]] = set()
    for cluster_entries in clusters.values():
        winner = max(
            cluster_entries,
            key=lambda e: (
                float(e.get("score", 0.0)),
                -section_order.index(e["section"]) if e.get("section") in section_order else -999,
                -int(e.get("idx", 0)),
            ),
        )
        keep.add((str(winner["section"]), int(winner["idx"])))

    removed_by_section: dict[str, int] = {}
    protected_sections = {"introduction", "results", "discussion", "conclusion"}
    for section in section_order:
        rows = sections_compact.get(section, [])
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if str(row.get("status", "")).strip().lower() != "found":
                continue
            if (section, idx) in keep:
                continue
            if section in protected_sections:
                continue
            new_status, new_statement = _compact_not_found_row_template(section, access_limited=access_limited)
            row["status"] = new_status
            row["statement"] = new_statement
            row["evidence_refs"] = []
            row["confidence"] = 0.0
            removed_by_section[section] = int(removed_by_section.get(section, 0) or 0) + 1

    return sections_compact, {
        "removed_count": int(sum(removed_by_section.values())),
        "removed_by_section": removed_by_section,
    }


def _augment_section_compact_from_raw_chunks(
    *,
    section_key: str,
    rows: list[dict[str, Any]],
    slot_specs: list[dict[str, Any]],
    text_chunk_records: list[dict[str, Any]],
    access_limited: bool,
) -> list[dict[str, Any]]:
    if not rows or not slot_specs or not text_chunk_records:
        return rows

    min_found = int(SECTION_COMPACT_MIN_FOUND_FOR_RESCUE.get(section_key, 0) or 0)
    found_count = sum(1 for row in rows if str(row.get("status", "")).strip().lower() == "found")
    if found_count >= min_found:
        return rows

    fallback_rows = _section_compact_rows_from_raw_chunks(
        section_key=section_key,
        slot_specs=slot_specs,
        text_chunk_records=text_chunk_records,
        access_limited=access_limited,
    )
    if not fallback_rows:
        return rows

    fallback_iter = iter([row for row in fallback_rows if str(row.get("status", "")).strip().lower() == "found"])
    used_statement_keys = {
        _canonical_text(str(row.get("statement", "")))
        for row in rows
        if str(row.get("status", "")).strip().lower() == "found"
    }
    replaced: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        if status == "found":
            replaced.append(row)
            continue
        next_row: dict[str, Any] | None = None
        while True:
            try:
                candidate = next(fallback_iter)
            except StopIteration:
                candidate = None
            if candidate is None:
                break
            key = _canonical_text(str(candidate.get("statement", "")))
            if key and key in used_statement_keys:
                continue
            if key:
                used_statement_keys.add(key)
            next_row = candidate
            break
        replaced.append(next_row if next_row is not None else row)
    return replaced


def _section_compact_rows_from_raw_chunks(
    *,
    section_key: str,
    slot_specs: list[dict[str, Any]],
    text_chunk_records: list[dict[str, Any]],
    access_limited: bool,
) -> list[dict[str, Any]]:
    if not slot_specs or not text_chunk_records:
        return []
    if section_key == "introduction":
        candidates = _intro_chunk_candidates(text_chunk_records, max_items=len(slot_specs))
    elif section_key == "conclusion":
        # Conclusion is often implicit in discussion prose; borrow conclusion-like
        # discussion lines when explicit conclusion candidates are sparse.
        candidates = _raw_chunk_section_candidates(text_chunk_records, section=section_key, max_items=len(slot_specs))
        if len(candidates) < len(slot_specs):
            seen_keys = {_canonical_statement_text(str(item.get("statement", ""))) for item in candidates}
            discussion_pool = _raw_chunk_section_candidates(
                text_chunk_records,
                section="discussion",
                max_items=max(2 * len(slot_specs), 6),
            )
            for item in discussion_pool:
                statement = str(item.get("statement", "")).strip()
                key = _canonical_statement_text(statement)
                if not statement or not key or key in seen_keys:
                    continue
                if not _is_conclusion_like_statement(statement):
                    continue
                seen_keys.add(key)
                candidates.append(item)
                if len(candidates) >= len(slot_specs):
                    break
    else:
        candidates = _raw_chunk_section_candidates(text_chunk_records, section=section_key, max_items=len(slot_specs))
    if not candidates:
        return []

    rows: list[dict[str, Any]] = []
    for idx, slot_spec in enumerate(slot_specs):
        slot_key = str(slot_spec.get("slot_key", ""))
        slot_label = str(slot_spec.get("label", slot_key))
        if idx < len(candidates):
            candidate = candidates[idx]
            anchor = str(candidate.get("anchor", "")).strip()
            rows.append(
                {
                    "section_key": section_key,
                    "slot_key": slot_key,
                    "label": slot_label,
                    "statement": _compact_section_statement(str(candidate.get("statement", ""))),
                    "status": "found",
                    "evidence_refs": [anchor] if anchor else [],
                    "confidence": 0.56,
                }
            )
            continue
        rows.append(
            {
                "section_key": section_key,
                "slot_key": slot_key,
                "label": slot_label,
                "statement": SECTION_ACCESS_LIMITED_STATEMENT if access_limited else SECTION_NOT_FOUND_STATEMENT,
                "status": "access_limited" if access_limited else "not_found",
                "evidence_refs": [],
                "confidence": 0.0,
            }
        )
    return rows


def _is_fragment_like_statement(statement: str) -> bool:
    text = " ".join(str(statement or "").split()).strip()
    if not text:
        return True
    if len(text) < 36:
        return False
    lowered = text.lower()
    if FRAGMENT_START_RE.search(text):
        if not re.search(r"\b(we|this study|participants?|results?|findings?|authors?)\b", lowered[:80]):
            return True
    if text.startswith(("(", "[", ",", ";", ":")):
        return True
    if text.count("...") >= 1 and len(text) <= 80:
        return True
    return False


def _section_compact_candidates(section_key: str, section_packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = str(section_key or "").strip().lower()
    out: list[dict[str, Any]] = []
    for idx, packet in enumerate(section_packets):
        statement = str(packet.get("statement", "")).strip()
        if not statement:
            continue
        if _is_fragment_like_statement(statement):
            continue
        confidence = clamp_confidence(packet.get("confidence", 0.0))
        min_conf = SECTIONS_FOUND_CONFIDENCE_THRESHOLD
        if wanted in {"discussion", "conclusion"}:
            min_conf = 0.45
        if confidence < min_conf:
            continue
        quality_flags = {str(flag).strip().lower() for flag in packet.get("quality_flags", [])}
        if "missing_evidence" in quality_flags:
            continue
        if wanted == "methods" and re.search(
            r"\b(our results|these results|results?\s+(?:show|suggest|indicat|corroborate|emphasize))\b",
            statement,
            re.IGNORECASE,
        ):
            continue
        if wanted in {"discussion", "conclusion"} and _is_method_like_statement(statement) and not DISCUSSION_FALLBACK_RE.search(statement):
            continue
        evidence_refs = [str(ref).strip() for ref in packet.get("evidence_refs", []) if str(ref).strip()][:5]
        if not evidence_refs:
            continue
        anchor = str(packet.get("anchor", "")).strip() or evidence_refs[0]
        packet_key = str(packet.get("finding_id") or f"{_canonical_text(statement)}|{anchor}|{idx}")
        out.append(
            {
                "packet_key": packet_key,
                "statement": statement,
                "anchor": anchor,
                "blob_lower": f"{statement} {anchor} {packet.get('category', '')}".lower(),
                "confidence": confidence,
                "evidence_refs": evidence_refs,
                "source": "packet",
            }
        )
    return out


def _section_compact_candidates_from_extracted(
    section_key: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted = str(section_key or "").strip().lower()
    if wanted not in {"introduction", "methods", "results", "discussion", "conclusion"}:
        return []
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        statement = _strip_confidence_annotations(str(row.get("statement", "")).strip())
        if not statement or _is_noise_statement(statement) or _is_fragment_like_statement(statement):
            continue
        refs = [str(ref).strip() for ref in row.get("evidence_refs", []) if str(ref).strip()][:5]
        if not refs:
            continue
        if wanted == "results" and not _has_concrete_result_outcome(statement) and not RESULTS_SECTION_RE.search(statement):
            continue
        packet_key = f"section_extract:{wanted}:{idx}:{_canonical_statement_text(statement)}"
        out.append(
            {
                "packet_key": packet_key,
                "statement": statement,
                "anchor": refs[0],
                "blob_lower": f"{statement} {refs[0]} section_extract {wanted}".lower(),
                "confidence": 0.88,
                "evidence_refs": refs,
                "source": "section_extract",
            }
        )
    return out


def _result_support_compact_candidates(result_support_packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, packet in enumerate(result_support_packets):
        statement = _strip_confidence_annotations(str(packet.get("statement", "")).strip())
        if not statement:
            continue
        if _is_fragment_like_statement(statement):
            continue
        if _is_generic_visual_statement(statement) or _is_method_like_statement(statement):
            continue
        if not _has_concrete_result_outcome(statement):
            continue
        quality_flags = {str(flag).strip().lower() for flag in packet.get("quality_flags", [])}
        if "missing_evidence" in quality_flags:
            continue
        confidence = clamp_confidence(packet.get("confidence", 0.0))
        if confidence < RESULT_SUPPORT_FOUND_CONFIDENCE_THRESHOLD:
            continue
        evidence_refs = [str(ref).strip() for ref in packet.get("evidence_refs", []) if str(ref).strip()][:5]
        anchor = str(packet.get("anchor", "")).strip() or (evidence_refs[0] if evidence_refs else "")
        if anchor and anchor not in evidence_refs:
            evidence_refs = [anchor] + evidence_refs
        if not evidence_refs:
            continue
        packet_key = str(packet.get("finding_id") or f"{_canonical_text(statement)}|{anchor}|support|{idx}")
        out.append(
            {
                "packet_key": packet_key,
                "statement": statement,
                "anchor": anchor,
                "blob_lower": f"{statement} {anchor} {packet.get('category', '')} {packet.get('modality', '')}".lower(),
                "confidence": confidence,
                "evidence_refs": evidence_refs,
                "source": "result_support",
            }
        )
    return out


def _section_slot_score(section_key: str, slot_spec: dict[str, Any], candidate: dict[str, Any]) -> int:
    blob = str(candidate.get("blob_lower", ""))
    keywords = [str(item).lower() for item in slot_spec.get("keywords", [])]
    keyword_hits = sum(1 for keyword in keywords if keyword and keyword in blob)
    if keyword_hits <= 0:
        return 0
    score = keyword_hits * 50
    score += int(round(clamp_confidence(candidate.get("confidence", 0.0)) * 20))
    score += min(8, len(candidate.get("evidence_refs", [])) * 2)
    anchor = str(candidate.get("anchor", "")).strip()
    anchor_section = _anchor_section_label(anchor)
    if anchor_section == section_key:
        score += 10
    elif anchor_section != "unknown" and anchor_section != section_key:
        score -= 12
    source = str(candidate.get("source", "")).strip().lower()
    if source == "section_extract":
        score += 8
    elif source == "result_support":
        score += 4
    statement = str(candidate.get("statement", "")).strip()
    lowered = statement.lower()
    if _is_fragment_like_statement(statement):
        score -= 30
    if section_key == "methods":
        if re.search(r"\b(our results|these results|results?\s+(?:show|suggest|indicat|corroborate|emphasize))\b", lowered):
            score -= 40
        if RESULTS_SECTION_RE.search(statement):
            score -= 30
    elif section_key in {"discussion", "conclusion"}:
        if _is_method_like_statement(statement) and not DISCUSSION_FALLBACK_RE.search(statement):
            score -= 25
        if RESULTS_SECTION_RE.search(statement) and not re.search(r"\b(suggest|implication|limitations?|overall|conclusion)\b", lowered):
            score -= 12
    if _has_concrete_result_outcome(str(candidate.get("statement", ""))):
        score += 8
    return score


def _compact_section_statement(statement: str, max_chars: int = METHOD_STATEMENT_MAX_CHARS) -> str:
    text = " ".join(_strip_confidence_annotations(str(statement or "")).split()).strip()
    if not text:
        return ""
    return _compact_without_cutoff(text, max_chars=max_chars)


def _method_candidate(packet: dict[str, Any], *, idx: int) -> dict[str, Any] | None:
    statement = str(packet.get("statement", "")).strip()
    if not statement:
        return None
    if _is_fragment_like_statement(statement):
        return None
    quality_flags = {str(flag).strip().lower() for flag in packet.get("quality_flags", [])}
    if "missing_evidence" in quality_flags:
        return None
    confidence = clamp_confidence(packet.get("confidence", 0.0))
    if confidence < METHODS_FOUND_CONFIDENCE_THRESHOLD:
        return None

    evidence_refs = [str(ref).strip() for ref in packet.get("evidence_refs", []) if str(ref).strip()][:5]
    if not evidence_refs:
        return None

    anchor = str(packet.get("anchor", "")).strip() or evidence_refs[0]
    methods_anchor = _is_methods_anchor(anchor)
    method_signal = _has_method_signal(statement)
    result_anchor = _is_results_anchor(anchor)
    section_label = str(packet.get("section_label", "")).strip().lower()
    if section_label in {"discussion", "conclusion"} and not methods_anchor and not method_signal:
        return None
    if CONCLUSION_FALLBACK_RE.search(statement) and not methods_anchor and not method_signal:
        return None
    if result_anchor and not method_signal:
        return None

    category_tokens = _category_tokens(str(packet.get("category", "")))
    packet_key = str(packet.get("finding_id") or f"{_canonical_text(statement)}|{anchor}|{idx}")
    return {
        "packet_key": packet_key,
        "statement": statement,
        "anchor": anchor,
        "anchor_key": _canonical_text(anchor),
        "anchor_lower": anchor.lower(),
        "blob_lower": f"{statement} {anchor}".lower(),
        "methods_anchor": methods_anchor,
        "method_signal": method_signal,
        "result_anchor": result_anchor,
        "category_tokens": set(category_tokens),
        "evidence_refs": evidence_refs,
        "confidence": confidence,
    }


def _method_slot_score(slot: dict[str, Any], candidate: dict[str, Any]) -> int:
    score = 0
    slot_key = str(slot.get("slot_key", ""))
    slot_keywords = [str(item).lower() for item in slot.get("keywords", [])]
    category_hints = {str(item).lower() for item in slot.get("category_hints", set())}
    blob = str(candidate.get("blob_lower", ""))
    keyword_hit = any(keyword and keyword in blob for keyword in slot_keywords)
    category_hit = bool(category_hints & set(candidate.get("category_tokens", set())))

    if slot_key in STRICT_METHOD_SLOT_KEYWORDS and not keyword_hit:
        return 0
    if not keyword_hit and not category_hit:
        return 0
    if keyword_hit:
        score += 80
    if category_hit:
        score += 26
    if candidate.get("methods_anchor"):
        score += 20
    if candidate.get("method_signal"):
        score += 16
    if candidate.get("result_anchor"):
        score -= 25

    score += _anchor_specificity_score(str(candidate.get("anchor", "")))
    score += int(round(clamp_confidence(candidate.get("confidence", 0.0)) * 12))
    score += min(8, len(candidate.get("evidence_refs", [])) * 2)
    return score


def _is_results_anchor(anchor: str) -> bool:
    return bool(RESULTS_ANCHOR_RE.search(str(anchor or "")))


def _anchor_specificity_score(anchor: str) -> int:
    value = str(anchor or "").lower()
    if not value:
        return 0
    score = 0
    if "section:" in value:
        score += 4
    if any(token in value for token in ("method", "participants", "analysis", "protocol", "acquisition", "design")):
        score += 14
    if "body" in value:
        score += 4
    if _is_results_anchor(value):
        score -= 18
    return score


def _compact_method_statement(statement: str, max_chars: int = METHOD_STATEMENT_MAX_CHARS) -> str:
    text = " ".join(_strip_confidence_annotations(str(statement or "")).split()).strip()
    if not text:
        return ""
    text = _strip_method_hedges(text)
    return _compact_without_cutoff(text, max_chars=max_chars)


def _strip_method_hedges(text: str) -> str:
    value = str(text or "").strip()
    replacements = (
        (r"^the study suggests that\s+", ""),
        (r"^the authors suggest that\s+", ""),
        (r"^the study indicates that\s+", ""),
        (r"^the authors report that\s+", ""),
    )
    out = value
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def _strip_confidence_annotations(text: str) -> str:
    clean = " ".join(CONFIDENCE_INLINE_RE.sub(" ", str(text or "")).split()).strip()
    if not clean:
        return ""
    clean = re.sub(r"^\s*[-*•]+\s*", "", clean).strip()
    # Remove citation-style numeric prefixes (for example, "4, 5 In contrast ...").
    while True:
        updated = LEADING_CITATION_PREFIX_RE.sub("", clean, count=1).strip()
        if updated == clean:
            break
        clean = updated
    # Drop trailing ellipsis markers that commonly indicate clipped extraction text.
    clean = re.sub(r"(?:\.\s*){3,}\s*$", "", clean).strip()
    clean = re.sub(r"…+\s*$", "", clean).strip()
    return clean


def _clean_section_statement(statement: str) -> str:
    clean = _strip_confidence_annotations(statement)
    if not clean:
        return ""
    trimmed = _trim_dangling_tail_tokens(clean)
    if trimmed:
        clean = trimmed
    if _is_incomplete_statement(clean):
        return ""
    return clean


def _is_redundant_statement_key(statement_key: str, seen_keys: set[str]) -> bool:
    if not statement_key or not seen_keys:
        return False
    for seen_key in seen_keys:
        if _are_near_duplicate_statement_keys(statement_key, seen_key):
            return True
    return False


def _methodology_details_from_compact(
    methods_compact: list[dict[str, Any]], max_items: int = 20
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for slot in methods_compact:
        if str(slot.get("status", "")) != "found":
            continue
        slot_key = str(slot.get("slot_key", "")).strip()
        statement = str(slot.get("statement", "")).strip()
        if not statement:
            continue
        details.append(
            {
                "statement": statement,
                "category": METHOD_SLOT_CATEGORY.get(slot_key, "methods"),
                "evidence_refs": [str(ref) for ref in slot.get("evidence_refs", [])][:5],
                "confidence": clamp_confidence(slot.get("confidence", 0.0)),
            }
        )
        if len(details) >= max_items:
            break
    return details


def _methods_strengths_from_compact(methods_compact: list[dict[str, Any]], max_items: int = 5) -> list[str]:
    strengths: list[str] = []
    for slot in methods_compact:
        if str(slot.get("status", "")) != "found":
            continue
        label = str(slot.get("label", "")).strip()
        statement = str(slot.get("statement", "")).strip()
        if not label or not statement:
            continue
        strengths.append(f"{label}: {statement}")
        if len(strengths) >= max_items:
            break
    return strengths


def _methods_weaknesses_from_compact(methods_compact: list[dict[str, Any]], max_items: int = 5) -> list[str]:
    by_slot = {str(slot.get("slot_key", "")): slot for slot in methods_compact}
    priority = [
        "study_design",
        "sample_population",
        "statistical_model",
        "covariates",
        "missing_data_and_sensitivity",
        "outcomes_measures",
        "reproducibility_transparency",
        "method_limitations",
        "inclusion_criteria",
        "exclusion_criteria",
        "intervention_or_exposure",
        "data_acquisition_protocol",
    ]
    weaknesses: list[str] = []
    for slot_key in priority:
        slot = by_slot.get(slot_key)
        if not slot:
            continue
        status = str(slot.get("status", "not_found"))
        if status == "found":
            continue
        label = str(slot.get("label", slot_key)).strip()
        if status == "access_limited":
            weaknesses.append(f"{label}: source appears access-limited; upload full PDF for full methods extraction.")
        else:
            weaknesses.append(f"{label}: N/A in parsed text.")
        if len(weaknesses) >= max_items:
            break
    return weaknesses


def _methodology_details(text_packets: list[dict[str, Any]], max_items: int = 20) -> list[dict[str, Any]]:
    preferred_categories = {
        "methods",
        "stats",
        "reproducibility",
        "ethics",
        "limitations",
        "clinical",
    }
    category_order = ["methods", "stats", "clinical", "limitations", "ethics", "reproducibility"]
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, packet in enumerate(text_packets):
        raw_category = str(packet.get("category", "other")).lower()
        tokens = _category_tokens(raw_category)
        matching_tokens = [token for token in tokens if token in preferred_categories]
        statement = str(packet.get("statement", "")).strip()
        if not statement:
            continue
        quality_flags = {str(flag).strip().lower() for flag in packet.get("quality_flags", [])}
        if "missing_evidence" in quality_flags:
            continue
        anchor = str(packet.get("anchor", "")).strip()
        evidence_refs = [str(ref) for ref in packet.get("evidence_refs", [])][:5]
        if not anchor and evidence_refs:
            anchor = evidence_refs[0]
        methods_anchor = _is_methods_anchor(anchor)
        method_signal = _has_method_signal(statement)

        if not matching_tokens and not methods_anchor and not method_signal:
            continue
        if "clinical" in matching_tokens and not methods_anchor and not method_signal:
            continue
        if _looks_like_outcome_summary(statement) and not methods_anchor and "methods" not in matching_tokens:
            continue

        key = _canonical_text(statement)
        if not key or key in seen:
            continue
        seen.add(key)
        category = _pick_method_category(
            matching_tokens=matching_tokens,
            category_order=category_order,
            methods_anchor=methods_anchor,
            method_signal=method_signal,
        )
        detail = {
            "statement": statement,
            "category": category,
            "evidence_refs": evidence_refs,
            "confidence": clamp_confidence(packet.get("confidence", 0.0)),
        }
        ranked.append(
            {
                "score": _method_detail_score(
                    matching_tokens=matching_tokens,
                    methods_anchor=methods_anchor,
                    method_signal=method_signal,
                    outcome_summary=_looks_like_outcome_summary(statement),
                    confidence=detail["confidence"],
                    evidence_refs=evidence_refs,
                ),
                "idx": idx,
                "detail": detail,
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["idx"]))
    return [item["detail"] for item in ranked[:max_items]]


def _is_methods_anchor(anchor: str) -> bool:
    if not anchor:
        return False
    return bool(METHOD_ANCHOR_RE.search(str(anchor)))


def _has_method_signal(statement: str) -> bool:
    if not statement:
        return False
    return bool(METHOD_SIGNAL_RE.search(statement))


def _looks_like_outcome_summary(statement: str) -> bool:
    if not statement:
        return False
    return bool(OUTCOME_SUMMARY_RE.search(statement))


def _pick_method_category(
    *,
    matching_tokens: list[str],
    category_order: list[str],
    methods_anchor: bool,
    method_signal: bool,
) -> str:
    if "methods" in matching_tokens:
        return "methods"
    if methods_anchor or method_signal:
        return "methods"
    for category in category_order:
        if category in matching_tokens:
            return category
    return matching_tokens[0] if matching_tokens else "other"


def _method_detail_score(
    *,
    matching_tokens: list[str],
    methods_anchor: bool,
    method_signal: bool,
    outcome_summary: bool,
    confidence: float,
    evidence_refs: list[str],
) -> int:
    score = 0
    if "methods" in matching_tokens:
        score += 60
    if "stats" in matching_tokens:
        score += 40
    if "reproducibility" in matching_tokens:
        score += 28
    if "limitations" in matching_tokens:
        score += 22
    if "ethics" in matching_tokens:
        score += 16
    if "clinical" in matching_tokens:
        score += 10
    if methods_anchor:
        score += 50
    if method_signal:
        score += 30
    if outcome_summary:
        score -= 30
    score += min(8, len(evidence_refs) * 2)
    score += int(round(clamp_confidence(confidence) * 10))
    return score


def _canonical_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def _canonical_statement_text(value: str) -> str:
    clean = _strip_confidence_annotations(str(value or ""))
    tokens = CANONICAL_STATEMENT_TOKEN_RE.findall(clean.lower())
    return " ".join(tokens)


def _statement_stem_key(value: str, *, min_tokens: int = 8, max_tokens: int = 14) -> str:
    tokens = _canonical_statement_text(value).split()
    if len(tokens) < min_tokens:
        return ""
    return " ".join(tokens[:max_tokens])


def _are_near_duplicate_statement_keys(left: str, right: str) -> bool:
    a = _canonical_statement_text(left)
    b = _canonical_statement_text(right)
    if not a or not b:
        return False
    if a in b or b in a:
        return True

    a_tokens = a.split()
    b_tokens = b.split()
    if not a_tokens or not b_tokens:
        return False

    min_len = min(len(a_tokens), len(b_tokens))
    if min_len >= 10:
        prefix_matches = 0
        for idx in range(min_len):
            if a_tokens[idx] != b_tokens[idx]:
                break
            prefix_matches += 1
        if (prefix_matches / float(min_len)) >= 0.86:
            return True

    a_set = set(a_tokens)
    b_set = set(b_tokens)
    inter = len(a_set & b_set)
    if inter <= 0:
        return False
    overlap_max = inter / max(len(a_set), len(b_set))
    overlap_min = inter / max(1, min(len(a_set), len(b_set)))
    if overlap_max >= 0.82:
        return True
    if overlap_min >= 0.90 and overlap_max >= 0.56:
        return True
    return False


def _category_tokens(value: str) -> list[str]:
    lowered = str(value or "").lower()
    return [token for token in re.split(r"[^a-z0-9]+", lowered) if token]


def _unique_strings(items: list[str], max_items: int | None = None) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = _canonical_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)
        if max_items is not None and len(unique) >= max_items:
            break
    return unique


SECTION_MIN_TARGETS = {
    "methods": 8,
    "results": 12,
    "discussion": 6,
    "conclusion": 4,
}


def _section_item_passes_fidelity(section_key: str, item: dict[str, Any]) -> bool:
    section = str(section_key or "").strip().lower()
    statement = _clean_section_statement(str(item.get("statement", "")).strip())
    if not statement:
        return False
    if _is_noise_statement(statement) or _is_layout_artifact_statement(statement):
        return False
    if _is_fragment_like_statement(statement):
        return False

    anchor = str(item.get("anchor", "")).strip()
    anchor_section = _anchor_section_label(anchor)
    lowered = statement.lower()
    panel_label_prefix = _has_panel_prefix(lowered)

    if section == "introduction":
        return _is_intro_fidelity_statement(statement)

    if section == "methods":
        return _is_method_like_statement(statement) or _has_method_signal(statement) or _is_methods_anchor(anchor)

    if section == "results":
        has_concrete = _has_concrete_result_outcome(statement)
        has_high_signal = _is_high_signal_result_statement(statement)
        if panel_label_prefix:
            return False
        if _is_visual_annotation_statement(statement):
            return False
        if _is_method_like_statement(statement) and not has_concrete:
            return False
        if _is_generic_visual_statement(statement) and not has_concrete:
            return False
        if not (has_concrete or has_high_signal or anchor_section == "results" or RESULTS_SECTION_RE.search(statement)):
            return False
        return True

    if section == "discussion":
        if panel_label_prefix:
            return False
        if _is_visual_annotation_statement(statement):
            return False
        if _is_method_like_statement(statement) and not DISCUSSION_FALLBACK_RE.search(statement):
            return False
        if _has_concrete_result_outcome(statement) and not re.search(
            r"\b(suggest|implication|interpret|limitation|generalizab|context|may|caution)\b",
            statement,
            re.IGNORECASE,
        ):
            return False
        return True

    if section == "conclusion":
        if panel_label_prefix or _is_visual_annotation_statement(statement):
            return False
        if _is_method_like_statement(statement):
            return False
        if re.search(
            r"\b(to examine|we analyzed|was used to|using linear|participants?\s+were|sample\s+included|consisted of)\b",
            statement,
            re.IGNORECASE,
        ):
            return False
        if _is_conclusion_like_statement(statement):
            return True
        if anchor_section == "conclusion":
            return bool(CONCLUSION_SIGNAL_RE.search(statement) or _has_concrete_result_outcome(statement))
        return bool(
            _has_concrete_result_outcome(statement)
            and re.search(
                r"\b(revealed|showed|demonstrated|identified|found|no significant|without significant)\b",
                statement,
                re.IGNORECASE,
            )
        )

    return True


def _filter_section_items_by_fidelity(section_key: str, items: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    section = str(section_key or "").strip().lower()
    if section not in {"introduction", "methods", "results", "discussion", "conclusion"}:
        return _dedupe_section_items(items, max_items=max_items)
    filtered = [item for item in items if isinstance(item, dict) and _section_item_passes_fidelity(section, item)]
    return _dedupe_section_items(filtered, max_items=max_items)


def _enforce_min_section_coverage(
    section_items: list[dict[str, Any]],
    fallback_candidates: list[dict[str, Any]],
    *,
    section_key: str,
    min_items: int,
    source_modality: str = "text",
    result_evidence_type: str | None = None,
) -> tuple[list[dict[str, Any]], list[str] | None]:
    if len(section_items) >= min_items:
        return section_items, None

    existing_hashes = {
        " ".join(str(item.get("statement", "")).lower().split()) for item in section_items if str(item.get("statement", "")).strip()
    }
    extras: list[dict[str, Any]] = []
    for packet in fallback_candidates:
        if len(section_items) + len(extras) >= min_items:
            break
        statement = str(packet.get("statement", "")).strip()
        if not statement:
            continue
        key = " ".join(statement.lower().split())
        if key in existing_hashes:
            continue
        candidate = _section_evidence_from_packet(
            packet,
            source_modality,
            flags=["min_coverage", "fallback"],
            result_evidence_type=result_evidence_type,
            is_untrusted=True,
        )
        if not _section_item_passes_fidelity(section_key, candidate):
            continue
        existing_hashes.add(key)
        extras.append(candidate)

    if not extras:
        return section_items, None

    reason = f"Section coverage was below target; added low-confidence {section_key} candidates for recall recovery."
    section_items = _dedupe_section_items(section_items + extras, max_items=max(min_items, len(section_items) + len(extras)))
    return section_items, reason


def _build_detailed_sections(
    *,
    text_packets: list[dict[str, Any]],
    table_packets: list[dict[str, Any]],
    figure_packets: list[dict[str, Any]],
    supp_packets: list[dict[str, Any]],
    methods_compact: list[dict[str, Any]],
    analysis_notes: list[str],
    text_chunk_records: list[dict[str, Any]] | None = None,
    sections_extracted: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    fallback_notes: list[str] = []
    diagnostics: dict[str, Any] = {}
    extracted = sections_extracted or {}

    def _record_fallback_note(section: str, reason: str | None) -> None:
        if not reason:
            return
        note = f"{section}: {reason}"
        if note not in fallback_notes:
            fallback_notes.append(note)

    intro_explicit = _select_text_packets_by_section(
        text_packets,
        "introduction",
        min_section_confidence=0.7,
        preferred_sources={"anchor"},
    )
    intro_soft = _select_text_packets_by_section(
        text_packets,
        "introduction",
        min_section_confidence=0.55,
        preferred_sources={"category", "lexical"},
    )
    intro_items = _dedupe_section_items(
        _section_evidence_from_extracted_bullets(extracted.get("introduction", []), section_key="introduction")
        + [
            _section_evidence_from_packet(packet, "text")
            for packet in (intro_explicit + intro_soft)
            if _is_intro_fidelity_statement(str(packet.get("statement", "")))
        ],
        max_items=18,
    )
    intro_items = _filter_section_items_by_fidelity("introduction", intro_items, max_items=18)
    intro_fallback_reason = None
    intro_cutoff_index = _intro_cutoff_index(text_packets)
    if len(intro_items) < 6:
        intro_fallback_packets = [packet for packet in text_packets if _is_intro_fallback_candidate(packet)]
        intro_positional_packets = [
            packet
            for idx, packet in enumerate(text_packets)
            if _is_intro_positional_candidate(packet, idx=idx, cutoff_index=intro_cutoff_index)
        ]
        extra_intro_items = _dedupe_section_items(
            [
                _section_evidence_from_packet(packet, "text", flags=["fallback"])
                for packet in (intro_fallback_packets + intro_positional_packets)
            ],
            max_items=12,
        )
        if extra_intro_items:
            intro_items = _dedupe_section_items(intro_items + extra_intro_items, max_items=18)
            intro_fallback_reason = (
                "Explicit introduction anchors were sparse; used background/objective lines and pre-Methods/Results context."
            )
            fallback_notes.append(f"Introduction: {intro_fallback_reason}")
    if len(intro_items) < 6 and text_chunk_records:
        raw_intro_candidates = _intro_chunk_candidates(text_chunk_records, max_items=8)
        raw_intro_items = _dedupe_section_items(
            [
                {
                    "statement": item.get("statement", ""),
                    "anchor": item.get("anchor", ""),
                    "evidence_refs": [item.get("anchor", "")] if item.get("anchor", "") else [],
                    "source_modality": "text",
                    "confidence": 0.58,
                    "section_confidence": 0.56,
                    "flags": ["fallback", "raw_chunk"],
                }
                for item in raw_intro_candidates
                if str(item.get("statement", "")).strip()
            ],
            max_items=12,
        )
        if raw_intro_items:
            intro_items = _dedupe_section_items(intro_items + raw_intro_items, max_items=18)
            intro_fallback_reason = (
                "Explicit introduction packets were sparse; supplemented with early-body text before Methods/Results."
            )
            fallback_notes.append(f"Introduction: {intro_fallback_reason}")

    methods_explicit = _select_text_packets_by_section(text_packets, "methods")
    methods_coverage_candidates: list[dict[str, Any]] = []
    methods_items = _dedupe_section_items(
        _section_evidence_from_extracted_bullets(extracted.get("methods", []), section_key="methods")
        + _section_evidence_from_method_slots(methods_compact)
        + [_section_evidence_from_packet(packet, "text") for packet in methods_explicit],
        max_items=24,
    )
    methods_items = _filter_section_items_by_fidelity("methods", methods_items, max_items=24)
    methods_fallback_reason = None
    if len(methods_items) < 6:
        methods_fallback_packets = [packet for packet in text_packets if _is_methods_fallback_candidate(packet)]
        methods_coverage_candidates.extend(methods_fallback_packets)
        methods_fallback_items = _dedupe_section_items(
            [_section_evidence_from_packet(packet, "text", flags=["fallback"]) for packet in methods_fallback_packets],
            max_items=10,
        )
        if methods_fallback_items:
            methods_items = _dedupe_section_items(methods_items + methods_fallback_items, max_items=24)
            methods_fallback_reason = (
                "Explicit methods anchors were sparse; supplemented with method-like protocol/statistical lines."
            )
            _record_fallback_note("Methods", methods_fallback_reason)
    methods_raw_packet_candidates: list[dict[str, Any]] = []
    if len(methods_items) < 8 and text_chunk_records:
        raw_methods = _raw_chunk_section_candidates(text_chunk_records, section="methods", max_items=8)
        for item in raw_methods:
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            methods_raw_packet_candidates.append(
                {
                    "statement": statement,
                    "anchor": item.get("anchor", ""),
                    "evidence_refs": [item.get("anchor", "")] if item.get("anchor", "") else [],
                    "confidence": 0.56,
                    "section_confidence": 0.56,
                }
            )
        raw_method_items = _dedupe_section_items(
            [
                {
                    "statement": item.get("statement", ""),
                    "anchor": item.get("anchor", ""),
                    "evidence_refs": [item.get("anchor", "")] if item.get("anchor", "") else [],
                    "source_modality": "text",
                    "confidence": 0.56,
                    "section_confidence": 0.56,
                    "flags": ["fallback", "raw_chunk"],
                }
                for item in raw_methods
                if str(item.get("statement", "")).strip()
            ],
            max_items=10,
        )
        if raw_method_items:
            methods_items = _dedupe_section_items(methods_items + raw_method_items, max_items=24)
            if not methods_fallback_reason:
                methods_fallback_reason = (
                    "Methods section was sparse; supplemented with protocol-focused statements from raw text chunks."
                )
                _record_fallback_note("Methods", methods_fallback_reason)
    methods_min_coverage_items, methods_min_coverage_reason = _enforce_min_section_coverage(
        methods_items,
        methods_coverage_candidates + methods_raw_packet_candidates,
        section_key="methods",
        min_items=SECTION_MIN_TARGETS["methods"],
        source_modality="text",
    )
    if methods_min_coverage_reason:
        methods_items = methods_min_coverage_items
        methods_fallback_reason = methods_min_coverage_reason
        _record_fallback_note("Methods", methods_min_coverage_reason)
    if not methods_items and analysis_notes:
        if not methods_fallback_reason:
            methods_fallback_reason = "Methods extraction limited by source access or parser coverage."
            _record_fallback_note("Methods", methods_fallback_reason)

    results_text_explicit = _select_text_packets_by_section(text_packets, "results")
    results_text_items = _dedupe_section_items(
        _section_evidence_from_extracted_bullets(extracted.get("results", []), section_key="results")
        + [
            _section_evidence_from_packet(packet, "text", result_evidence_type="text_primary")
            for packet in _filter_result_text_packets(results_text_explicit)
        ],
        max_items=24,
    )
    results_text_items = _filter_section_items_by_fidelity("results", results_text_items, max_items=24)
    results_fallback_reason = None
    results_coverage_candidates: list[dict[str, Any]] = []
    if len(results_text_items) < 6:
        results_text_fallback = [
            packet
            for packet in text_packets
            if _packet_section_label(packet) not in {"methods", "introduction"}
            and _has_concrete_result_outcome(str(packet.get("statement", "")))
            and not _is_method_like_statement(str(packet.get("statement", "")))
        ]
        results_coverage_candidates.extend(results_text_fallback)
        fallback_items = _dedupe_section_items(
            [
                _section_evidence_from_packet(
                    packet,
                    "text",
                    flags=["fallback"],
                    result_evidence_type="text_primary",
                )
                for packet in results_text_fallback
            ],
            max_items=14,
        )
        if fallback_items:
            results_text_items = _dedupe_section_items(results_text_items + fallback_items, max_items=24)
            results_fallback_reason = (
                "Explicit results anchors were limited; supplemented with concrete outcome statements from nearby sections."
            )
            _record_fallback_note("Results", results_fallback_reason)
    raw_results_packet_candidates: list[dict[str, Any]] = []
    if len(results_text_items) < 8 and text_chunk_records:
        raw_results = _raw_chunk_section_candidates(text_chunk_records, section="results", max_items=8)
        for item in raw_results:
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            raw_results_packet_candidates.append(
                {
                    "statement": statement,
                    "anchor": item.get("anchor", ""),
                    "evidence_refs": [item.get("anchor", "")] if item.get("anchor", "") else [],
                    "confidence": 0.56,
                    "section_confidence": 0.56,
                }
            )
        raw_result_items = _dedupe_section_items(
            [
                {
                    "statement": item.get("statement", ""),
                    "anchor": item.get("anchor", ""),
                    "evidence_refs": [item.get("anchor", "")] if item.get("anchor", "") else [],
                    "source_modality": "text",
                    "confidence": 0.56,
                    "section_confidence": 0.56,
                    "flags": ["fallback", "raw_chunk"],
                    "result_evidence_type": "text_primary",
                }
                for item in raw_results
                if str(item.get("statement", "")).strip()
            ],
            max_items=10,
        )
        if raw_result_items:
            results_text_items = _dedupe_section_items(results_text_items + raw_result_items, max_items=24)
            if not results_fallback_reason:
                results_fallback_reason = (
                    "Results section was sparse; supplemented with concrete outcome statements from raw text chunks."
                )
                _record_fallback_note("Results", results_fallback_reason)
    results_min_coverage_items, results_min_coverage_reason = _enforce_min_section_coverage(
        results_text_items,
        results_coverage_candidates + raw_results_packet_candidates,
        section_key="results",
        min_items=SECTION_MIN_TARGETS["results"],
        source_modality="text",
        result_evidence_type="text_primary",
    )
    if results_min_coverage_reason:
        results_text_items = results_min_coverage_items
        results_fallback_reason = results_min_coverage_reason
        _record_fallback_note("Results", results_min_coverage_reason)
    results_text_items = _filter_section_items_by_fidelity("results", results_text_items, max_items=24)

    results_media_items = _dedupe_section_items(
        [
            _section_evidence_from_packet(packet, "table", result_evidence_type="media_support")
            for packet in _filter_result_media_packets(table_packets)
        ]
        + [
            _section_evidence_from_packet(packet, "figure", result_evidence_type="media_support")
            for packet in _filter_result_media_packets(figure_packets)
        ]
        + [
            _section_evidence_from_packet(packet, "supplement", result_evidence_type="media_support")
            for packet in _filter_result_media_packets(supp_packets)
        ],
        max_items=16,
    )
    results_media_items = _filter_section_items_by_fidelity("results", results_media_items, max_items=16)
    results_items = _dedupe_section_items(results_text_items + results_media_items, max_items=36)
    results_items = _filter_section_items_by_fidelity("results", results_items, max_items=36)

    discussion_explicit = _select_text_packets_by_section(text_packets, "discussion", min_section_confidence=0.6)
    discussion_coverage_candidates: list[dict[str, Any]] = []
    discussion_items = _dedupe_section_items(
        _section_evidence_from_extracted_bullets(extracted.get("discussion", []), section_key="discussion")
        + [
            _section_evidence_from_packet(packet, "text")
            for packet in discussion_explicit
            if not _is_method_like_statement(str(packet.get("statement", "")))
        ],
        max_items=18,
    )
    discussion_items = _filter_section_items_by_fidelity("discussion", discussion_items, max_items=18)
    discussion_fallback_reason = None
    if len(discussion_items) < 4:
        discussion_fallback_packets = [packet for packet in text_packets if _is_discussion_fallback_candidate(packet)]
        discussion_coverage_candidates.extend(discussion_fallback_packets)
        discussion_fallback_items = _dedupe_section_items(
            [_section_evidence_from_packet(packet, "text", flags=["fallback"]) for packet in discussion_fallback_packets],
            max_items=10,
        )
        if discussion_fallback_items:
            discussion_items = _dedupe_section_items(discussion_items + discussion_fallback_items, max_items=18)
            discussion_fallback_reason = (
                "Explicit discussion anchors were limited; supplemented with interpretive/limitations statements."
            )
            _record_fallback_note("Discussion", discussion_fallback_reason)
    raw_discussion_packet_candidates: list[dict[str, Any]] = []
    if len(discussion_items) < 5 and text_chunk_records:
        raw_discussion = _raw_chunk_section_candidates(text_chunk_records, section="discussion", max_items=7)
        for item in raw_discussion:
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            raw_discussion_packet_candidates.append(
                {
                    "statement": statement,
                    "anchor": item.get("anchor", ""),
                    "evidence_refs": [item.get("anchor", "")] if item.get("anchor", "") else [],
                    "confidence": 0.56,
                    "section_confidence": 0.56,
                }
            )
        raw_discussion_items = _dedupe_section_items(
            [
                {
                    "statement": item.get("statement", ""),
                    "anchor": item.get("anchor", ""),
                    "evidence_refs": [item.get("anchor", "")] if item.get("anchor", "") else [],
                    "source_modality": "text",
                    "confidence": 0.56,
                    "section_confidence": 0.56,
                    "flags": ["fallback", "raw_chunk"],
                }
                for item in raw_discussion
                if str(item.get("statement", "")).strip()
            ],
            max_items=10,
        )
        if raw_discussion_items:
            discussion_items = _dedupe_section_items(discussion_items + raw_discussion_items, max_items=18)
            if not discussion_fallback_reason:
                discussion_fallback_reason = (
                    "Discussion section was sparse; supplemented with interpretive statements from raw text chunks."
                )
                _record_fallback_note("Discussion", discussion_fallback_reason)
    discussion_min_coverage_items, discussion_min_coverage_reason = _enforce_min_section_coverage(
        discussion_items,
        discussion_coverage_candidates + raw_discussion_packet_candidates,
        section_key="discussion",
        min_items=SECTION_MIN_TARGETS["discussion"],
        source_modality="text",
    )
    if discussion_min_coverage_reason:
        discussion_items = discussion_min_coverage_items
        discussion_fallback_reason = discussion_min_coverage_reason
        _record_fallback_note("Discussion", discussion_min_coverage_reason)
    discussion_items = _filter_section_items_by_fidelity("discussion", discussion_items, max_items=18)

    conclusion_explicit = _select_text_packets_by_section(text_packets, "conclusion", min_section_confidence=0.6)
    conclusion_coverage_candidates: list[dict[str, Any]] = []
    conclusion_items = _dedupe_section_items(
        _section_evidence_from_extracted_bullets(extracted.get("conclusion", []), section_key="conclusion")
        + [
            _section_evidence_from_packet(packet, "text")
            for packet in conclusion_explicit
            if not _is_method_like_statement(str(packet.get("statement", "")))
        ],
        max_items=16,
    )
    conclusion_items = _filter_section_items_by_fidelity("conclusion", conclusion_items, max_items=16)
    conclusion_fallback_reason = None
    if len(conclusion_items) < 4:
        conclusion_fallback_pool = [
            packet
            for packet in (discussion_explicit + text_packets)
            if _is_conclusion_fallback_candidate(packet)
        ]
        conclusion_coverage_candidates.extend(conclusion_fallback_pool)
        fallback_items = _dedupe_section_items(
            [_section_evidence_from_packet(packet, "text", flags=["fallback"]) for packet in conclusion_fallback_pool],
            max_items=10,
        )
        extracted_discussion_fallback = _dedupe_section_items(
            [
                item
                for item in _section_evidence_from_extracted_bullets(
                    extracted.get("discussion", []),
                    section_key="conclusion",
                )
                if _is_conclusion_like_statement(str(item.get("statement", "")))
            ],
            max_items=8,
        )
        combined_fallback = _dedupe_section_items(fallback_items + extracted_discussion_fallback, max_items=12)
        if combined_fallback:
            conclusion_items = _dedupe_section_items(conclusion_items + combined_fallback, max_items=16)
            if not conclusion_fallback_reason:
                conclusion_fallback_reason = (
                    "Conclusion section was sparse; supplemented with conclusion-like discussion statements."
                )
                _record_fallback_note("Conclusion", conclusion_fallback_reason)
            if not conclusion_explicit:
                conclusion_fallback_reason = (
                    "No explicit conclusion heading detected; used conclusion-like discussion statements."
                )
                _record_fallback_note("Conclusion", conclusion_fallback_reason)

    raw_conclusion_packet_candidates: list[dict[str, Any]] = []
    if len(conclusion_items) < 4 and text_chunk_records:
        raw_conclusion = _raw_chunk_section_candidates(text_chunk_records, section="conclusion", max_items=6)
        if len(raw_conclusion) < 6:
            seen_keys = {_canonical_statement_text(str(item.get("statement", ""))) for item in raw_conclusion}
            raw_discussion = _raw_chunk_section_candidates(text_chunk_records, section="discussion", max_items=10)
            for item in raw_discussion:
                statement = str(item.get("statement", "")).strip()
                key = _canonical_statement_text(statement)
                if not statement or not key or key in seen_keys:
                    continue
                if not _is_conclusion_like_statement(statement):
                    continue
                seen_keys.add(key)
                raw_conclusion.append(item)
                if len(raw_conclusion) >= 6:
                    break
        for item in raw_conclusion:
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            if not _is_conclusion_like_statement(statement) and _anchor_section_label(str(item.get("anchor", ""))) != "conclusion":
                continue
            raw_conclusion_packet_candidates.append(
                {
                    "statement": statement,
                    "anchor": item.get("anchor", ""),
                    "evidence_refs": [item.get("anchor", "")] if item.get("anchor", "") else [],
                    "confidence": 0.56,
                    "section_confidence": 0.56,
                }
            )
        raw_conclusion_items = _dedupe_section_items(
            [
                {
                    "statement": item.get("statement", ""),
                    "anchor": item.get("anchor", ""),
                    "evidence_refs": [item.get("anchor", "")] if item.get("anchor", "") else [],
                    "source_modality": "text",
                    "confidence": 0.56,
                    "section_confidence": 0.56,
                    "flags": ["fallback", "raw_chunk"],
                }
                for item in raw_conclusion_packet_candidates
                if str(item.get("statement", "")).strip()
            ],
            max_items=10,
        )
        if raw_conclusion_items:
            conclusion_items = _dedupe_section_items(conclusion_items + raw_conclusion_items, max_items=16)
            if not conclusion_fallback_reason:
                conclusion_fallback_reason = (
                    "Conclusion section was sparse; supplemented with conclusion-like statements from raw text chunks."
                )
                _record_fallback_note("Conclusion", conclusion_fallback_reason)
            conclusion_coverage_candidates.extend(raw_conclusion_packet_candidates)

    conclusion_min_coverage_items, conclusion_min_coverage_reason = _enforce_min_section_coverage(
        conclusion_items,
        conclusion_coverage_candidates + raw_conclusion_packet_candidates,
        section_key="conclusion",
        min_items=SECTION_MIN_TARGETS["conclusion"],
        source_modality="text",
    )
    if conclusion_min_coverage_reason:
        conclusion_items = conclusion_min_coverage_items
        conclusion_fallback_reason = conclusion_min_coverage_reason
        _record_fallback_note("Conclusion", conclusion_min_coverage_reason)
    conclusion_items = _filter_section_items_by_fidelity("conclusion", conclusion_items, max_items=16)

    sections = {
        "introduction": _build_section_block(
            "Introduction",
            intro_items,
            fallback_used=bool(intro_fallback_reason),
            fallback_reason=intro_fallback_reason,
            max_items=10,
        ),
        "methods": _build_section_block(
            "Methods",
            methods_items,
            fallback_used=bool(methods_fallback_reason),
            fallback_reason=methods_fallback_reason,
            max_items=12,
        ),
        "results": _build_section_block(
            "Results",
            results_items,
            fallback_used=bool(results_fallback_reason),
            fallback_reason=results_fallback_reason,
            max_items=12,
        ),
        "discussion": _build_section_block(
            "Discussion",
            discussion_items,
            fallback_used=bool(discussion_fallback_reason),
            fallback_reason=discussion_fallback_reason,
            max_items=8,
        ),
        "conclusion": _build_section_block(
            "Conclusion",
            conclusion_items,
            fallback_used=bool(conclusion_fallback_reason),
            fallback_reason=conclusion_fallback_reason,
            max_items=6,
        ),
    }
    sections, cross_section_dedupe = _dedupe_items_across_sections(sections)
    if int(cross_section_dedupe.get("removed_count", 0) or 0) > 0:
        diagnostics["cross_section_dedupe"] = cross_section_dedupe

    diagnostics["introduction"] = _section_diagnostics_entry(intro_explicit, sections["introduction"])
    diagnostics["methods"] = _section_diagnostics_entry(methods_explicit, sections["methods"])
    diagnostics["results"] = {
        **_section_diagnostics_entry(results_text_explicit, sections["results"]),
        "text_primary_count": sum(
            1
            for item in sections["results"]["items"]
            if str(item.get("result_evidence_type") or "") == "text_primary"
        ),
        "media_support_count": sum(
            1
            for item in sections["results"]["items"]
            if str(item.get("result_evidence_type") or "") == "media_support"
        ),
    }
    diagnostics["discussion"] = _section_diagnostics_entry(discussion_explicit, sections["discussion"])
    diagnostics["conclusion"] = _section_diagnostics_entry(conclusion_explicit, sections["conclusion"])
    diagnostics["fallback_notes"] = fallback_notes
    return sections, diagnostics, fallback_notes


def _section_statement_alignment_score(section: str, statement: str) -> float:
    text = str(statement or "").strip()
    if not text:
        return 0.0
    key = str(section or "").strip().lower()
    if key == "introduction":
        return 0.18 if INTRO_SECTION_RE.search(text) else 0.0
    if key == "methods":
        return 0.22 if METHODS_SECTION_RE.search(text) or METHOD_SIGNAL_RE.search(text) else 0.0
    if key == "results":
        if CONCRETE_RESULT_RE.search(text):
            return 0.24
        return 0.16 if RESULTS_SECTION_RE.search(text) else 0.0
    if key == "discussion":
        return 0.18 if DISCUSSION_SECTION_RE.search(text) else 0.0
    if key == "conclusion":
        return 0.20 if CONCLUSION_SECTION_RE.search(text) else 0.0
    return 0.0


def _cross_section_item_score(item: dict[str, Any], section: str) -> float:
    score = (1.4 * clamp_confidence(item.get("section_confidence", 0.0))) + (
        1.0 * clamp_confidence(item.get("confidence", 0.0))
    )
    statement = str(item.get("statement", "")).strip()
    score += _section_statement_alignment_score(section, statement)

    anchor = str(item.get("anchor", "")).strip()
    anchor_section = _anchor_section_label(anchor)
    if anchor_section == section:
        score += 0.35
    elif anchor_section != "unknown" and anchor_section != section:
        score -= 0.30

    source_modality = _normalize_modality_name(str(item.get("source_modality", "text")))
    if section == "results" and source_modality in {"figure", "table", "supplement"}:
        score += 0.06

    section_source = str(item.get("section_source", "") or "").strip().lower()
    if section_source in {"anchor", "explicit_heading", "heading", "structured_abstract", "meta"}:
        score += 0.12
    elif section_source in {"position", "fallback"}:
        score -= 0.10

    flags = {str(flag).strip().lower() for flag in item.get("flags", []) if str(flag).strip()}
    if "fallback" in flags:
        score -= 0.18
    if "raw_chunk" in flags:
        score -= 0.12
    if "min_coverage" in flags:
        score -= 0.10
    if bool(item.get("is_untrusted")):
        score -= 0.20

    return float(score)


def _dedupe_items_across_sections(sections: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(sections, dict):
        return sections, {"removed_count": 0, "removed_by_section": {}, "kept_by_section": {}}

    section_order = [section for section in EXEC_REPORT_SECTION_ORDER if isinstance(sections.get(section), dict)]
    entries: list[dict[str, Any]] = []
    candidate_index_by_section: dict[str, set[int]] = {section: set() for section in section_order}
    for section in section_order:
        block = sections.get(section, {})
        items = block.get("items", []) if isinstance(block, dict) else []
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            key = _canonical_statement_text(statement)
            if not key:
                continue
            candidate_index_by_section[section].add(idx)
            entries.append(
                {
                    "section": section,
                    "idx": idx,
                    "item": item,
                    "key": key,
                    "score": _cross_section_item_score(item, section),
                }
            )

    if len(entries) <= 1:
        kept_by_section = {
            section: len((sections.get(section, {}) or {}).get("items", []))
            for section in section_order
            if isinstance(sections.get(section), dict)
        }
        return sections, {"removed_count": 0, "removed_by_section": {}, "kept_by_section": kept_by_section}

    clusters: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        cluster_key = _find_near_duplicate_key(entry["key"], clusters.keys()) or entry["key"]
        clusters.setdefault(cluster_key, []).append(entry)

    keep_index_by_section: dict[str, set[int]] = {section: set() for section in section_order}
    for cluster_entries in clusters.values():
        winner = max(
            cluster_entries,
            key=lambda e: (
                float(e.get("score", 0.0)),
                -section_order.index(e["section"]) if e.get("section") in section_order else -999,
                -int(e.get("idx", 0)),
            ),
        )
        keep_index_by_section[winner["section"]].add(int(winner["idx"]))

    entries_by_section: dict[str, list[dict[str, Any]]] = {section: [] for section in section_order}
    for entry in entries:
        section = str(entry.get("section", ""))
        if section in entries_by_section:
            entries_by_section[section].append(entry)

    # Prevent over-pruning in discussion/conclusion when we already had enough candidates.
    for section, min_keep in CROSS_SECTION_DEDUPE_MIN_KEEP.items():
        if section not in section_order:
            continue
        configured_min = int(min_keep or 0)
        if configured_min <= 0:
            continue
        candidate_count = len(candidate_index_by_section.get(section, set()))
        if candidate_count < configured_min:
            continue
        keep_indices = keep_index_by_section.setdefault(section, set())
        if len(keep_indices) >= configured_min:
            continue
        ranked_section_entries = sorted(
            entries_by_section.get(section, []),
            key=lambda e: (float(e.get("score", 0.0)), -int(e.get("idx", 0))),
            reverse=True,
        )
        for entry in ranked_section_entries:
            idx = int(entry.get("idx", 0))
            if idx in keep_indices:
                continue
            keep_indices.add(idx)
            if len(keep_indices) >= configured_min:
                break

    removed_by_section: dict[str, int] = {}
    kept_by_section: dict[str, int] = {}
    for section in section_order:
        block = sections.get(section, {})
        if not isinstance(block, dict):
            continue
        items = block.get("items", [])
        if not isinstance(items, list):
            continue
        keep_indices = keep_index_by_section.get(section, set())
        kept: list[dict[str, Any]] = []
        removed_count = 0
        candidate_indices = candidate_index_by_section.get(section, set())
        for idx, item in enumerate(items):
            if idx not in candidate_indices:
                kept.append(item)
                continue
            if idx in keep_indices:
                kept.append(item)
            else:
                removed_count += 1
        block["items"] = kept
        block["evidence_refs"] = _collect_section_refs(kept, max_items=30)
        sections[section] = block
        kept_by_section[section] = len(kept)
        if removed_count > 0:
            removed_by_section[section] = removed_count

    removed_count_total = sum(removed_by_section.values())
    return sections, {
        "removed_count": int(removed_count_total),
        "removed_by_section": removed_by_section,
        "kept_by_section": kept_by_section,
    }


def _build_section_block(
    title: str,
    items: list[dict[str, Any]],
    *,
    fallback_used: bool,
    fallback_reason: str | None,
    max_items: int = 30,
) -> dict[str, Any]:
    deduped = _dedupe_section_items(items, max_items=max_items)
    refs = _collect_section_refs(deduped, max_items=30)
    return {
        "items": deduped,
        "evidence_refs": refs,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
    }


def _section_diagnostics_entry(explicit_packets: list[dict[str, Any]], section_block: dict[str, Any]) -> dict[str, Any]:
    items = section_block.get("items", []) if isinstance(section_block, dict) else []
    untrusted_count = 0
    confidence_total = 0.0
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            if bool(item.get("is_untrusted")):
                untrusted_count += 1
            try:
                confidence_total += float(item.get("section_confidence", 0.0))
            except Exception:
                continue
    avg_section_confidence = round(confidence_total / max(1, len(items)), 3) if isinstance(items, list) else 0.0
    return {
        "explicit_packet_count": len(explicit_packets),
        "final_item_count": len(items) if isinstance(items, list) else 0,
        "untrusted_item_count": untrusted_count,
        "avg_section_confidence": avg_section_confidence,
        "fallback_used": bool(section_block.get("fallback_used")) if isinstance(section_block, dict) else False,
        "fallback_reason": str(section_block.get("fallback_reason") or "") if isinstance(section_block, dict) else "",
    }


def _select_text_packets_by_section(
    text_packets: list[dict[str, Any]],
    section: str,
    *,
    min_section_confidence: float = 0.0,
    preferred_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = str(section or "").strip().lower()
    out: list[dict[str, Any]] = []
    for packet in text_packets:
        if _packet_section_label(packet) == wanted:
            statement = str(packet.get("statement", "")).strip()
            if _is_noise_statement(statement):
                continue
            confidence = clamp_confidence(packet.get("section_confidence", packet.get("confidence", 0.0)))
            if confidence < min_section_confidence:
                continue
            if preferred_sources:
                source = str(packet.get("section_source", "") or "").strip().lower()
                if source not in preferred_sources:
                    continue
            out.append(packet)
    return out


def _is_intro_fidelity_statement(statement: str) -> bool:
    text = str(statement or "").strip()
    if not text:
        return False
    if INTRO_EXCLUDE_RE.search(text):
        return False
    return True


def _is_intro_fallback_candidate(packet: dict[str, Any]) -> bool:
    statement = str(packet.get("statement", "")).strip()
    if not statement:
        return False
    if _is_noise_statement(statement):
        return False
    if _packet_section_label(packet) in {"methods", "results", "discussion", "conclusion"}:
        return False
    if _is_method_like_statement(statement):
        return False
    if INTRO_EXCLUDE_RE.search(statement):
        return False
    if not INTRO_FALLBACK_RE.search(_packet_blob(packet)):
        return False
    return True


def _intro_cutoff_index(text_packets: list[dict[str, Any]]) -> int:
    for idx, packet in enumerate(text_packets):
        if _packet_section_label(packet) in {"methods", "results"}:
            return idx
    if not text_packets:
        return 0
    return max(2, min(len(text_packets), len(text_packets) // 3 or 2))


def _is_intro_positional_candidate(packet: dict[str, Any], *, idx: int, cutoff_index: int) -> bool:
    if idx >= cutoff_index:
        return False
    statement = str(packet.get("statement", "")).strip()
    if not statement:
        return False
    if _is_noise_statement(statement):
        return False
    if _packet_section_label(packet) in {"methods", "results", "discussion", "conclusion"}:
        return False
    if _is_method_like_statement(statement):
        return False
    if _has_concrete_result_outcome(statement):
        return False
    if INTRO_EXCLUDE_RE.search(statement):
        return False
    conf = clamp_confidence(packet.get("section_confidence", packet.get("confidence", 0.0)))
    if conf < 0.45:
        return False
    return True


def _is_methods_fallback_candidate(packet: dict[str, Any]) -> bool:
    statement = str(packet.get("statement", "")).strip()
    if not statement:
        return False
    if _is_noise_statement(statement):
        return False
    section_label = _packet_section_label(packet)
    if section_label in {"introduction", "discussion", "conclusion"}:
        return False
    anchor = str(packet.get("anchor", "")).strip()
    if not (_is_method_like_statement(statement) or _has_method_signal(statement) or _is_methods_anchor(anchor)):
        return False
    if _has_concrete_result_outcome(statement) and not _is_methods_anchor(anchor):
        return False
    return True


def _is_discussion_fallback_candidate(packet: dict[str, Any]) -> bool:
    statement = str(packet.get("statement", "")).strip()
    if not statement:
        return False
    if _is_noise_statement(statement):
        return False
    if _is_layout_artifact_statement(statement):
        return False
    section_label = _packet_section_label(packet)
    if section_label in {"methods", "introduction", "conclusion"}:
        return False
    if not DISCUSSION_FALLBACK_RE.search(statement):
        return False
    # Avoid pure numeric outcomes in discussion fallback.
    if _has_concrete_result_outcome(statement) and not re.search(
        r"\b(suggest|may|consistent with|limitations?|interpret|implication)\b",
        statement,
        re.IGNORECASE,
    ):
        return False
    return True


def _is_conclusion_like_statement(statement: str) -> bool:
    text = str(statement or "").strip()
    if not text:
        return False
    if _is_noise_statement(text):
        return False
    if _is_layout_artifact_statement(text):
        return False
    has_signal = bool(CONCLUSION_FALLBACK_RE.search(text) or CONCLUSION_RECOVERY_RE.search(text))
    if not has_signal:
        return False
    if _is_method_like_statement(text) and not CONCLUSION_RECOVERY_RE.search(text):
        return False
    if _has_concrete_result_outcome(text) and not re.search(
        r"\b(suggest|support|indicat|overall|conclusion|future|implication|clinical|target|caution|may)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return True


def _is_conclusion_fallback_candidate(packet: dict[str, Any]) -> bool:
    statement = str(packet.get("statement", "")).strip()
    if not statement:
        return False
    section_label = _packet_section_label(packet)
    if section_label in {"methods", "introduction"}:
        return False
    return _is_conclusion_like_statement(statement)


def _packet_section_label(packet: dict[str, Any]) -> str:
    section_label = str(packet.get("section_label", "") or "").strip().lower()
    if section_label in SECTION_LABELS and section_label != "unknown":
        return section_label
    anchor = str(packet.get("anchor", "") or "")
    anchor_section = _anchor_section_label(anchor)
    if anchor_section != "unknown":
        return anchor_section
    category = str(packet.get("category", "") or "").strip().lower()
    if category in SECTION_LABELS and category != "unknown":
        return category
    blob = _packet_blob(packet)
    if CONCLUSION_SECTION_RE.search(blob):
        return "conclusion"
    if DISCUSSION_SECTION_RE.search(blob):
        return "discussion"
    if INTRO_SECTION_RE.search(blob):
        return "introduction"
    if RESULTS_SECTION_RE.search(blob):
        return "results"
    if METHODS_SECTION_RE.search(blob):
        return "methods"
    return "unknown"


def _anchor_section_label(anchor: str) -> str:
    value = str(anchor or "").strip()
    if not value:
        return "unknown"
    if value.lower() == "abstract":
        return "introduction"
    if not value.lower().startswith("section:"):
        return "unknown"
    parts = value.split(":")
    if len(parts) <= 2:
        section_name = parts[1] if len(parts) == 2 else ""
    else:
        section_name = ":".join(parts[1:-1])
    lowered = section_name.lower()
    if CONCLUSION_SECTION_RE.search(lowered):
        return "conclusion"
    if DISCUSSION_SECTION_RE.search(lowered):
        return "discussion"
    if INTRO_SECTION_RE.search(lowered):
        return "introduction"
    if RESULTS_SECTION_RE.search(lowered):
        return "results"
    if METHODS_SECTION_RE.search(lowered):
        return "methods"
    return "unknown"


def _section_evidence_from_method_slots(methods_compact: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for slot in methods_compact:
        if str(slot.get("status", "")) != "found":
            continue
        statement = _clean_section_statement(str(slot.get("statement", "")).strip())
        if not statement:
            continue
        refs = [str(ref).strip() for ref in slot.get("evidence_refs", []) if str(ref).strip()]
        out.append(
            {
                "statement": statement,
                "anchor": refs[0] if refs else "",
                "evidence_refs": refs[:5],
                "source_modality": "text",
                "confidence": clamp_confidence(slot.get("confidence", 0.0)),
                "section_confidence": max(0.7, clamp_confidence(slot.get("confidence", 0.0))),
                "flags": ["methods_compact"],
            }
        )
    return out


def _section_evidence_from_extracted_bullets(
    rows: list[dict[str, Any]],
    *,
    section_key: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        statement = _clean_section_statement(str(row.get("statement", "")).strip())
        if not statement:
            continue
        refs = [str(ref).strip() for ref in row.get("evidence_refs", []) if str(ref).strip()]
        if not refs:
            continue
        source_modality = _infer_modality_from_refs(refs)
        result_evidence_type = (
            "text_primary"
            if source_modality == "text" and section_key == "results"
            else "media_support"
            if section_key == "results"
            else None
        )
        out.append(
            {
                "statement": statement,
                "anchor": refs[0],
                "evidence_refs": refs[:6],
                "source_modality": source_modality,
                "section_source": "llm_section_extract",
                "confidence": 0.82,
                "section_confidence": 0.82,
                "flags": ["llm_section_extract"],
                "result_evidence_type": result_evidence_type,
            }
        )
    return out


def _section_evidence_from_packet(
    packet: dict[str, Any],
    source_modality: str,
    *,
    flags: list[str] | None = None,
    result_evidence_type: str | None = None,
    is_untrusted: bool = False,
) -> dict[str, Any]:
    refs = [str(ref).strip() for ref in packet.get("evidence_refs", []) if str(ref).strip()]
    anchor = str(packet.get("anchor", "")).strip()
    if not anchor and refs:
        anchor = refs[0]
    if anchor and anchor not in refs:
        refs = [anchor] + refs
    return {
        "statement": _clean_section_statement(str(packet.get("statement", "")).strip()),
        "anchor": anchor,
        "evidence_refs": refs[:6],
        "source_modality": _normalize_modality_name(source_modality),
        "section_source": str(packet.get("section_source", "") or "").strip().lower(),
        "confidence": clamp_confidence(packet.get("confidence", 0.0)),
        "section_confidence": clamp_confidence(packet.get("section_confidence", packet.get("confidence", 0.0))),
        "is_untrusted": bool(is_untrusted),
        "flags": list(flags or []),
        "result_evidence_type": _normalize_result_evidence_type(result_evidence_type),
    }


def _packet_refs(packet: dict[str, Any], *, max_items: int = 6) -> list[str]:
    refs = [str(ref).strip() for ref in packet.get("evidence_refs", []) if str(ref).strip()]
    anchor = str(packet.get("anchor", "")).strip()
    if anchor and anchor not in refs:
        refs = [anchor] + refs
    return refs[:max_items]


def _infer_modality_from_refs(refs: list[str]) -> str:
    for ref in refs:
        lower = str(ref or "").strip().lower()
        if not lower:
            continue
        if lower.startswith("figure:"):
            return "figure"
        if lower.startswith("table:"):
            return "table"
        if lower.startswith("supp"):
            return "supplement"
    return "text"


def _is_results_ref(ref: str, text_ref_section: dict[str, str]) -> bool:
    value = str(ref or "").strip()
    if not value:
        return False
    lower = value.lower()
    if lower.startswith("figure:") or lower.startswith("table:") or lower.startswith("supp"):
        return True
    if text_ref_section.get(value) == "results":
        return True
    if lower.startswith("section:") and "result" in lower:
        return True
    return False


def _normalize_modality_name(value: str) -> str:
    token = str(value or "").strip().lower()
    if token in {"text", "table", "figure", "supplement"}:
        return token
    if token == "supp":
        return "supplement"
    return "text"


def _dedupe_section_items(items: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        statement = _clean_section_statement(str(item.get("statement", "")).strip())
        if not statement:
            continue
        key = _canonical_statement_text(statement)
        if not key:
            continue
        refs = [str(ref).strip() for ref in item.get("evidence_refs", []) if str(ref).strip()]
        flags = [str(flag).strip() for flag in item.get("flags", []) if str(flag).strip()]
        current = {
            "statement": statement,
            "anchor": str(item.get("anchor", "")).strip(),
            "evidence_refs": refs,
            "source_modality": _normalize_modality_name(str(item.get("source_modality", "text"))),
            "section_source": str(item.get("section_source", "") or "").strip().lower(),
            "confidence": clamp_confidence(item.get("confidence", 0.0)),
            "section_confidence": clamp_confidence(item.get("section_confidence", item.get("confidence", 0.0))),
            "is_untrusted": bool(item.get("is_untrusted", False)),
            "flags": flags,
            "result_evidence_type": _normalize_result_evidence_type(item.get("result_evidence_type")),
        }
        existing = merged.get(key)
        if existing is None:
            duplicate_key = _find_near_duplicate_key(statement, merged.keys())
            if duplicate_key:
                current_conf = current["confidence"]
                existing_conf = clamp_confidence(merged[duplicate_key].get("confidence", 0.0))
                if current_conf > existing_conf:
                    existing_item = merged.pop(duplicate_key)
                    merged[key] = _merge_section_items(existing_item, current)
                else:
                    merged[duplicate_key] = _merge_section_items(merged[duplicate_key], current)
                continue
            merged[key] = current
            continue
        merged[key] = _merge_section_items(existing, current)
    out = list(merged.values())
    return out[:max_items]


def _merge_section_items(existing: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    out = dict(existing)
    out["confidence"] = max(existing.get("confidence", 0.0), current.get("confidence", 0.0))
    out["section_confidence"] = max(existing.get("section_confidence", 0.0), current.get("section_confidence", 0.0))
    out["is_untrusted"] = bool(existing.get("is_untrusted", False) and current.get("is_untrusted", False))
    out["evidence_refs"] = _unique_strings(
        list(existing.get("evidence_refs", [])) + list(current.get("evidence_refs", [])),
        max_items=8,
    )
    out["flags"] = _unique_strings(
        list(existing.get("flags", [])) + list(current.get("flags", [])),
        max_items=8,
    )
    if not out.get("anchor") and current.get("anchor"):
        out["anchor"] = current["anchor"]
    if not out.get("section_source") and current.get("section_source"):
        out["section_source"] = current["section_source"]
    if not out.get("result_evidence_type") and current.get("result_evidence_type"):
        out["result_evidence_type"] = current["result_evidence_type"]
    if len(str(current.get("statement", "")).strip()) > len(str(out.get("statement", "")).strip()):
        out["statement"] = str(current.get("statement", "")).strip()
    return out


def _find_near_duplicate_key(statement: str, existing_keys: Any) -> str | None:
    target = _canonical_statement_text(statement)
    if not target:
        return None
    for key in existing_keys:
        existing = str(key or "")
        if not existing:
            continue
        if _are_near_duplicate_statement_keys(target, existing):
            return existing
    return None


def _normalize_result_evidence_type(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if token in {"text_primary", "media_support"}:
        return token
    return None


def _collect_section_refs(items: list[dict[str, Any]], max_items: int = 30) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for item in items:
        values = []
        anchor = str(item.get("anchor", "")).strip()
        if anchor:
            values.append(anchor)
        values.extend([str(ref).strip() for ref in item.get("evidence_refs", []) if str(ref).strip()])
        for ref in values:
            key = ref.lower()
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
            if len(refs) >= max_items:
                return refs
    return refs


EXEC_REPORT_SECTION_ORDER: tuple[str, ...] = (
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
)
EXEC_REPORT_SECTION_HEADERS: dict[str, str] = {
    "introduction": "Introduction",
    "methods": "Methods",
    "results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
}
EXEC_REPORT_SECTION_LIMITS: dict[str, int] = {
    "introduction": 4,
    "methods": 6,
    "results": 9,
    "discussion": 6,
    "conclusion": 4,
}
CROSS_SECTION_DEDUPE_MIN_KEEP: dict[str, int] = {
    "discussion": 4,
    "conclusion": 3,
}
EXEC_REPORT_SECTION_SUMMARY_MAX_CHARS: dict[str, int] = {
    "introduction": 380,
    "methods": 520,
    "results": 560,
    "discussion": 420,
    "conclusion": 360,
}


def _build_extractive_evidence(
    *,
    sections_payload: dict[str, Any],
    text_chunk_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_section: dict[str, list[dict[str, Any]]] = {section: [] for section in EXEC_REPORT_SECTION_ORDER}
    anchor_index = _build_text_chunk_anchor_index(text_chunk_records)
    for section in EXEC_REPORT_SECTION_ORDER:
        block = sections_payload.get(section, {}) if isinstance(sections_payload, dict) else {}
        items = block.get("items", []) if isinstance(block, dict) else []
        if not isinstance(items, list):
            continue
        seen_keys: set[str] = set()
        seen_statement_keys: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_statement = str(item.get("statement", "")).strip()
            statement = _strip_confidence_annotations(raw_statement)
            if not statement or _is_noise_statement(statement):
                continue
            refs = [str(ref).strip() for ref in item.get("evidence_refs", []) if str(ref).strip()]
            anchor = str(item.get("anchor", "")).strip()
            if not anchor and refs:
                anchor = refs[0]
            if anchor and anchor not in refs:
                refs = [anchor] + refs
            source_modality = _normalize_modality_name(str(item.get("source_modality", "text")))
            text_anchor = _first_text_anchor(refs) or (anchor if _is_text_anchor(anchor) else "")
            source_text = anchor_index.get(text_anchor, "")
            verbatim = _best_verbatim_excerpt(statement=statement, source_text=source_text)
            repaired_statement = _repair_statement_from_verbatim(raw_statement, verbatim)
            if repaired_statement:
                statement = repaired_statement
            if _is_incomplete_statement(statement):
                continue
            span_start, span_end = _extractive_span(source_text, verbatim)
            evidence_id = f"{section}:e{len(by_section[section]) + 1}"
            dedupe_key = _canonical_text(f"{section}|{anchor}|{statement}")
            if not dedupe_key or dedupe_key in seen_keys:
                continue
            statement_key = _canonical_statement_text(statement)
            if statement_key and _is_redundant_statement_key(statement_key, seen_statement_keys):
                continue
            seen_keys.add(dedupe_key)
            if statement_key:
                seen_statement_keys.add(statement_key)
            by_section[section].append(
                {
                    "evidence_id": evidence_id,
                    "section": section,
                    "statement": statement,
                    "rephrased_statement": _light_rephrase_statement(statement),
                    "anchor": anchor,
                    "evidence_refs": refs[:6],
                    "source_modality": source_modality,
                    "verbatim_text": verbatim,
                    "span_start": span_start,
                    "span_end": span_end,
                    "confidence": clamp_confidence(item.get("confidence", 0.0)),
                    "section_confidence": clamp_confidence(item.get("section_confidence", item.get("confidence", 0.0))),
                    "is_untrusted": bool(item.get("is_untrusted", False)),
                    "flags": [str(flag).strip() for flag in item.get("flags", []) if str(flag).strip()],
                    "result_evidence_type": _normalize_result_evidence_type(item.get("result_evidence_type")),
                }
            )
    return by_section


def _build_presentation_evidence(
    *,
    extractive_evidence: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    limits = {
        "introduction": 5,
        "methods": 10,
        "results": 14,
        "discussion": 10,
        "conclusion": 6,
    }
    anchor_caps = {
        "introduction": 2,
        "methods": 2,
        "results": 2,
        "discussion": 2,
        "conclusion": 2,
    }
    out: dict[str, list[dict[str, Any]]] = {section: [] for section in EXEC_REPORT_SECTION_ORDER}
    for section in EXEC_REPORT_SECTION_ORDER:
        rows = extractive_evidence.get(section, []) if isinstance(extractive_evidence, dict) else []
        ranked = sorted(
            [row for row in rows if isinstance(row, dict)],
            key=_extractive_rank_score,
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        seen_statement_keys: set[str] = set()
        anchor_counts: dict[str, int] = {}
        for row in ranked:
            statement = _clean_section_statement(str(row.get("statement", "")).strip())
            if not statement or _is_noise_statement(statement):
                continue
            anchor = str(row.get("anchor", "")).strip()
            if not _section_item_passes_fidelity(section, {"statement": statement, "anchor": anchor}):
                continue
            statement_key = _canonical_statement_text(statement)
            if statement_key and _is_redundant_statement_key(statement_key, seen_statement_keys):
                continue
            if anchor:
                current_anchor_count = int(anchor_counts.get(anchor, 0) or 0)
                if current_anchor_count >= int(anchor_caps.get(section, 2)):
                    continue
            if statement_key:
                seen_statement_keys.add(statement_key)
            if anchor:
                anchor_counts[anchor] = int(anchor_counts.get(anchor, 0) or 0) + 1
            selected.append(
                {
                    "statement": statement,
                    "anchor": anchor,
                    "evidence_refs": [str(ref).strip() for ref in row.get("evidence_refs", []) if str(ref).strip()][:6],
                    "confidence": clamp_confidence(row.get("confidence", 0.0)),
                    "section_confidence": clamp_confidence(row.get("section_confidence", row.get("confidence", 0.0))),
                    "source_modality": _normalize_modality_name(str(row.get("source_modality", "text"))),
                    "is_untrusted": bool(row.get("is_untrusted", False)),
                    "flags": [str(flag).strip() for flag in row.get("flags", []) if str(flag).strip()],
                }
            )
            if len(selected) >= int(limits.get(section, 8)):
                break
        out[section] = selected
    return out


def _build_text_chunk_anchor_index(text_chunk_records: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for record in text_chunk_records:
        if not isinstance(record, dict):
            continue
        anchor = str(record.get("anchor", "")).strip()
        if not anchor or anchor in out:
            continue
        content = " ".join(str(record.get("content", "")).split()).strip()
        if not content:
            continue
        out[anchor] = content
    return out


def _first_text_anchor(refs: list[str]) -> str:
    for ref in refs:
        if _is_text_anchor(ref):
            return ref
    return ""


def _is_text_anchor(ref: str) -> bool:
    lower = str(ref or "").strip().lower()
    if not lower:
        return False
    if lower.startswith("figure:") or lower.startswith("table:") or lower.startswith("supp"):
        return False
    return True


def _best_verbatim_excerpt(*, statement: str, source_text: str) -> str:
    text = " ".join(str(source_text or "").split()).strip()
    if not text:
        return ""
    target = " ".join(str(statement or "").split()).strip()
    if target:
        match_idx = text.lower().find(target.lower())
        if match_idx >= 0:
            return _summary_fragment(text[match_idx : match_idx + max(280, len(target) + 80)], max_chars=460)
    sentences = _split_sentences(text)
    if not sentences:
        return _summary_fragment(text, max_chars=460)
    target_tokens = set(re.findall(r"[a-z0-9]+", target.lower()))
    best_sentence = ""
    best_score = -1.0
    for sentence in sentences:
        candidate = " ".join(sentence.split()).strip()
        if not candidate or _is_noise_statement(candidate):
            continue
        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))
        if not candidate_tokens:
            continue
        lexical_overlap = (
            (len(candidate_tokens & target_tokens) / max(1, len(target_tokens)))
            if target_tokens
            else 0.0
        )
        score = lexical_overlap + min(0.15, len(candidate_tokens) / 120.0)
        if score > best_score:
            best_score = score
            best_sentence = candidate
    if not best_sentence and sentences:
        best_sentence = " ".join(str(sentences[0]).split()).strip()
    return _summary_fragment(best_sentence, max_chars=460)


def _extractive_span(source_text: str, excerpt: str) -> tuple[int | None, int | None]:
    text = " ".join(str(source_text or "").split()).strip()
    value = " ".join(str(excerpt or "").split()).strip()
    if not text or not value:
        return None, None
    idx = text.lower().find(value.lower())
    if idx < 0:
        return None, None
    return idx, idx + len(value)


def _light_rephrase_statement(statement: str, *, max_chars: int = 260) -> str:
    text = " ".join(_strip_confidence_annotations(str(statement or "")).split()).strip()
    if not text:
        return ""
    for pattern in (
        r"^in (?:the )?present study,\s*",
        r"^in our study,\s*",
        r"^the authors (?:found|report(?:ed)?|identified|showed)\s+that\s*",
        r"^we found that\s*",
    ):
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = text[:1].upper() + text[1:] if text else text
    return _summary_fragment(text, max_chars=max_chars)


def _extractive_rank_score(entry: dict[str, Any]) -> float:
    score = 0.0
    score += float(clamp_confidence(entry.get("confidence", 0.0))) * 1.2
    score += float(clamp_confidence(entry.get("section_confidence", 0.0))) * 1.1
    if str(entry.get("source_modality", "")).strip().lower() == "text":
        score += 0.25
    if str(entry.get("verbatim_text", "")).strip():
        score += 0.2
    if bool(entry.get("is_untrusted", False)):
        score -= 0.45
    result_evidence_type = str(entry.get("result_evidence_type") or "").strip().lower()
    if result_evidence_type == "text_primary":
        score += 0.35
    elif result_evidence_type == "media_support":
        score -= 0.05
    flags = {str(flag).strip().lower() for flag in entry.get("flags", [])}
    if "fallback" in flags:
        score -= 0.2
    if "raw_chunk" in flags:
        score -= 0.15
    return score


def _build_executive_report(
    *,
    extractive_evidence: dict[str, list[dict[str, Any]]],
    fallback_summary: str,
) -> dict[str, Any]:
    section_rows: list[dict[str, Any]] = []
    overview_parts: list[str] = []
    total_evidence_items = 0
    for section in EXEC_REPORT_SECTION_ORDER:
        rows = extractive_evidence.get(section, []) if isinstance(extractive_evidence, dict) else []
        ranked = sorted(
            [row for row in rows if isinstance(row, dict)],
            key=_extractive_rank_score,
            reverse=True,
        )
        limit = max(1, int(EXEC_REPORT_SECTION_LIMITS.get(section, 4)))
        selected = ranked[:limit]
        bullets: list[dict[str, Any]] = []
        for row in selected:
            rephrased = _light_rephrase_statement(str(row.get("rephrased_statement") or row.get("statement") or ""))
            if not rephrased:
                rephrased = _light_rephrase_statement(str(row.get("verbatim_text", "")))
            if not rephrased:
                continue
            evidence_id = str(row.get("evidence_id", "")).strip()
            bullets.append(
                {
                    "text": rephrased,
                    "evidence_ids": [evidence_id] if evidence_id else [],
                    "anchors": [str(row.get("anchor", "")).strip()] if str(row.get("anchor", "")).strip() else [],
                }
            )
        summary_text = _summary_fragment(
            "; ".join(str(bullet.get("text", "")).strip() for bullet in bullets if str(bullet.get("text", "")).strip()),
            max_chars=EXEC_REPORT_SECTION_SUMMARY_MAX_CHARS.get(section, 380),
        )
        section_rows.append(
            {
                "section": section,
                "summary": summary_text,
                "bullets": bullets,
            }
        )
        if summary_text:
            header = EXEC_REPORT_SECTION_HEADERS.get(section, section.title())
            overview_parts.append(f"{header}: {summary_text}.")
        total_evidence_items += len(bullets)

    overview = " ".join(overview_parts).strip()
    if not overview:
        overview = _summary_fragment(str(fallback_summary or "").strip(), max_chars=2200)

    return {
        "style": "succinct_grounded_v1",
        "overview": overview,
        "sections": section_rows,
        "total_evidence_items": total_evidence_items,
    }


def _filter_result_text_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for packet in packets:
        statement = _clean_section_statement(str(packet.get("statement", "")).strip())
        if not statement:
            continue
        if _is_noise_statement(statement):
            continue
        if _is_visual_annotation_statement(statement):
            continue
        has_concrete = _has_concrete_result_outcome(statement)
        has_high_signal = _is_high_signal_result_statement(statement)
        if _is_method_like_statement(statement) and not has_concrete:
            continue
        if _is_generic_visual_statement(statement) and not has_concrete:
            continue
        if not has_concrete and not has_high_signal:
            continue
        out.append({**packet, "statement": statement})
    return out


def _filter_result_media_packets(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for packet in packets:
        statement = str(packet.get("statement", "")).strip()
        if not statement:
            continue
        if not _has_concrete_result_outcome(statement):
            continue
        if _is_method_like_statement(statement):
            continue
        out.append(packet)
    return out


def _is_method_like_statement(statement: str) -> bool:
    return bool(METHOD_LIKE_RE.search(str(statement or "")))


def _has_concrete_result_outcome(statement: str) -> bool:
    return bool(CONCRETE_RESULT_RE.search(str(statement or "")))


def _is_high_signal_result_statement(statement: str) -> bool:
    return bool(HIGH_SIGNAL_RESULT_RE.search(str(statement or "")))


def _is_generic_visual_statement(statement: str) -> bool:
    text = str(statement or "").strip()
    if not text:
        return False
    if not GENERIC_VISUAL_RE.search(text):
        return False
    return not _has_concrete_result_outcome(text)


def _has_panel_prefix(statement: str) -> bool:
    return bool(re.match(r"^\s*[a-h](?:\s*(?:and|,)\s*[a-h])?,\s+", str(statement or ""), flags=re.IGNORECASE))


def _is_visual_annotation_statement(statement: str) -> bool:
    text = str(statement or "").strip()
    if not text:
        return False
    if _has_panel_prefix(text):
        return True
    if VISUAL_ANNOTATION_RE.search(text):
        return True
    return False


def _is_layout_artifact_statement(value: str) -> bool:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return False
    lowered = text.lower()
    if LAYOUT_ARTIFACT_RE.search(text):
        return True
    direction_hits = re.findall(r"\b(left|right|top|bottom)\b", lowered)
    if len(direction_hits) >= 2 and re.search(
        r"\b(panel|projection|surface|hemisphere|view|row|column)\b",
        lowered,
    ):
        return True
    if re.search(r"\bfig(?:ure)?\.?\s*\d+[a-z]?\b", lowered) and re.search(
        r"\b(left|right|top|bottom|panel)\b",
        lowered,
    ):
        return True
    return False


def _is_noise_statement(value: str) -> bool:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return True
    if text.startswith("$^{"):
        return True
    if _is_layout_artifact_statement(text):
        return True
    if NOISE_STATEMENT_RE.search(text):
        return True
    if re.search(r"\bAm\s+J\s+Psychiatry\b", text):
        return True
    if re.search(r"\b(Drs?\.|Department of|Perelman School|Address correspondence)\b", text):
        return True
    if re.match(r"^(table|figure)\s+\d+\b", text, re.IGNORECASE):
        return True
    if re.fullmatch(r"\d{1,4}", text):
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if upper_ratio > 0.85 and len(text) >= 16:
            return True
    if re.search(r"(M\.D\.|Ph\.D\.|B\.A\.|B\.S\.)", text) and (text.count(",") >= 2 or len(text.split()) <= 18):
        return True
    return False


def _coverage_block_gaps(coverage_block: dict[str, Any], label: str) -> list[str]:
    if not isinstance(coverage_block, dict):
        return []
    missing = coverage_block.get("missing_refs", []) or []
    if not missing:
        return []
    return [f"Missing {label} refs: {', '.join(str(item) for item in missing)}"]


def _coverage_gaps(coverage: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    gaps += _coverage_block_gaps(coverage.get("figures", {}), "figure")
    gaps += _coverage_block_gaps(coverage.get("tables", {}), "table")
    gaps += _coverage_block_gaps(coverage.get("supp_figures", {}), "supplement figure")
    gaps += _coverage_block_gaps(coverage.get("supp_tables", {}), "supplement table")
    return gaps


def _section_packet_counts(text_packets: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for packet in text_packets:
        section = _packet_section_label(packet)
        source = str(packet.get("section_source", "fallback") or "fallback").strip().lower()
        section_block = out.setdefault(section, {"count": 0, "sources": {}})
        section_block["count"] += 1
        section_block["sources"][source] = int(section_block["sources"].get(source, 0) or 0) + 1
    return out


def _section_conflict_count(text_packets: list[dict[str, Any]]) -> int:
    count = 0
    for packet in text_packets:
        flags = {str(flag).strip().lower() for flag in packet.get("quality_flags", [])}
        if "section_conflict_resolved" in flags:
            count += 1
    return count


def _unknown_section_count(text_packets: list[dict[str, Any]]) -> int:
    return sum(1 for packet in text_packets if _packet_section_label(packet) == "unknown")


def _slot_fill_rates(sections_compact: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for section_key, rows in sections_compact.items():
        total = len(rows) if isinstance(rows, list) else 0
        found = 0
        if isinstance(rows, list):
            found = sum(1 for row in rows if str(row.get("status", "")).lower() == "found")
        out[section_key] = {
            "found": found,
            "total": total,
            "rate": (found / total) if total else 0.0,
        }
    return out


def _cross_section_rejections(text_packets: list[dict[str, Any]]) -> int:
    rejections = 0
    for packet in text_packets:
        anchor_label = _anchor_section_label(str(packet.get("anchor", "")))
        packet_label = _packet_section_label(packet)
        if anchor_label == "unknown" or packet_label == "unknown":
            continue
        if anchor_label != packet_label:
            rejections += 1
    return rejections


def _coerce_narrative_overrides(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    return {
        "executive_summary": _strip_confidence_annotations(str(parsed.get("executive_summary", "")).strip()),
        "methods_strengths": _as_str_list(parsed.get("methods_strengths")),
        "methods_weaknesses": _as_str_list(parsed.get("methods_weaknesses")),
        "reproducibility_ethics": _as_str_list(parsed.get("reproducibility_ethics")),
        "uncertainty_gaps": _as_str_list(parsed.get("uncertainty_gaps")),
    }


def _apply_narrative_overrides(draft: dict[str, Any], overrides: dict[str, Any]) -> None:
    executive_summary = str(overrides.get("executive_summary") or "").strip()
    methods_strengths = _as_str_list(overrides.get("methods_strengths"))
    methods_weaknesses = _as_str_list(overrides.get("methods_weaknesses"))
    reproducibility_ethics = _as_str_list(overrides.get("reproducibility_ethics"))
    uncertainty_gaps = _as_str_list(overrides.get("uncertainty_gaps"))
    if executive_summary:
        draft["executive_summary"] = executive_summary
    if methods_strengths:
        draft["methods_strengths"] = methods_strengths
    if methods_weaknesses:
        draft["methods_weaknesses"] = methods_weaknesses
    if reproducibility_ethics:
        draft["reproducibility_ethics"] = reproducibility_ethics
    if uncertainty_gaps:
        draft["uncertainty_gaps"] = uncertainty_gaps


def _ensure_executive_summary_components(draft: dict[str, Any], payload: dict[str, Any]) -> None:
    _trace_synthesis_step("ensure_summary:start")
    sections_extracted = draft.get("sections_extracted")
    if not isinstance(sections_extracted, dict):
        sections_extracted = {}
    if settings.analysis_section_extraction_enabled and not sections_extracted:
        _trace_synthesis_step("ensure_summary:section_extract:start")
        sections_extracted = _llm_section_extraction(payload)
        _trace_synthesis_step("ensure_summary:section_extract:done")
        if sections_extracted:
            draft["sections_extracted_version"] = 1
            draft["sections_extracted"] = sections_extracted
            _trace_synthesis_step("ensure_summary:section_extract:stored")

    sections_compact = draft.get("sections_compact")
    if not isinstance(sections_compact, dict):
        _trace_synthesis_step("ensure_summary:build_sections_compact")
        methods_compact = _methods_compact(
            list(payload.get("text_packets", [])),
            analysis_notes=_as_str_list(payload.get("analysis_notes")),
        )
        sections_compact = _sections_compact(
            text_packets=list(payload.get("text_packets", [])),
            methods_compact=methods_compact,
            analysis_notes=_as_str_list(payload.get("analysis_notes")),
            text_chunk_records=list(payload.get("text_chunk_records", [])),
            result_support_packets=list(payload.get("table_packets", []))
            + list(payload.get("figure_packets", []))
            + list(payload.get("supp_packets", [])),
            sections_extracted=sections_extracted,
        )
        sections_compact, _ = _dedupe_sections_compact_rows(
            sections_compact,
            access_limited=any(
                ACCESS_LIMITED_NOTE_RE.search(str(note or ""))
                for note in _as_str_list(payload.get("analysis_notes"))
            ),
        )
        draft["sections_compact"] = sections_compact
        draft["sections_compact_version"] = SECTIONS_COMPACT_VERSION
        _trace_synthesis_step("ensure_summary:build_sections_compact_done")
    _trace_synthesis_step("ensure_summary:build_exec:start")
    rebuilt = _build_executive_summary(
        sections_compact=sections_compact,
        existing_summary=str(draft.get("executive_summary", "")),
        sections_extracted=sections_extracted,
    )
    _trace_synthesis_step("ensure_summary:build_exec:done")
    if settings.analysis_summary_polish_enabled:
        _trace_synthesis_step("ensure_summary:polish:start")
        rebuilt = _constrained_summary_polish(rebuilt, sections_compact)
        _trace_synthesis_step("ensure_summary:polish:done")
    if rebuilt:
        draft["executive_summary"] = rebuilt
    _trace_synthesis_step("ensure_summary:done")


def _build_executive_summary(
    *,
    sections_compact: dict[str, list[dict[str, Any]]],
    existing_summary: str,
    sections_extracted: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    clean_existing = _strip_confidence_annotations(" ".join(str(existing_summary or "").split()).strip())
    if clean_existing and _summary_has_all_components(clean_existing):
        return clean_existing
    extracted = sections_extracted or {}
    intro_text = _summary_text_from_compact_section(
        sections_compact.get("introduction", []),
        default_text="the study motivation and objective were only partially extracted from available text",
        max_points=4,
        max_chars=420,
        extracted_rows=extracted.get("introduction", []),
    )
    methods_text = _summary_text_from_compact_section(
        sections_compact.get("methods", []),
        default_text="method details were extracted incompletely, so design interpretation should remain cautious",
        max_points=6,
        max_chars=620,
        prioritized_slot_keys=[
            "study_design",
            "sample_population",
            "outcomes_measures",
            "statistical_model",
            "covariates",
            "missing_data_and_sensitivity",
            "reproducibility_transparency",
        ],
        extracted_rows=extracted.get("methods", []),
    )
    results_text = _summary_text_from_compact_section(
        sections_compact.get("results", []),
        default_text="key quantitative outcomes were limited in the extracted packets",
        max_points=7,
        max_chars=680,
        extracted_rows=extracted.get("results", []),
    )
    discussion_text = _summary_text_from_compact_section(
        sections_compact.get("discussion", []),
        default_text="discussion and limitations detail was limited in extracted text",
        max_points=5,
        max_chars=520,
        extracted_rows=extracted.get("discussion", []),
    )
    conclusion_text = _summary_text_from_compact_section(
        sections_compact.get("conclusion", []),
        default_text="the paper's overall conclusion was only partially explicit in extracted content",
        max_points=5,
        max_chars=520,
        extracted_rows=extracted.get("conclusion", []),
    )
    parts = [
        f"Introduction: {_summary_fragment(intro_text, max_chars=420)}.",
        f"Methods: {_summary_fragment(methods_text, max_chars=620)}.",
        f"Results: {_summary_fragment(results_text, max_chars=680)}.",
        f"Discussion: {_summary_fragment(discussion_text, max_chars=520)}.",
        f"Conclusion: {_summary_fragment(conclusion_text, max_chars=520)}.",
    ]
    return " ".join(parts).strip()


def _summary_text_from_compact_section(
    rows: list[dict[str, Any]],
    *,
    default_text: str,
    max_points: int = 2,
    max_chars: int = 240,
    prioritized_slot_keys: list[str] | None = None,
    extracted_rows: list[dict[str, Any]] | None = None,
) -> str:
    extracted = _summary_text_from_extracted_rows(extracted_rows or [], max_points=max_points, max_chars=max_chars)
    if extracted:
        return extracted

    found_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status", "")).strip().lower() != "found":
            continue
        statement = _clean_section_statement(str(row.get("statement", "")).strip())
        if not statement:
            continue
        found_rows.append({**row, "statement": statement})
    if not found_rows:
        return _strip_confidence_annotations(default_text)

    if prioritized_slot_keys:
        rank = {slot: idx for idx, slot in enumerate(prioritized_slot_keys)}
        found_rows.sort(key=lambda row: rank.get(str(row.get("slot_key", "")), len(rank)))

    selected: list[str] = []
    selected_keys: set[str] = set()
    for row in found_rows:
        statement = str(row.get("statement", "")).strip()
        key = _canonical_text(statement)
        if not key or _is_redundant_statement_key(key, selected_keys):
            continue
        selected.append(statement)
        selected_keys.add(key)
        if len(selected) >= max(1, int(max_points)):
            break
    if not selected:
        selected = [str(found_rows[0].get("statement", "")).strip()]
    return _summary_fragment("; ".join(selected), max_chars=max_chars).rstrip(".")


def _summary_text_from_extracted_rows(
    rows: list[dict[str, Any]],
    *,
    max_points: int,
    max_chars: int,
) -> str:
    if not rows:
        return ""
    selected: list[str] = []
    selected_keys: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        statement = _clean_section_statement(str(row.get("statement", "")).strip())
        if not statement:
            continue
        key = _canonical_text(statement)
        if not key or _is_redundant_statement_key(key, selected_keys):
            continue
        selected.append(statement)
        selected_keys.add(key)
        if len(selected) >= max(1, int(max_points)):
            break
    if not selected:
        return ""
    return _summary_fragment("; ".join(selected), max_chars=max_chars).rstrip(".")


def _constrained_summary_polish(summary: str, sections_compact: dict[str, list[dict[str, Any]]]) -> str:
    original = " ".join(str(summary or "").split()).strip()
    if not original:
        return ""
    candidate = original
    sections = _summary_headed_sections(candidate)
    if sections:
        normalized_parts = []
        for heading in ("introduction", "methods", "results", "discussion", "conclusion"):
            text = sections.get(heading, "")
            normalized_parts.append(f"{heading.capitalize()}: {_summary_fragment(text, max_chars=320)}.")
        candidate = " ".join(normalized_parts).strip()
    if _summary_polish_valid(original, candidate):
        return candidate
    return original


def _summary_headed_sections(summary: str) -> dict[str, str]:
    text = str(summary or "").strip()
    if not text:
        return {}
    pattern = re.compile(
        r"(Introduction|Methods|Results|Discussion|Conclusion)\s*:\s*(.*?)(?=(?:Introduction|Methods|Results|Discussion|Conclusion)\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    out: dict[str, str] = {}
    for match in pattern.finditer(text):
        key = str(match.group(1) or "").strip().lower()
        value = str(match.group(2) or "").strip(" .")
        if key and value:
            out[key] = value
    return out


def _summary_polish_valid(original: str, candidate: str) -> bool:
    original_sections = _summary_headed_sections(original)
    candidate_sections = _summary_headed_sections(candidate)
    required = {"introduction", "methods", "results", "discussion", "conclusion"}
    if set(candidate_sections.keys()) != required:
        return False
    if not required.issubset(set(original_sections.keys())):
        return False
    original_numbers = set(re.findall(r"\d+(?:\.\d+)?", original))
    candidate_numbers = set(re.findall(r"\d+(?:\.\d+)?", candidate))
    if not candidate_numbers.issubset(original_numbers):
        return False
    for key in required:
        source_tokens = set(re.findall(r"[a-z0-9]+", original_sections.get(key, "").lower()))
        polished_tokens = set(re.findall(r"[a-z0-9]+", candidate_sections.get(key, "").lower()))
        if not polished_tokens:
            return False
        overlap = len(source_tokens & polished_tokens) / max(1, len(polished_tokens))
        if overlap < 0.45:
            return False
    return True


def _packet_blob(packet: dict[str, Any]) -> str:
    anchor = str(packet.get("anchor", "")).strip()
    category = str(packet.get("category", "")).strip()
    statement = str(packet.get("statement", "")).strip()
    return f"{anchor} {category} {statement}".lower()


def _intro_compact_rows_from_chunks(
    *,
    slot_specs: list[dict[str, Any]],
    text_chunk_records: list[dict[str, Any]],
    access_limited: bool,
) -> list[dict[str, Any]]:
    if not slot_specs:
        return []
    candidates = _intro_chunk_candidates(text_chunk_records, max_items=len(slot_specs))
    if not candidates:
        return []
    rows: list[dict[str, Any]] = []
    for idx, slot_spec in enumerate(slot_specs):
        slot_key = str(slot_spec.get("slot_key", ""))
        slot_label = str(slot_spec.get("label", slot_key))
        if idx < len(candidates):
            candidate = candidates[idx]
            anchor = str(candidate.get("anchor", "")).strip()
            rows.append(
                {
                    "section_key": "introduction",
                    "slot_key": slot_key,
                    "label": slot_label,
                    "statement": _compact_section_statement(str(candidate.get("statement", ""))),
                    "status": "found",
                    "evidence_refs": [anchor] if anchor else [],
                    "confidence": 0.58,
                }
            )
            continue
        rows.append(
            {
                "section_key": "introduction",
                "slot_key": slot_key,
                "label": slot_label,
                "statement": SECTION_ACCESS_LIMITED_STATEMENT if access_limited else SECTION_NOT_FOUND_STATEMENT,
                "status": "access_limited" if access_limited else "not_found",
                "evidence_refs": [],
                "confidence": 0.0,
            }
        )
    return rows


def _intro_chunk_candidates(text_chunk_records: list[dict[str, Any]], *, max_items: int = 8) -> list[dict[str, str]]:
    if not text_chunk_records:
        return []
    ordered = sorted(
        [(idx, chunk) for idx, chunk in enumerate(text_chunk_records) if isinstance(chunk, dict)],
        key=lambda pair: _chunk_order_key(pair[1], pair[0]),
    )
    if not ordered:
        return []

    cutoff_idx = len(ordered)
    for pos, (_idx, chunk) in enumerate(ordered):
        anchor = str(chunk.get("anchor", "")).strip()
        section_label = _anchor_section_label(anchor)
        anchor_blob = anchor.lower()
        if section_label in {"methods", "results"} or METHODS_SECTION_RE.search(anchor_blob) or RESULTS_SECTION_RE.search(anchor_blob):
            cutoff_idx = pos
            break
    if cutoff_idx == len(ordered):
        cutoff_idx = max(6, min(len(ordered), len(ordered) // 3 or 6))

    prefix = ordered[: cutoff_idx if cutoff_idx > 0 else min(len(ordered), 6)]
    if not prefix:
        prefix = ordered[: min(len(ordered), 6)]

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _idx, chunk in prefix:
        anchor = str(chunk.get("anchor", "")).strip()
        section_label = _anchor_section_label(anchor)
        if section_label in {"methods", "results", "discussion", "conclusion"}:
            continue
        content = " ".join(str(chunk.get("content", "")).split()).strip()
        if not content:
            continue
        for sentence_idx, sentence in enumerate(_split_sentences(content)):
            statement = " ".join(sentence.split()).strip()
            if len(statement) < 70:
                continue
            if _is_noise_statement(statement):
                continue
            if _is_method_like_statement(statement):
                continue
            if _has_concrete_result_outcome(statement):
                continue
            if INTRO_EXCLUDE_RE.search(statement):
                continue
            key = _canonical_text(statement)
            if not key or key in seen:
                continue
            seen.add(key)
            score = 0
            if INTRO_FALLBACK_RE.search(statement):
                score += 3
            if INTRO_SECTION_RE.search(statement):
                score += 2
            if re.search(r"\b(we (?:examined|investigated|tested)|objective|aim|hypothesis|background)\b", statement, re.IGNORECASE):
                score += 2
            score += max(0, 2 - sentence_idx)
            candidates.append({"statement": statement, "anchor": anchor, "score": score})

    if not candidates:
        for _idx, chunk in prefix:
            anchor = str(chunk.get("anchor", "")).strip()
            content = " ".join(str(chunk.get("content", "")).split()).strip()
            if not content:
                continue
            for sentence in _split_sentences(content):
                statement = " ".join(sentence.split()).strip()
                if len(statement) < 70:
                    continue
                if _is_noise_statement(statement):
                    continue
                if _is_method_like_statement(statement) or _has_concrete_result_outcome(statement):
                    continue
                key = _canonical_text(statement)
                if not key or key in seen:
                    continue
                seen.add(key)
                candidates.append({"statement": statement, "anchor": anchor, "score": 1})
                break

    candidates.sort(key=lambda item: (-int(item.get("score", 0)), len(str(item.get("statement", "")))))
    return [
        {
            "statement": _compact_section_statement(str(item.get("statement", "")), max_chars=220),
            "anchor": str(item.get("anchor", "")).strip(),
        }
        for item in candidates[:max_items]
    ]


def _raw_chunk_section_candidates(
    text_chunk_records: list[dict[str, Any]],
    *,
    section: str,
    max_items: int = 10,
) -> list[dict[str, str]]:
    wanted = str(section or "").strip().lower()
    if wanted not in {"introduction", "methods", "results", "discussion", "conclusion"}:
        return []
    if not text_chunk_records:
        return []
    ordered = sorted(
        [(idx, chunk) for idx, chunk in enumerate(text_chunk_records) if isinstance(chunk, dict)],
        key=lambda pair: _chunk_order_key(pair[1], pair[0]),
    )
    if not ordered:
        return []

    total = len(ordered)
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pos_idx, (_idx, chunk) in enumerate(ordered):
        anchor = str(chunk.get("anchor", "")).strip()
        content = " ".join(str(chunk.get("content", "")).split()).strip()
        if not content:
            continue
        position = float(pos_idx) / float(max(1, total - 1)) if total > 1 else 0.0
        for sentence in _split_sentences(content):
            statement = " ".join(sentence.split()).strip()
            if len(statement) < 36:
                continue
            if _is_noise_statement(statement):
                continue
            key = _canonical_text(statement)
            if not key or key in seen:
                continue
            score = _raw_chunk_section_score(wanted, statement, position=position)
            if score <= 0.0:
                continue
            seen.add(key)
            ranked.append({"statement": statement, "anchor": anchor, "score": score})

    ranked.sort(key=lambda item: (-float(item.get("score", 0.0)), len(str(item.get("statement", "")))))
    return [
        {
            "statement": _compact_section_statement(str(item.get("statement", "")), max_chars=220),
            "anchor": str(item.get("anchor", "")).strip(),
        }
        for item in ranked[:max_items]
    ]


def _raw_chunk_section_score(section: str, statement: str, *, position: float) -> float:
    text = str(statement or "").strip()
    if not text:
        return 0.0
    lowered = text.lower()
    score = 0.0
    if section == "introduction":
        if INTRO_FALLBACK_RE.search(text):
            score += 2.2
        if _is_method_like_statement(text) or _has_concrete_result_outcome(text):
            score -= 1.1
        score += max(0.0, 0.8 - position)
    elif section == "methods":
        if METHODS_SECTION_RE.search(text) or _is_method_like_statement(text):
            score += 2.4
        if re.search(
            r"\b(participants?|sample|scanner|acquisition|protocol|regression|covariates?|inclusion|exclusion|quality assurance|mdmr)\b",
            lowered,
        ):
            score += 1.7
        if _has_concrete_result_outcome(text):
            score -= 1.2
        if position <= 0.72:
            score += 0.8
    elif section == "results":
        if _has_concrete_result_outcome(text):
            score += 2.6
        if re.search(r"\b(t\s*=|p\s*[<=>]|increased|decreased|hyperconnectivity|dysconnectivity|associated with)\b", lowered):
            score += 1.8
        if _is_method_like_statement(text):
            score -= 1.3
        if 0.25 <= position <= 0.9:
            score += 0.8
    elif section == "discussion":
        if DISCUSSION_FALLBACK_RE.search(text) or DISCUSSION_SECTION_RE.search(text):
            score += 2.5
        if re.search(r"\b(to date|in our study|in contrast|consistent with|may reflect|limitations?)\b", lowered):
            score += 1.6
        if _is_method_like_statement(text):
            score -= 1.0
        if position >= 0.52:
            score += 0.8
    elif section == "conclusion":
        if CONCLUSION_FALLBACK_RE.search(text) or CONCLUSION_SECTION_RE.search(text):
            score += 2.6
        if CONCLUSION_RECOVERY_RE.search(text):
            score += 1.2
        if re.search(r"\b(overall|these results|future research|longitudinal|target)\b", lowered):
            score += 1.5
        if position >= 0.7:
            score += 0.8
    return score


def _chunk_order_key(chunk: dict[str, Any], idx: int) -> tuple[int, int]:
    anchor = str(chunk.get("anchor", "")).strip()
    match = re.search(r":(\d+)$", anchor)
    if match:
        try:
            return (0, int(match.group(1)))
        except Exception:
            return (1, idx)
    return (1, idx)


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]


def _tail_token(value: str) -> str:
    token = str(value or "").strip().lower()
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", token)


def _trim_dangling_tail_tokens(text: str) -> str:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return ""
    clean = clean.rstrip(" ,;:/-")
    parts = clean.split()
    removed = False
    while parts:
        if _tail_token(parts[-1]) not in INCOMPLETE_TAIL_TOKENS:
            break
        parts.pop()
        removed = True
    trimmed = " ".join(parts).strip().rstrip(" ,;:/-")
    if removed and trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


def _is_incomplete_statement(value: str) -> bool:
    raw = " ".join(str(value or "").split()).strip()
    if not raw:
        return True
    if raw.endswith("...") or raw.endswith("…"):
        return True
    clean = _strip_confidence_annotations(raw)
    if not clean:
        return True
    if clean[-1] in ",;:/-([{":
        return True
    if clean.count("(") > clean.count(")"):
        return True
    if re.match(r"^\s*(?:our|this|these)\s+finding(?:s)?\s+that\b", clean, flags=re.IGNORECASE):
        if not re.search(
            r"\b(suggest|indicat|support|demonstrat|highlight|imply|point(?:s)?\s+to|show(?:ed|s)?)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            return True
    tail = _tail_token(clean.split()[-1])
    if tail in INCOMPLETE_TAIL_TOKENS:
        return True
    if (
        tail.isalpha()
        and len(tail) <= 3
        and tail not in ALLOWED_SHORT_FINAL_TOKENS
        and len(clean) >= 110
        and clean[-1] not in ".!?"
    ):
        return True
    return False


def _repair_statement_from_verbatim(statement: str, verbatim: str) -> str:
    raw = " ".join(str(statement or "").split()).strip()
    clean = _strip_confidence_annotations(raw)
    if not clean:
        return ""
    needs_repair = _is_incomplete_statement(raw) or _is_incomplete_statement(clean)
    if not needs_repair:
        return clean

    source = " ".join(_strip_confidence_annotations(verbatim).split()).strip()
    if source:
        sentences = _split_sentences(source)
        if sentences:
            first_sentence = _trim_dangling_tail_tokens(sentences[0])
            if first_sentence and not _is_incomplete_statement(first_sentence):
                return first_sentence
        if source.lower().startswith(clean.lower()) and len(source) > len(clean) + 8:
            expanded = _trim_dangling_tail_tokens(source[:420])
            if expanded and not _is_incomplete_statement(expanded):
                return expanded

    return _trim_dangling_tail_tokens(clean)


def _compact_without_cutoff(text: str, *, max_chars: int) -> str:
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return ""
    if len(clean) <= max_chars:
        return clean

    # Prefer complete sentences that fit the configured limit.
    sentences = _split_sentences(clean)
    if len(sentences) > 1:
        picked: list[str] = []
        for sentence in sentences:
            candidate = " ".join(picked + [sentence]).strip()
            if len(candidate) > max_chars:
                break
            picked.append(sentence)
        if picked:
            return " ".join(picked).strip()

    # If no full sentence fits, try clause-boundary trim without ellipsis.
    window = clean[:max_chars]
    for marker in ("; ", ": ", ", "):
        idx = window.rfind(marker)
        if idx >= int(max_chars * 0.65):
            trimmed = window[:idx].rstrip(" ,;:")
            if trimmed and trimmed[-1] not in ".!?":
                trimmed += "."
            return trimmed

    # Avoid mid-thought truncation; keep the full statement.
    return clean

def _summary_has_all_components(summary: str) -> bool:
    text = summary.lower()
    checks = [
        ["introduction", "objective", "background", "aim", "rationale"],
        ["methods", "methodology", "design", "participants", "analysis"],
        ["results", "findings", "outcomes", "effect"],
        ["discussion", "limitations", "implications", "interpretation"],
        ["conclusion", "conclusions", "overall"],
    ]
    return all(any(keyword in text for keyword in group) for group in checks)

def _summary_fragment(value: str, *, max_chars: int, sentence: bool = False) -> str:
    text = " ".join(_strip_confidence_annotations(str(value or "")).split()).strip()
    if not text:
        return ""
    if sentence:
        parts = re.split(r"(?<=[.!?])\s+", text)
        if parts:
            text = parts[0].strip()
    text = text.rstrip(" \t\r\n.;,:!?")
    if len(text) > max_chars:
        text = _compact_without_cutoff(text, max_chars=max_chars).rstrip(" \t\r\n.;,:!?")
    if sentence and text and text[-1] not in ".!?":
        text += "."
    return text


def _coverage_int(coverage_block: Any, key: str) -> int:
    if not isinstance(coverage_block, dict):
        return 0
    try:
        return max(0, int(coverage_block.get(key, 0) or 0))
    except Exception:
        return 0


def _media_counts_line(coverage: dict[str, Any]) -> str:
    if not isinstance(coverage, dict):
        return ""
    main_fig_text = _coverage_int(coverage.get("figures", {}), "expected")
    main_fig_extracted = _coverage_int(coverage.get("figures", {}), "extracted")
    main_table_text = _coverage_int(coverage.get("tables", {}), "expected")
    main_table_extracted = _coverage_int(coverage.get("tables", {}), "extracted")
    supp_fig_text = _coverage_int(coverage.get("supp_figures", {}), "expected")
    supp_fig_extracted = _coverage_int(coverage.get("supp_figures", {}), "extracted")
    supp_table_text = _coverage_int(coverage.get("supp_tables", {}), "expected")
    supp_table_extracted = _coverage_int(coverage.get("supp_tables", {}), "extracted")
    return (
        "Figure/Table Coverage: "
        f"Main figures (text-cited/extracted): {main_fig_text}/{main_fig_extracted}; "
        f"Main tables (text-cited/extracted): {main_table_text}/{main_table_extracted}; "
        f"Supplementary figures (text-cited/extracted): {supp_fig_text}/{supp_fig_extracted}; "
        f"Supplementary tables (text-cited/extracted): {supp_table_text}/{supp_table_extracted}."
    )


def _ensure_executive_summary_counts(draft: dict[str, Any], coverage: dict[str, Any]) -> None:
    counts_line = _media_counts_line(coverage)
    if not counts_line:
        return
    summary = str(draft.get("executive_summary") or "").strip()
    if counts_line.lower() in summary.lower():
        return
    if summary:
        draft["executive_summary"] = summary.rstrip() + "\n\n" + counts_line
    else:
        draft["executive_summary"] = counts_line


def _discrepancy_uncertainty_line(item: dict[str, Any]) -> str:
    reason = str(item.get("reason", "")).strip().lower()
    claim = str(item.get("claim", "")).strip()
    if not claim:
        return ""
    if reason == "missing_modality":
        return f"Manual verification required: claim lacks cross-modality corroboration ({claim})."
    if reason == "unsupported":
        return f"Manual verification required: extracted evidence may not fully support this claim ({claim})."
    if reason == "contradicted":
        return f"Manual verification required: potential cross-modality conflict ({claim})."
    if reason == "magnitude_mismatch":
        return f"Manual verification required: potential numeric mismatch across modalities ({claim})."
    return f"Manual verification required: potential discrepancy ({claim})."


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
