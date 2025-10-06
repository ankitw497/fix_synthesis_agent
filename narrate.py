from __future__ import annotations

import json
import re
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# ---------- Logger ----------
logger = logging.getLogger("synthesis.narrate")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------- LLM client wrapper ----------
try:
    from llm_client import call_vertex  # type: ignore
except Exception:
    def call_vertex(prompt: str, model: str = "gemini-2.5-flash",
                    temperature: float = 0.6, timeout: int = 300) -> str:
        """
        Minimal fallback. Replace with your production Vertex/GenAI wrapper.
        """
        try:
            import vertexai  # type: ignore
            from vertexai.generative_models import GenerativeModel, GenerationConfig  # type: ignore
            genai = GenerativeModel(model)
            cfg = GenerationConfig(temperature=temperature, max_output_tokens=1024)
            resp = genai.generate_content([prompt], generation_config=cfg)
            return resp.text if hasattr(resp, "text") else (resp.candidates[0].content.parts[0].text or "")
        except Exception:
            pass
        try:
            import google.generativeai as genai  # type: ignore
            model_obj = genai.GenerativeModel(model_name=model)
            resp = model_obj.generate_content(prompt, generation_config={"temperature": temperature})
            if hasattr(resp, "text"):
                return resp.text
            if hasattr(resp, "candidates") and resp.candidates:
                return getattr(resp.candidates[0], "content", {}).get("parts", [{}])[0].get("text", "")
        except Exception as e:
            logger.warning("LLM fallback failed: %s", e)
            return json.dumps({"title":"Insight unavailable.","bullets":[f"LLM call failed: {e}"],"strapline":""})
        return json.dumps({"title":"Insight unavailable.","bullets":["LLM call failed (no client configured)."],"strapline":""})


# ---------- Prompt templates (view-aware) ----------
VIEW_TEMPLATES = {
    "trend": """You are generating concise business insights for a time-series trend chart.
Write:
- A single, natural-sounding TITLE (<= {title_max} chars) that states the key trend plainly.
- {bullet_min}-{bullet_max} short BULLETS (full sentences).
- A one-line STRAPLINE (<= {strap_max} chars) that ties to business objectives.

Context:
- Metric: {metric_id}
- Periods: {period_span}
- Key facts (JSON): {facts_json}

Guidance:
- Describe the direction, magnitude, and inflection points over time.
- Avoid generic labels; write as if it’s a headline.
- If seasonality or sustained shift exists, say so.
- Do NOT mention QoQ or YoY here; this is longitudinal.

Return JSON with keys: title, bullets, strapline.""",

    "qoq_split": """You are generating concise business insights for a QoQ chart SPLIT BY {split_label}.
Write:
- A single, natural-sounding TITLE (<= {title_max} chars) that highlights tier mix and contribution changes.
- {bullet_min}-{bullet_max} short BULLETS (full sentences).
- A one-line STRAPLINE (<= {strap_max} chars) that ties to business objectives.

Context:
- Metric: {metric_id}
- Period: {current_period}
- Split by: {split_label}
- Key facts (JSON): {facts_json}

Guidance:
- Call out leading vs lagging {split_label_lower} and any share shifts QoQ.
- Name 1–2 {split_label_lower} explicitly (e.g., SUPER PRIME, SUBPRIME) when relevant.
- Quantify shares or deltas when available in facts.
- This is a QoQ view: avoid YoY language.

Return JSON with keys: title, bullets, strapline.""",

    "yoy": """You are generating concise business insights for a YoY comparison chart.
Write:
- A single, natural-sounding TITLE (<= {title_max} chars) that states the YoY outcome.
- {bullet_min}-{bullet_max} short BULLETS (full sentences).
- A one-line STRAPLINE (<= {strap_max} chars) that ties to business objectives.

Context:
- Metric: {metric_id}
- Current vs Prior Year Periods: {period_span}
- Key facts (JSON): {facts_json}

Guidance:
- Use YoY language: “vs last year”, “YoY”.
- If composition/tier is relevant, mention top movers briefly, but stay focused on YoY.
- Quantify YoY % and absolute where available in facts.
- Avoid long-term trend or QoQ phrasing.

Return JSON with keys: title, bullets, strapline."""
}

# ---------- Core helpers ----------

TIER_LABELS = {"SUBPRIME","NEAR_PRIME","PRIME","PRIME_PLUS","SUPER_PRIME","UNSCORED","OTHER","TOTAL"}

def ensure_sentence_end(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    return text if text.endswith((".", "!", "?")) else (text + ".")

def parse_llm_json(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        title = ensure_sentence_end((data.get("title") or "").strip())
        bullets_in = data.get("bullets") or []
        bullets = [ensure_sentence_end(b.strip()) for b in bullets_in if isinstance(b, str) and b.strip()]
        strap = (data.get("strapline") or "").strip()
        return {"title": title, "bullets": bullets[:3], "strapline": strap}
    except Exception:
        return {"title":"Insight unavailable.","bullets":["Automatic fallback due to parsing error."],"strapline":""}

def _facts_json(facts: Dict[str, Any]) -> str:
    try:
        return json.dumps(facts or {}, ensure_ascii=False, separators=(",", ":"))[:6000]
    except Exception:
        return "{}"

def _build_period_fingerprint(periods: List[str], title: str = "", granularity: str = "") -> str:
    parts = [granularity or "", "|".join([str(p) for p in periods[:24]]), title or ""]
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:12]

def _cache_key(metric_id: str, view_type: str, split_dim: str,
               facts: Dict[str, Any], period_fingerprint: str) -> str:
    payload = {"metric": metric_id, "view": view_type, "split": split_dim,
               "facts": facts or {}, "period_fp": period_fingerprint}
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def choose_template(view_type: str) -> str:
    return VIEW_TEMPLATES.get(view_type, VIEW_TEMPLATES["trend"])

def detect_view_type(chart_type: str, df: Optional[pd.DataFrame] = None) -> str:
    if chart_type == "A3":
        return "qoq_split"
    if chart_type == "A4":
        return "yoy"
    return "trend"

def infer_split_dimension_from_df(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    if {"credit_tier","tier"}.intersection(cols):
        return "credit_tier"
    if {"score_bin","score_bins"}.intersection(cols):
        return "score_bin"
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            values = set(str(v).upper() for v in df[c].dropna().unique().tolist()[:20])
            if values & TIER_LABELS and len(values & TIER_LABELS) >= 3:
                return "credit_tier"
    return "none"

def compute_basic_stats(df: pd.DataFrame, period_col: str = "period",
                        value_col: str = "value") -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    if df is None or df.empty or value_col not in df.columns:
        return facts
    if period_col in df.columns:
        periods = sorted([str(x) for x in df[period_col].dropna().unique().tolist()])
        if periods:
            last_mask = df[period_col] == periods[-1]
            facts["level_last"] = float(df.loc[last_mask, value_col].sum())
    for c in df.columns:
        cn = str(c).lower()
        if "yoy" in cn or "qoq" in cn or cn.endswith("_pp") or cn.endswith("_pct"):
            try:
                series = df[c].dropna()
                if series.size:
                    facts[f"stat_{cn}_mean"] = float(series.mean())
            except Exception:
                pass
    return facts

def compute_tier_facts(df: pd.DataFrame, period_col: str = "period",
                       value_col: str = "value") -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    if df is None or df.empty or value_col not in df.columns or period_col not in df.columns:
        return facts
    periods = sorted([str(x) for x in df[period_col].dropna().unique().tolist()])
    if not periods:
        return facts
    last = df[df[period_col] == periods[-1]].copy()
    if last.empty:
        return facts
    total = float(last[value_col].sum()) or 1.0
    if "credit_tier" in last.columns:
        share = (last.groupby("credit_tier")[value_col].sum() / total).sort_values(ascending=False)
        facts["tier_top_share"] = [{"tier": str(t), "share": round(float(s) * 100, 1)} for t, s in share.head(2).items()]
    qoq_cols = [c for c in last.columns if "qoq" in str(c).lower()]
    if qoq_cols and "credit_tier" in last.columns:
        qcol = qoq_cols[0]
        d = last.groupby("credit_tier")[qcol].mean().sort_values()
        facts["tier_qoq_winners"] = [{"tier": str(t), "qoq": round(float(p) * 100 if abs(float(p)) <= 1.5 else float(p), 2)} for t, p in d.tail(2).items()]
        facts["tier_qoq_losers"]  = [{"tier": str(t), "qoq": round(float(p) * 100 if abs(float(p)) <= 1.5 else float(p), 2)} for t, p in d.head(2).items()]
    return facts

def _extract_numbers_from_text(text: str) -> List[float]:
    if not text:
        return []
    # capture integers, decimals, signed, and percents
    nums = re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', text.replace(",", ""))
    vals = []
    for n in nums:
        try:
            vals.append(float(n))
        except Exception:
            pass
    return vals

def _verify_numbers_against_data_card(data_card: Dict[str, Any],
                                      narrative: Dict[str, Any],
                                      tolerance: float = 0.05) -> List[str]:
    """Basic numeric plausibility checks; returns a list of soft warnings."""
    warnings: List[str] = []
    facts = data_card.get("facts", {})
    text = " ".join([narrative.get("title","")] + (narrative.get("bullets") or []) + [narrative.get("strapline","")])
    vals = _extract_numbers_from_text(text)
    if not vals:
        return warnings
    # Example: if level_last present, ensure no claimed % exceeds unrealistic bounds (e.g., > 1000%)
    for v in vals:
        if abs(v) > 10000:
            warnings.append(f"Number {v} looks unusually large.")
    # Add domain checks as needed
    return warnings

def _scan_contradictions(narrative: Dict[str, Any]) -> List[str]:
    """Naive contradiction scan (e.g., 'up' and 'down' simultaneously)."""
    hay = (narrative.get("title","") + " " + " ".join(narrative.get("bullets",[]))).lower()
    flags: List[str] = []
    if ("up" in hay and "down" in hay) or ("increase" in hay and "decrease" in hay):
        flags.append("Contains both up/increase and down/decrease.")
    return flags

def _reorder_narratives(narrative: Dict[str, Any]) -> Dict[str, Any]:
    """Put bullets with numbers first to feel 'insight-led'."""
    bullets = narrative.get("bullets") or []
    with_num = [b for b in bullets if _extract_numbers_from_text(b)]
    no_num = [b for b in bullets if not _extract_numbers_from_text(b)]
    narrative["bullets"] = (with_num + no_num)[:3]
    return narrative

def _validate_narrative(narrative: Dict[str, Any]) -> Dict[str, Any]:
    title = narrative.get("title","").strip()
    bullets = narrative.get("bullets") or []
    strap = narrative.get("strapline","").strip()
    if not title:
        narrative["title"] = "Insight unavailable."
    if not bullets:
        narrative["bullets"] = ["Performance update generated automatically."]
    narrative["title"] = ensure_sentence_end(narrative["title"])
    narrative["bullets"] = [ensure_sentence_end(b) for b in narrative["bullets"]][:3]
    narrative["strapline"] = strap
    return narrative

# ---------- Public API & wrappers ----------

def compile_chart_context(df: pd.DataFrame,
                          metric: str,
                          chart_type: str,
                          composition_base: Optional[str] = None) -> Dict[str, Any]:
    return build_data_card(df, metric, chart_type, composition_base)

def build_data_card(df: pd.DataFrame,
                    metric: str,
                    chart_type: str,
                    composition_base: Optional[str] = None) -> Dict[str, Any]:
    period_col = "period" if "period" in df.columns else None
    for cand in ("date","month","quarter","time"):
        if not period_col and cand in df.columns:
            period_col = cand
    periods: List[str] = []
    start_period = end_period = ""
    if period_col:
        periods = sorted([str(x) for x in df[period_col].dropna().unique().tolist()])
        if periods:
            start_period, end_period = periods[0], periods[-1]

    view_type = detect_view_type(chart_type, df)
    split_dim = infer_split_dimension_from_df(df)

    value_col = "value" if "value" in df.columns else df.columns[-1]
    facts = compute_basic_stats(df, period_col or "period", value_col)
    if split_dim == "credit_tier":
        facts.update(compute_tier_facts(df, period_col or "period", value_col))

    granularity = "quarterly" if any("Q" in p for p in periods) else "monthly"

    data_card: Dict[str, Any] = {
        "metric": metric,
        "chart_type": chart_type,
        "view_type": view_type,
        "split_dim": split_dim,
        "periods": periods,
        "start_period": start_period,
        "end_period": end_period,
        "facts": facts,
        "granularity": granularity,
        "composition_base": composition_base or "",
        "series_labels": [],
        "title": f"{metric} {chart_type}",
    }
    return data_card

def prepare_prompt_vars(data_card: Dict[str, Any], config) -> Tuple[str, Dict[str, Any]]:
    limits = getattr(config, "narrative_limits", {
        "title_max_chars": 85, "bullet_min": 2, "bullet_max": 3, "strapline_max_chars": 140
    })
    metric_id = data_card.get("metric","")
    view_type = data_card.get("view_type","trend")
    split_dim = data_card.get("split_dim","none")
    facts = data_card.get("facts",{})
    period_span = f"{data_card.get('start_period','')} → {data_card.get('end_period','')}"
    current_period = data_card.get("end_period","")
    vars = {
        "title_max": limits["title_max_chars"],
        "bullet_min": limits["bullet_min"],
        "bullet_max": limits["bullet_max"],
        "strap_max": limits["strapline_max_chars"],
        "metric_id": metric_id,
        "period_span": period_span,
        "facts_json": _facts_json(facts),
        "current_period": current_period,
        "split_label": "Credit Tier" if split_dim == "credit_tier" else ("Score Bin" if split_dim == "score_bin" else "Category"),
        "split_label_lower": "credit tier" if split_dim == "credit_tier" else ("score bin" if split_dim == "score_bin" else "category"),
    }
    return view_type, vars

def render_prompt_text(view_type: str, prompt_vars: Dict[str, Any], split_dim: str) -> str:
    template = choose_template(view_type)
    prompt = template.format(**prompt_vars)
    if split_dim == "credit_tier" and view_type == "qoq_split":
        prompt += "\nHard rule: Explicitly mention at least one Credit Tier by name (e.g., SUPER PRIME, SUBPRIME) and its QoQ movement."
    return prompt

def _build_insight_prompt(data_card: Dict[str, Any], config) -> str:
    view_type, pv = prepare_prompt_vars(data_card, config)
    return render_prompt_text(view_type, pv, data_card.get("split_dim","none"))

def _build_summary_prompt(data_card: Dict[str, Any], narrative: Dict[str, Any], config) -> str:
    """Optional: build a strapline refinement prompt (not used by default)."""
    base = (
        "Write a single-line strapline (<= {strap_max} chars) that ties the insight to business goals.\n"
        "Metric: {metric_id}\n"
        "View: {view_type}\n"
        "Current period: {end_p}\n"
        "Facts: {facts_json}\n"
        "Insight title: {title}\n"
        "Bullets: {bullets}\n"
    )
    limits = getattr(config, "narrative_limits", {"strapline_max_chars": 140})
    return base.format(
        strap_max=limits.get("strapline_max_chars", 140),
        metric_id=data_card.get("metric",""),
        view_type=data_card.get("view_type",""),
        end_p=data_card.get("end_period",""),
        facts_json=_facts_json(data_card.get("facts",{})),
        title=narrative.get("title",""),
        bullets=" | ".join(narrative.get("bullets",[])),
    )

def _build_notes_prompt(data_card: Dict[str, Any], narrative: Dict[str, Any], config) -> str:
    """Optional: build a speaker-notes prompt (not used by default)."""
    return (
        f"Create short speaker notes for {data_card.get('metric','Metric')} covering "
        f"{data_card.get('start_period','')} to {data_card.get('end_period','')}.\n"
        f"Insight: {narrative.get('title','')}\n"
        f"Bullets: {' | '.join(narrative.get('bullets',[]))}\n"
        "Keep it concise and executive-friendly."
    )

def _call_llm(prompt: str, config) -> str:
    model = getattr(config, "generative_model_name", "gemini-2.5-flash")
    temperature = getattr(config, "temperature", 0.6)
    timeout_seconds = getattr(config, "timeout_seconds", 300)
    return call_vertex(prompt=prompt, model=model, temperature=temperature, timeout=timeout_seconds)

# Back-compat alias
def call_llm(prompt: str, config) -> str:
    return _call_llm(prompt, config)

def generate_narrative(data_card: Dict[str, Any], config, use_stub: bool=False) -> Dict[str, Any]:
    metric_id = data_card.get("metric", "")
    view_type = data_card.get("view_type", "trend")
    split_dim = data_card.get("split_dim", "none")
    facts = data_card.get("facts", {})
    periods = data_card.get("periods", [])
    period_fp = _build_period_fingerprint(periods, data_card.get("title",""), data_card.get("granularity",""))
    ck = _cache_key(metric_id, view_type, split_dim, facts, period_fp)

    if use_stub:
        return _generate_fallback_narrative(data_card)

    prompt = _build_insight_prompt(data_card, config)
    raw = _call_llm(prompt, config)
    narr = parse_llm_json(raw)
    return narr

def llm_narrate(data_card: Dict[str, Any], config) -> Dict[str, Any]:
    narrative = generate_narrative(data_card, config)
    return narrative_qc(data_card, narrative, config)

def narrative_qc(data_card: Dict[str, Any],
                 narrative: Dict[str, Any],
                 config) -> Dict[str, Any]:
    title = (narrative.get("title") or "").strip()
    bullets = list(narrative.get("bullets") or [])
    strap = (narrative.get("strapline") or "").strip()

    if not bullets:
        bullets = ["Performance update generated automatically."]

    # Enforce tier mention for credit-tier QoQ split
    if data_card.get("split_dim") == "credit_tier" and data_card.get("view_type") == "qoq_split":
        tier_names = [ "SUPER PRIME","PRIME_PLUS","PRIME","NEAR_PRIME","SUBPRIME" ]
        hay = (title + " " + " ".join(bullets)).upper()
        if not any(t in hay for t in tier_names):
            facts = data_card.get("facts", {})
            pick = None
            for key in ("tier_top_share","tier_qoq_winners","tier_qoq_losers"):
                arr = facts.get(key) or []
                if arr:
                    pick = str(arr[0].get("tier","")).upper()
                    break
            if pick and pick in tier_names:
                bullets = [f"{pick.title()} shows a notable QoQ movement among credit tiers."] + bullets
            else:
                bullets = ["Credit tiers show clear mix shifts QoQ; higher-score tiers are leading."] + bullets

    narrative = {"title": ensure_sentence_end(title),
                 "bullets": [ensure_sentence_end(b) for b in bullets][:3],
                 "strapline": strap}

    # Soft QA hooks
    _ = _verify_numbers_against_data_card(data_card, narrative)
    _ = _scan_contradictions(narrative)
    narrative = _reorder_narratives(narrative)
    narrative = _validate_narrative(narrative)
    return narrative

def summarize_for_notes(narrative: Dict[str, Any]) -> str:
    t = (narrative.get("title") or "").rstrip(".")
    bs = [b.rstrip(".") for b in (narrative.get("bullets") or [])][:2]
    parts = [p for p in [t] + bs if p]
    return ". ".join(parts) + ("." if parts else "")

def generate_speaker_notes(data_card: Dict[str, Any],
                           narrative: Dict[str, Any],
                           config) -> str:
    metric = data_card.get("metric","Metric")
    view = data_card.get("view_type","trend")
    start_p = data_card.get("start_period","")
    end_p = data_card.get("end_period","")
    summary = summarize_for_notes(narrative)
    view_label = {"trend":"trend over time","qoq_split":"QoQ split by segment","yoy":"year-over-year comparison"}.get(view,"chart")
    notes = [f"{metric}: {summary}", f"This {view_label} covers {start_p} to {end_p}."]
    if data_card.get("split_dim") == "credit_tier" and view == "qoq_split":
        notes.append("Call out one or two credit tiers explicitly if probed.")
    return " ".join([n for n in notes if n])

def normalize_narrative(narr: Dict[str, Any]) -> Dict[str, Any]:
    title = ensure_sentence_end((narr.get("title") or "").strip())
    bullets = [ensure_sentence_end(b.strip()) for b in (narr.get("bullets") or []) if isinstance(b, str)]
    strap = (narr.get("strapline") or "").strip()
    return {"title": title, "bullets": bullets[:3], "strapline": strap}

def serialize_narrative(narr: Dict[str, Any]) -> str:
    return json.dumps({"title": narr.get("title",""),
                       "bullets": narr.get("bullets",[]),
                       "strapline": narr.get("strapline","")},
                      ensure_ascii=False)

def _generate_fallback_narrative(data_card: Dict[str, Any]) -> Dict[str, Any]:
    metric = data_card.get("metric","Metric")
    view = data_card.get("view_type","trend")
    facts = data_card.get("facts",{})
    title = f"{metric} shows stable {view.replace('_',' ')} performance."
    bullets = []
    if "level_last" in facts:
        bullets.append(f"Latest level is {facts['level_last']}.")
    if data_card.get("split_dim") == "credit_tier":
        tt = facts.get("tier_top_share", [])
        if tt:
            bullets.append(f"{tt[0]['tier'].title()} holds ~{tt[0]['share']}% share most recently.")
    return {"title": title, "bullets": bullets or ["Automated summary."], "strapline": ""}

def stub_llm_narrative(data_card: Dict[str, Any]) -> Dict[str, Any]:
    return _generate_fallback_narrative(data_card)

def debug_prompt_preview(data_card: Dict[str, Any], config) -> str:
    return _build_insight_prompt(data_card, config)
