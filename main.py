"""
Main orchestrator for Agent 3 synthesis pipeline.
Implements CLI, family resolution, and end-to-end pipeline execution.
"""

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import warnings
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

from synthesis_agent.config import load_config, SynthesisConfig
from synthesis_agent.utils import (
    setup_logging,
    slugify_client,
    ensure_triplet,
    coalesce_suffix_columns,
    prev_quarter,
    prev_year_same_quarter,
    prev_month,
    canonical_sort_periods,
    pretty_label,
    fmt_percent,
    fmt_delta,
)
from synthesis_agent.io_normalize import (
    load_frame, load_manifest, attach_lob_context,
    align_periods, normalize_column_names, normalize_rates,
    compute_display_derivations, validate_data_integrity,
    # added to fix silent file loads:
    clean_numeric_columns, normalize_period, normalize_score_tiers,
    hydrate_topic_groups_with_columns, compute_calculated_metrics,
    compute_missing_deltas, normalize_engagement, select_composition_base,
    extract_tier_columns, is_rate_metric, qa_assert_delta_sanity,
    infer_percent_scale,
)
from synthesis_agent.validate_plan import (
    CoverageLedger, check_product_restrictions,
    check_delta_base_availability, get_appropriate_footer,
    rule_applicable, apply_fallback_strategy, validate_plan_consistency
)
from synthesis_agent.semantics import resolve_analysis_modes
from synthesis_agent.charts import (
    line_trend_A2, stacked100_A3, dual_axis_A5, delta_over_time_A4,
    delinquency_small_multiples, get_cached_figure, clear_figure_cache,
    counts_deltas_A4, select_delta_col,
    build_trend_line_spec, build_compare_grouped_by_tier_spec,
    build_dpd_trend_spec, build_dpd_severity_spec,
    build_multi_period_grouped_by_tier_spec,
    prepare_dpd_trend_dataframe,
)
from synthesis_agent.narrate import (
    build_data_card, llm_narrate, narrative_qc, generate_speaker_notes
)
from synthesis_agent.deck import (
    bind_template, add_cover_slide, add_agenda_slide,
    add_insight_slide, add_topic_summary, add_appendix_slide,
    add_thank_you_slide, export_pptx, export_pdf, add_section_divider,
    add_chart_slide
)

logger = setup_logging("main")


DPD_METRIC_SHORT_LABELS = {
    'deliq_30_acct_rate': 'Account 30+ DPD',
    'deliq_60_acct_rate': 'Account 60+ DPD',
    'deliq_90_acct_rate': 'Account 90+ DPD',
    'cnsmr_cnts_w_deliq_bal_30_rate': 'Consumer 30+ DPD',
    'cnsmr_cnts_w_deliq_bal_60_rate': 'Consumer 60+ DPD',
    'cnsmr_cnts_w_deliq_bal_90_rate': 'Consumer 90+ DPD',
    'deliq_30_acct_bal_rate': 'Balance 30+ DPD',
    'deliq_60_acct_bal_rate': 'Balance 60+ DPD',
    'deliq_90_acct_bal_rate': 'Balance 90+ DPD',
}


_TITLE_LONG_NUMBER_RE = re.compile(r"\d{5,}")
_TITLE_ID_TOKEN_RE = re.compile(r"\b(?:account|acct|loan|customer|id)\s*#?\d{3,}\b", re.IGNORECASE)
_QUARTER_WORDS = {
    "Q1": "First-quarter",
    "Q2": "Second-quarter",
    "Q3": "Third-quarter",
    "Q4": "Fourth-quarter",
}


def _contains_numeric_tokens(text: Optional[str]) -> bool:
    if not text:
        return False
    if _TITLE_LONG_NUMBER_RE.search(text):
        return True
    if _TITLE_ID_TOKEN_RE.search(text):
        return True
    return False


def _sanitize_descriptive_title(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = _TITLE_ID_TOKEN_RE.sub("", text)
    return " ".join(str(cleaned).split())


def _quarter_label_to_words(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    upper = str(label).upper()
    if upper in _QUARTER_WORDS:
        return _QUARTER_WORDS[upper]
    return upper


def _descriptive_chart_title(
    metric: str,
    chart_type: str,
    short_delta: Optional[str] = None,
    focus_label: Optional[str] = None,
) -> str:
    base = pretty_label(metric)
    suffix = "Performance overview"
    if chart_type == 'A2':
        focus_words = _quarter_label_to_words(focus_label)
        if focus_words:
            suffix = f"{focus_words} trend across years"
        else:
            suffix = "Trend over time"
    elif chart_type == 'A3':
        suffix = "Credit tier mix"
        if focus_label:
            focus_words = _quarter_label_to_words(focus_label)
            if focus_words:
                suffix = f"{focus_words} tier mix across years"
    elif chart_type == 'A4':
        delta_map = {
            'qoq': 'Quarter-over-quarter change',
            'yoy': 'Year-over-year change',
            'mom': 'Month-over-month change',
        }
        if short_delta:
            suffix = delta_map.get(short_delta.lower(), 'Period-over-period change')
        else:
            suffix = 'Period-over-period change'
    elif chart_type == 'A5':
        suffix = 'Dual-axis comparison'
    return f"{base} — {suffix}"



def _focus_suffix_from_period(period: Optional[str]) -> Optional[str]:
    """Return a canonical focus suffix (e.g., 'Q2') from a period label."""

    if period is None:
        return None
    text = str(period).upper()
    if "Q" in text:
        trailing = text.split("Q")[-1]
        digits = "".join(ch for ch in trailing if ch.isdigit())
        if digits:
            return f"Q{digits}"
        if trailing:
            return f"Q{trailing[0]}"
        return "Q"
    return None


@dataclass
class PipelineContext:
    """
    Tracks artifacts generated during pipeline execution.
    Used for acceptance checks to verify actual outputs match spec.
    """
    config: SynthesisConfig
    figures: List[Dict[str, Any]] = field(default_factory=list)
    a3_figures: List[Dict[str, Any]] = field(default_factory=list)
    slides: List[Dict[str, Any]] = field(default_factory=list)
    narratives: List[Dict[str, Any]] = field(default_factory=list)
    all_slides_single_content: bool = True
    figures_cached: int = 0
    strict_mode: bool = False
    dry_run: bool = False
    topic_has_image: Dict[str, bool] = field(default_factory=dict)
    latest_period: Optional[str] = None
    # Deduplication tracking
    slide_cache_keys: Set[str] = field(default_factory=set)
    slide_content_hashes: Set[str] = field(default_factory=set)
    sections_added: Set[str] = field(default_factory=set)
    
    def add_figure(self, fig_data: Dict[str, Any], chart_type: str):
        """Add a figure artifact with metadata."""
        self.figures.append(fig_data)
        if chart_type == 'A3':
            self.a3_figures.append(fig_data)
    
    def add_slide(self, slide_data: Dict[str, Any]):
        """Add a slide artifact."""
        self.slides.append(slide_data)
        # Check if slide violates single-content
        if slide_data.get('content_count', 1) > 1:
            self.all_slides_single_content = False
    
    def add_narrative(self, narrative_data: Dict[str, Any]):
        """Add a narrative artifact."""
        self.narratives.append(narrative_data)


def _delinquency_headline(
    trend_df: pd.DataFrame,
    period_col: str,
    metric: str,
    granularity: Optional[str],
    label_hint: str,
) -> str:
    """Build a deterministic delinquency headline using YoY delta when available."""

    fallback_label = DPD_METRIC_SHORT_LABELS.get(metric) or (
        f"{label_hint} Delinquencies" if label_hint else pretty_label(metric)
    )
    if trend_df is None or metric not in trend_df.columns or trend_df.empty:
        return fallback_label

    frame = trend_df[[period_col, metric]].dropna(subset=[metric]).copy()
    if frame.empty:
        return fallback_label

    frame[period_col] = frame[period_col].astype(str)
    ordered = canonical_sort_periods(frame[period_col].tolist())
    if ordered:
        try:
            frame = frame.set_index(period_col).reindex(ordered).reset_index()
        except Exception:
            frame = frame.sort_values(period_col)

    frame = frame.dropna(subset=[metric], how='all')
    if frame.empty:
        return fallback_label

    latest_row = frame.iloc[[-1]]
    latest_period = str(latest_row[period_col].iloc[0])
    latest_value = pd.to_numeric(latest_row[metric], errors='coerce').iloc[0]

    series_values = pd.to_numeric(frame[metric], errors='coerce')
    scale = infer_percent_scale(series_values)
    level_value = latest_value
    if pd.notna(level_value):
        if scale == 'percent' or (scale == 'unknown' and abs(level_value) > 1.5):
            level_value = level_value / 100.0
    level_text = fmt_percent(level_value) if pd.notna(level_value) else None

    gran = (granularity or 'quarterly').lower()
    delta_frame = compute_missing_deltas(frame.copy(), metric, 'yoy', 'quarterly' if 'quarter' in gran else 'monthly')
    delta_col = f"{metric}_yoy_pp"
    delta_value = None
    if delta_col in delta_frame.columns:
        match = delta_frame[delta_frame[period_col].astype(str) == latest_period]
        if not match.empty:
            raw = pd.to_numeric(match.iloc[-1][delta_col], errors='coerce')
            if pd.notna(raw):
                delta_value = float(raw) / 100.0

    short_label = DPD_METRIC_SHORT_LABELS.get(metric) or (
        f"{label_hint.strip()}" if label_hint else pretty_label(metric)
    )

    direction_phrase = None
    if delta_value is not None and abs(delta_value) >= 0.0005:
        direction_phrase = "rising year over year" if delta_value > 0 else "declining year over year"
    if direction_phrase is None and len(frame) >= 2:
        previous_value = pd.to_numeric(frame.iloc[-2][metric], errors='coerce')
        if pd.notna(previous_value) and pd.notna(latest_value):
            diff = latest_value - previous_value
            threshold = 0.0005
            if scale not in {'percent', 'unknown'}:
                threshold = 0.5 if abs(latest_value) > 1 else 0.0005
            if abs(diff) >= threshold:
                direction_phrase = "trending higher recently" if diff > 0 else "trending lower recently"
    if direction_phrase is None:
        direction_phrase = "holding steady"
    return f"{short_label} trend {direction_phrase}"


def _delinquency_series_tag(metrics: List[str]) -> str:
    """Return a compact tag like '30/60/90' for available delinquency metrics."""

    seen: List[str] = []
    for metric in metrics:
        label = DPD_METRIC_SHORT_LABELS.get(metric)
        if not label:
            continue
        match = re.search(r"(\d+\+?)", label)
        tag = match.group(1) if match else label
        if tag not in seen:
            seen.append(tag)

    if not seen:
        return ""

    def _numeric_key(tag: str) -> int:
        digits = "".join(ch for ch in tag if ch.isdigit())
        return int(digits) if digits else 0

    ordered = sorted(seen, key=_numeric_key)
    return "/".join(ordered)


def _delinquency_headline_multi(
    trend_df: pd.DataFrame,
    period_col: str,
    metrics: List[str],
    granularity: Optional[str],
    label_hint: str,
) -> str:
    """Generate a delinquency headline covering all provided metrics."""

    if not metrics:
        return (
            f"{label_hint} Delinquencies" if label_hint else "Delinquency trend"
        )

    series_tag = _delinquency_series_tag(metrics)
    preferred = next(
        (
            metric
            for needle in ("90", "60", "30")
            for metric in metrics
            if needle in DPD_METRIC_SHORT_LABELS.get(metric, "")
        ),
        metrics[0],
    )

    base = _delinquency_headline(trend_df, period_col, preferred, granularity, label_hint)
    prefix_candidates = [
        DPD_METRIC_SHORT_LABELS.get(preferred),
        f"{label_hint} Delinquencies" if label_hint else None,
    ]
    replacement = " ".join(
        part for part in [label_hint, f"{series_tag} DPD" if series_tag else "Delinquencies"] if part
    ).strip()
    for prefix in prefix_candidates:
        if prefix and base.startswith(prefix):
            return base.replace(prefix, replacement, 1).strip()
    return f"{replacement} — {base}".strip(" –-")


def _clean_tier_label(value: Any) -> str:
    text = str(value or "").replace('_', ' ').strip()
    if not text:
        return "Other tiers"
    return text.title()


def _format_tier_caption(tiers: List[Any]) -> str:
    cleaned: List[str] = []
    for tier in tiers:
        label = _clean_tier_label(tier)
        if not label or label.lower() == "total":
            continue
        if label not in cleaned:
            cleaned.append(label)
    if not cleaned:
        return "Selected tiers"
    return ", ".join(cleaned)


def _delinquency_split_insight(
    latest_df: pd.DataFrame,
    tier_col: str,
    metrics: List[str],
) -> str:
    """Generate a descriptive delinquency severity headline."""

    if latest_df is None or latest_df.empty or tier_col not in latest_df.columns:
        return "Delinquency severity by tier"

    preferred = [m for m in metrics if "90" in m] or [m for m in metrics if "60" in m] or list(metrics[:1])
    if not preferred:
        return "Delinquency severity by tier"

    metric = preferred[0]
    if metric not in latest_df.columns:
        return "Delinquency severity by tier"

    frame = latest_df[[tier_col, metric]].dropna()
    if frame.empty:
        return "Delinquency severity by tier"

    frame[tier_col] = frame[tier_col].astype(str).str.upper()
    grouped = frame.groupby(tier_col, as_index=False)[metric].mean()
    if grouped.empty:
        return "Delinquency severity by tier"

    grouped = grouped.sort_values(metric, ascending=False)
    top_tiers = grouped[tier_col].head(2).tolist()
    readable = [_clean_tier_label(tier) for tier in top_tiers if tier]

    if len(readable) >= 2:
        return f"Delinquency severity concentrates in {readable[0]} and {readable[1]} tiers"
    if len(readable) == 1:
        return f"Delinquency severity concentrates in {readable[0]} tier"
    return "Delinquency severity by tier"


def _qc_implied_avg_alert(df: pd.DataFrame, period_col: str, tot_acct_col: str, 
                          tot_cnsmr_col: str, avg_col: str, delta_col: str) -> Optional[str]:
    """
    QC check for contradictory math in average metrics.
    Compares implied average change from totals with reported delta.
    
    Args:
        df: DataFrame with metrics
        period_col: Period column name
        tot_acct_col: Total accounts column
        tot_cnsmr_col: Total consumers column
        avg_col: Average column
        delta_col: Delta column
    
    Returns:
        Alert message if contradiction found, None otherwise
    """
    try:
        import numpy as np
        
        # Select relevant columns and drop NaN
        cols = [period_col, tot_acct_col, tot_cnsmr_col, avg_col, delta_col]
        w = df[cols].dropna()
        
        if len(w) < 2:
            return None  # Need at least 2 periods for comparison
            
        w = w.sort_values(period_col)
        
        # Calculate implied change from totals
        implied_avg = w[tot_acct_col] / w[tot_cnsmr_col]
        implied_change = implied_avg.pct_change().iloc[-1] * 100
        
        # Get reported change
        reported = w[delta_col].iloc[-1]
        
        # Check for contradiction (different signs or large difference)
        if np.sign(implied_change) != np.sign(reported) and abs(reported - implied_change) > 5:
            return (f"⚠ Integrity: {avg_col} {reported:+.1f}% conflicts with implied "
                   f"change {implied_change:+.1f}% from totals.")
    except Exception as e:
        logger.debug(f"QC check failed: {e}")
        pass
    
    return None


def _scan_latest_period(manifest: Dict) -> Optional[str]:
    """
    Scan manifest files to find the latest period.
    
    Args:
        manifest: Manifest dictionary with file information
    
    Returns:
        Latest period string or None
    """
    latest = None
    for f in manifest.get('files', []):
        p = f.get('path') or f.get('file')
        if not p:
            continue
        try:
            df = load_frame(p)
            if 'period' in df.columns and len(df) > 0:
                # Get the max period value
                lp = str(df['period'].max())
                if (latest is None) or (lp > latest):
                    latest = lp
        except Exception as e:
            logger.debug(f"Could not scan {p} for period: {e}")
            continue
    
    # Format period for display (e.g., "2022Q4" -> "Q4 2022")
    if latest:
        # Handle quarterly format
        if 'Q' in latest:
            year = latest[:4]
            quarter = latest[4:]
            return f"{quarter} {year}"
        # Handle monthly format (YYYYMM)
        elif len(latest) == 6 and latest.isdigit():
            year = latest[:4]
            month = int(latest[4:])
            months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            if 1 <= month <= 12:
                return f"{months[month]} {year}"
    
    return latest


def log_startup_matrix(engagement_data: Optional[Dict], requested_metrics: Dict):
    """
    Print requested (topic, group, metric) matrix before generation starts.
    Per dev plan requirement.
    """
    logger.info("=== Requested Coverage Matrix ===")
    
    if not engagement_data:
        logger.info("No engagement spec loaded - using column-driven coverage")
        return
    
    total_metrics = 0
    for topic_name, groups in requested_metrics.items():
        logger.info(f"Topic: {topic_name}")
        for group_name, metrics in groups.items():
            logger.info(f"  Group: {group_name}")
            for metric in metrics:
                logger.info(f"    - {metric}")
                total_metrics += 1
    
    logger.info(f"Total requested metrics: {total_metrics}")
    logger.info("=================================")


def detect_family_resolution(manifest: Dict, engagement: Dict, config: SynthesisConfig) -> Dict[str, Any]:
    """
    Detect granularity and delta family from data.
    
    Args:
        manifest: Manifest dictionary
        engagement: Engagement dictionary
        config: Synthesis configuration
    
    Returns:
        Family resolution dictionary
    """
    # Check what's available in manifest
    m_tokens = {str(f.get("analysis_type", "")).lower()
                for f in manifest.get("files", []) if f.get("analysis_type")}
    
    # Check what's requested in engagement
    e_tokens = set(engagement.get("analysis_types", [])) if engagement else set()
    
    # Pick short delta
    def pick_delta():
        for k in ("qoq", "mom", "yoy"):
            if k in m_tokens:
                return k
        for k in ("qoq", "mom", "yoy"):
            if k in e_tokens:
                return k
        return "qoq"  # Safe default
    
    short_delta = pick_delta()
    
    # Check YoY availability
    use_yoy = ("yoy" in m_tokens) or ("yoy" in e_tokens) or config.logic.use_yoy_default
    
    # Detect granularity
    if config.logic.granularity_family != "auto":
        gran = config.logic.granularity_family
        # Special case: "both" means include all granularities
        if gran == "both":
            # Check what's actually available
            has_quarterly = any("quarterly" in str(f.get("granularity", "")).lower()
                               for f in manifest.get("files", []))
            has_monthly = any("monthly" in str(f.get("granularity", "")).lower()
                             for f in manifest.get("files", []))
            if not (has_quarterly and has_monthly):
                # If we don't have both, fall back to what we have
                gran = "quarterly" if has_quarterly else "monthly"
    else:
        # Infer from manifest
        has_quarterly = any("quarterly" in str(f.get("granularity", "")).lower()
                           for f in manifest.get("files", []))
        has_monthly = any("monthly" in str(f.get("granularity", "")).lower()
                         for f in manifest.get("files", []))
        
        if has_quarterly:
            gran = "quarterly"
        elif has_monthly:
            gran = "monthly"
        else:
            gran = "quarterly"  # Default
    
    result = {"granularity": gran, "short_delta": short_delta, "use_yoy": use_yoy}
    
    # CRITICAL: Log exact format
    logger.info(f"Resolved families → granularity: {gran}, short_delta: {short_delta}, use_yoy: {str(use_yoy).lower()}")
    
    return result


def _resolve_families(manifest: Dict, config: SynthesisConfig) -> Dict[str, str]:
    """
    Resolve single family for entire run (deprecated - use detect_family_resolution).
    CRITICAL: Must log exact resolution line.
    
    Args:
        manifest: Manifest dictionary
        config: Synthesis configuration
    
    Returns:
        Family resolution dictionary
    """
    files = manifest.get('files', [])
    
    # Detect available granularities
    has_quarterly = any('quarterly' in f.get('granularity', '') for f in files)
    has_monthly = any('monthly' in f.get('granularity', '') for f in files)
    
    # Detect available delta types
    has_yoy = any('yoy' in f.get('analysis_type', '') for f in files)
    has_qoq = any('qoq' in f.get('analysis_type', '') for f in files)
    has_mom = any('mom' in f.get('analysis_type', '') for f in files)
    
    # Resolve based on config
    if config.logic.granularity_family == 'quarterly':
        granularity = 'quarterly'
        short_delta = 'qoq' if has_qoq else 'yoy'
    elif config.logic.granularity_family == 'monthly':
        granularity = 'monthly'
        short_delta = 'mom' if has_mom else 'qoq'
    elif config.logic.granularity_family == 'both':
        granularity = 'both'  # Include all granularities
        # Pick preferred short_delta based on what's available
        if has_qoq:
            short_delta = 'qoq'
        elif has_mom:
            short_delta = 'mom'
        else:
            short_delta = 'yoy'
    else:  # auto
        if has_quarterly:
            granularity = 'quarterly'
            short_delta = 'qoq' if has_qoq else 'yoy'
        else:
            granularity = 'monthly'
            short_delta = 'mom' if has_mom else 'qoq'
    
    use_yoy = has_yoy and config.logic.use_yoy_default
    
    resolution = {
        'granularity': granularity,
        'short_delta': short_delta,
        'use_yoy': use_yoy
    }
    
    # CRITICAL: Log exact resolution line
    logger.info(f"Resolved families → granularity: {granularity}, short_delta: {short_delta}, use_yoy: {str(use_yoy).lower()}")
    
    return resolution


def load_all_frames(files: List[Dict], config: SynthesisConfig) -> Optional[pd.DataFrame]:
    """
    Load and concatenate all frames from a list of files.
    
    Args:
        files: List of file dictionaries
        config: Synthesis configuration
    
    Returns:
        Concatenated DataFrame or None if no files loaded
    """
    frames = []
    for f in files:
        try:
            df = load_frame(Path(f['path']))
            df = clean_numeric_columns(df)
            df = normalize_column_names(df)
            df = normalize_period(df)
            # Store granularity column
            df['granularity'] = str(f.get('granularity', '')).lower()
            frames.append(df)
            logger.debug(f"Loaded {f.get('filename', f['path'])}: {len(df)} rows")
        except Exception as e:
            logger.error(
                f"Skipping {f.get('filename', f['path'])} due to {type(e).__name__}: {e}",
                exc_info=True
            )
    
    if not frames:
        return None
    
    # Concatenate all frames
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates()
    logger.info(f"Concatenated {len(frames)} files into {len(df_all)} rows")
    return df_all


def process_topic(topic: str,
                 topic_files: List[Dict],
                 config: SynthesisConfig,
                coverage_ledger: CoverageLedger,
                presentation: Any,
                ctx: PipelineContext,
                topic_spec: Optional[Dict] = None,
                family_resolution: Optional[Dict] = None,
                allowed_tiers: Optional[List[str]] = None,
                analysis_modes: Optional[Dict[str, Any]] = None,
                current_period: Optional[str] = None,
                comparison_period: Optional[str] = None) -> Dict[str, Any]:
    """
    Process single topic with semantic understanding.
    
    Args:
        topic: Topic name
        topic_files: List of file info for this topic
        config: Synthesis configuration
        coverage_ledger: Coverage tracking
        presentation: Presentation object
        ctx: Pipeline context for tracking artifacts
        topic_spec: Topic specification from normalized engagement
        family_resolution: Family resolution with allowed delta types
        allowed_tiers: Allowed credit tiers from engagement
    
    Returns:
        Processing results
    """
    results = {
        'topic': topic,
        'slides_created': 0,
        'charts_generated': 0,
        'figures_cached': 0,
        'narratives_generated': 0,
        'files_processed': 0,
        'warnings': []
    }
    
    # Early exit if no files (PoC mode)
    if not topic_files:
        logger.warning(f"No files found for topic {topic}, skipping (PoC mode)")
        return results
    
    logger.info(f"Processing topic: {topic} with {len(topic_files)} files")
    
    # Add topic section divider once
    if presentation and not ctx.dry_run and topic not in ctx.sections_added:
        add_section_divider(presentation, topic.replace('_', ' ').title(), '', config, prefer_blue_cover=False)
        ctx.sections_added.add(topic)
    
    # Load and normalize ALL files for this topic
    from functools import reduce
    from synthesis_agent.semantics import topic_to_lob
    
    # Group files by analysis type
    files_by_analysis = {}
    for f in topic_files:
        analysis_type = str(f.get('analysis_type', 'TRENDS')).upper()
        if analysis_type not in files_by_analysis:
            files_by_analysis[analysis_type] = []
        files_by_analysis[analysis_type].append(f)
    
    # Load and concatenate all files for each analysis type
    frames_by_type = {}
    for analysis_type, files in files_by_analysis.items():
        df_combined = load_all_frames(files, config)
        if df_combined is not None:
            # Apply additional normalizations
            df_combined = normalize_score_tiers(df_combined, allowed_tiers=allowed_tiers)
            df_combined = normalize_rates(df_combined)
            df_combined = attach_lob_context(df_combined, topic)
            frames_by_type[analysis_type] = df_combined
            results['files_processed'] += len(files)
            logger.info(f"Loaded {analysis_type}: {len(df_combined)} rows from {len(files)} files")
    
    if not frames_by_type:
        logger.warning(f"No usable files for topic {topic}")
        return results
    
    logger.info(f"Loaded frames: {', '.join([f'{k}: {len(v)} rows' for k, v in frames_by_type.items()])}")
    
    # Build a union frame for metrics discovery and A2/A5
    # join keys: period + (tier if present)
    def _keys(df):
        return ['period'] + (['score_curr_tier'] if 'score_curr_tier' in df.columns else [])
    
    base = next(iter(frames_by_type.values()))
    union = base.copy()
    # Ensure period is string in union
    if 'period' in union.columns:
        union['period'] = union['period'].astype(str)
    
    for k, frame in frames_by_type.items():
        if frame is base: 
            continue
        
        # Ensure period is string in frame to merge
        df_to_merge = frame.copy()
        if 'period' in df_to_merge.columns:
            df_to_merge['period'] = df_to_merge['period'].astype(str)
        
        on_cols = list(set(_keys(union)) & set(_keys(df_to_merge)))
        if on_cols:
            union = pd.merge(union, df_to_merge, on=on_cols, how='outer', suffixes=('', f'__{k}'))
        else:
            # if no common keys, just concat (very rare in our spec)
            union = pd.concat([union, df_to_merge], axis=0, ignore_index=True)
    
    # Canonical sort of periods after building union
    from synthesis_agent.utils import canonical_sort_periods
    if 'period' in union.columns:
        # Build canonical category order for period
        ordered = canonical_sort_periods(union['period'].astype(str).unique().tolist())
        
        # Ensure uniqueness (critical!)
        ordered = list(dict.fromkeys(ordered))  # dedupe while preserving order
        
        # Make sure the column is string and drop NA rows before categorizing
        union = union.dropna(subset=['period']).copy()
        union['period'] = union['period'].astype(str)
        union['period'] = pd.Categorical(union['period'], categories=ordered, ordered=True)
        union = union.sort_values('period').reset_index(drop=True)
    
    # Coalesce suffixed columns from outer merges
    union = coalesce_suffix_columns(union)
    
    # Helper to prefer one granularity for trend/snapshot
    def _prefer_granularity(df, prefer='quarterly', min_points=3):
        if df is None or 'granularity' not in df.columns:
            return df
        # try preferred first
        pref = df[df['granularity'] == prefer]
        if len(pref) >= min_points:
            return pref
        # otherwise try the other if it has enough data
        other = df[df['granularity'] != prefer]
        return other if len(other) >= min_points else df
    
    # Helpers to pick the right source frame per chart
    short_delta = (family_resolution or {}).get('short_delta', 'qoq')
    
    # Fixed: Use explicit None checks instead of 'or' with DataFrames
    df_trend = frames_by_type.get('TRENDS')
    if df_trend is None:
        df_trend = frames_by_type.get('SNAPSHOT')
        if df_trend is None:
            df_trend = union
    
    df_snapshot = frames_by_type.get('SNAPSHOT')
    if df_snapshot is None:
        df_snapshot = union
    
    # Apply granularity preference
    df_trend = _prefer_granularity(df_trend, prefer='quarterly', min_points=config.logic.min_data_points)
    df_snapshot = _prefer_granularity(df_snapshot, prefer='quarterly', min_points=config.logic.min_data_points)
    if df_snapshot is not None and 'period' in df_snapshot.columns:
        df_snapshot = df_snapshot.copy()
        df_snapshot['period'] = df_snapshot['period'].astype(str)
    
    # Ensure SNAPSHOT frame exists if we have tier columns (for A3 composition charts)
    if extract_tier_columns(union):
        if 'SNAPSHOT' not in frames_by_type:
            frames_by_type['SNAPSHOT'] = union
            logger.info("Added union as SNAPSHOT frame since tier columns were detected")
        # Also ensure df_snapshot is set
        if df_snapshot is None or len(df_snapshot) == 0:
            df_snapshot = union
    
    df_delta = frames_by_type.get(short_delta.upper())
    
    logger.info(f"Union frame has {len(union)} rows, {len(union.columns)} columns")
    
    # Hydrate topic spec with concrete metrics if provided (use union frame)
    if topic_spec:
        topic_spec = hydrate_topic_groups_with_columns(topic_spec, union)
    
    # Validate data
    is_valid, issues = validate_data_integrity(union)
    if not is_valid:
        results['warnings'].extend(issues)
    
    # If no topic spec, fall back to column-driven coverage
    if not topic_spec or not topic_spec.get('insight_groups'):
        logger.debug(f"No insight groups for {topic}, using column-driven coverage")
        # Create synthetic groups from available columns
        numeric_cols = union.select_dtypes(include=['number']).columns
        topic_spec = {
            'name': topic,
            'insight_groups': [{
                'name': 'Available Metrics',
                'metrics': [col for col in numeric_cols if 'period' not in col.lower()],
                'calculated': []
            }]
        }
    
    # Process each insight group
    for group in topic_spec.get('insight_groups', []):
        group_name = group['name']
        metrics = group.get('metrics', [])
        calculated = group.get('calculated', [])
        
        # Auto-expand empty metrics list from semantic mappings
        if not metrics:
            from synthesis_agent.semantics import INSIGHT_GROUP_TO_METRICS
            if group_name in INSIGHT_GROUP_TO_METRICS:
                metrics = [m["column"] for m in INSIGHT_GROUP_TO_METRICS[group_name]["required_metrics"]]
                logger.info(f"Auto-expanded empty metrics for {group_name}: {len(metrics)} metrics from semantics")
            else:
                # Fallback: try common metrics based on available columns
                fallback_candidates = ["tot_acct_bal", "tot_acct_vol", "tot_new_acct", "tot_new_acct_bal", 
                                      "avg_acct_bal", "avg_acct_vol", "tot_cnsmr_cnts"]
                metrics = [m for m in fallback_candidates if m in union.columns]
                if metrics:
                    logger.info(f"Using fallback metrics for {group_name}: {metrics}")
        
        # Check if any metrics available
        available_metrics = [m for m in metrics if m in union.columns]
        
        if not available_metrics and not calculated:
            logger.warning(f"Pruned group: {group_name} (0/{len(metrics)} metrics available)")
            # Try one more fallback - any numeric column
            numeric_cols = union.select_dtypes(include=['number']).columns
            fallback = next((col for col in numeric_cols if 'tot' in col or 'avg' in col), None)
            if fallback:
                available_metrics = [fallback]
                logger.info(f"Using last-resort fallback metric for {group_name}: {fallback}")
            else:
                continue
        
        logger.info(f"Processing group {group_name}: {len(available_metrics)}/{len(metrics)} metrics available")
        
        # Compute calculated metrics if enabled
        if config.logic.compute_calculated_metrics and calculated:
            union = compute_calculated_metrics(union, calculated)
            # Add newly calculated metrics to available list
            for calc_spec in calculated:
                if calc_spec['name'] in union.columns:
                    available_metrics.append(calc_spec['name'])
        
        # Process available metrics
        metrics_processed = set()
        chart_types_generated = set()  # Track which chart types we've generated

        # Special handling for delinquency groups: generate all slides once and skip per-metric loop
        if group_name.lower().startswith('delinquen') and presentation and not ctx.dry_run:
            tier_col = 'score_curr_tier' if 'score_curr_tier' in df_snapshot.columns else 'score_tier'
            tiers_to_use = (
                allowed_tiers
                or df_snapshot.get(tier_col, pd.Series()).astype(str).str.upper().dropna().unique().tolist()
            )
            latest_df = df_snapshot[df_snapshot['period'] == current_period]
            if latest_df is None or len(latest_df) == 0:
                latest_p = str(df_snapshot['period'].max())
                latest_df = df_snapshot[df_snapshot['period'] == latest_p]
            granularity = (family_resolution or {}).get('granularity', 'quarterly')
            series_sets = [
                (['deliq_30_acct_rate', 'deliq_60_acct_rate', 'deliq_90_acct_rate'], 'Account'),
                (
                    [
                        'cnsmr_cnts_w_deliq_bal_30_rate',
                        'cnsmr_cnts_w_deliq_bal_60_rate',
                        'cnsmr_cnts_w_deliq_bal_90_rate',
                    ],
                    'Consumer',
                ),
                (
                    ['deliq_30_acct_bal_rate', 'deliq_60_acct_bal_rate', 'deliq_90_acct_bal_rate'],
                    'Balance',
                ),
            ]
            for cols, label in series_sets:
                avail = [c for c in cols if c in df_trend.columns]
                if not avail:
                    continue
                prepared_trend = prepare_dpd_trend_dataframe(df_trend, 'period', avail)
                if prepared_trend is None or prepared_trend.empty:
                    continue
                series_tag = _delinquency_series_tag(avail)
                series_phrase = f"{series_tag} DPD" if series_tag else "Delinquency"
                base_phrase = " ".join(part for part in [label, series_phrase] if part)
                insight_title = _delinquency_headline_multi(
                    prepared_trend,
                    'period',
                    avail,
                    granularity,
                    label,
                )
                trend_spec = build_dpd_trend_spec(
                    prepared_trend,
                    'period',
                    avail,
                    f"{base_phrase} Rates",
                    config,
                )
                trend_chart_title = f"{base_phrase} Trend"
                add_chart_slide(presentation, insight_title, trend_chart_title, trend_spec, config)
                ctx.add_slide(
                    {
                        'topic': topic,
                        'metric': avail[0],
                        'title': insight_title,
                        'chart_title': trend_chart_title,
                        'subtitle_rendered': bool(
                            getattr(config.features, 'show_subtitle', False) and trend_chart_title
                        ),
                        'variant': 'delinquency_trend',
                        'content_count': 1,
                    }
                )
                coverage_ledger.mark_coverage(topic, group_name, avail[0], 'A2')
                chart_types_generated.add('A2')
                results['slides_created'] += 1

                sev_spec = build_dpd_severity_spec(
                    latest_df, tier_col, avail, tiers_to_use, config
                )
                split_title = _delinquency_split_insight(latest_df, tier_col, avail)
                if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(split_title):
                    split_title = _sanitize_descriptive_title(split_title)
                tier_source = tiers_to_use or (
                    latest_df[tier_col].astype(str).str.upper().dropna().unique().tolist()
                    if tier_col in latest_df.columns
                    else []
                )
                severity_caption = _format_tier_caption(tier_source)
                severity_chart_title = (
                    f"{label} Delinquency Severity by Tier — {severity_caption}"
                )
                add_chart_slide(
                    presentation,
                    split_title,
                    severity_chart_title,
                    sev_spec,
                    config,
                )
                ctx.add_slide(
                    {
                        'topic': topic,
                        'metric': avail[0],
                        'title': split_title,
                        'chart_title': severity_chart_title,
                        'subtitle_rendered': bool(
                            getattr(config.features, 'show_subtitle', False) and severity_chart_title
                        ),
                        'variant': 'delinquency_severity',
                        'content_count': 1,
                    }
                )
                coverage_ledger.mark_coverage(topic, group_name, avail[0], 'A4')
                chart_types_generated.add('A4')
                results['slides_created'] += 1
            ctx.topic_has_image[topic] = False
            continue

        for metric in available_metrics:
            # Skip if already processed
            metric_key = f"{topic}_{group_name}_{metric}"
            if metric_key in metrics_processed:
                continue
            
            # Apply figure selection guardrails
            if len(union) < config.logic.min_periods_for_trend:
                logger.warning(f"Insufficient periods for {metric}: {len(union)} < {config.logic.min_periods_for_trend}")
                continue
            
            # Check product restrictions
            is_allowed, restriction = check_product_restrictions(union, metric, config)
            if not is_allowed:
                coverage_ledger.add_entry(topic, group_name, metric).skip_reason = restriction
                continue
            
            # BEFORE building chart, decide chart_type using a guess, then pick df
            # use a neutral guess so _select_chart_type can do the right thing
            analysis_hint = short_delta if df_delta is not None else 'TRENDS'
            chart_type = _select_chart_type(union, metric, analysis_hint, family_resolution)
            
            # Add sane fallbacks based on group/metric name
            if group_name.lower().startswith("score"):
                # Score distributions always use A3 (composition)
                chart_type = "A3"
            elif not chart_type or chart_type == 'A2':  # If no specific chart or just trend
                if group_name.lower() in ("balances", "originations", "delinquencies"):
                    chart_type = chart_type or "A2"
                elif any(c.endswith(("_qoq_pct", "_mom_pct", "_yoy_pct")) for c in union.columns):
                    # If delta columns exist, prefer A4
                    chart_type = "A4"
                else:
                    chart_type = "A2"  # Default fallback
            
            # Force at least one Trend (A2) per group to ensure coverage
            if 'A2' not in chart_types_generated and 'snapshot' not in str(analysis_hint).lower():
                chart_type = 'A2'
                logger.debug(f"Forcing A2 for {metric} to ensure group has at least one trend chart")
            
            # choose the dataframe to actually plot
            src_df = union
            if chart_type == 'A3':
                src_df = df_snapshot
            elif chart_type == 'A4':
                if df_delta is not None:
                    src_df = df_delta
                else:
                    src_df = union
                    # make sure deltas exist when we fell back to union
                    src_df = compute_missing_deltas(src_df, metric, short_delta,
                                                    (family_resolution or {}).get('granularity','quarterly'),
                                                    min_periods=config.logic.min_data_points)
                    qa_assert_delta_sanity(src_df, metric, short_delta)
                    if (family_resolution or {}).get('use_yoy', False):
                        src_df = compute_missing_deltas(src_df, metric, 'yoy',
                                                        (family_resolution or {}).get('granularity','quarterly'),
                                                        min_periods=config.logic.min_data_points)
                        qa_assert_delta_sanity(src_df, metric, 'yoy')
            
            # Check if rule applicable
            if not rule_applicable(src_df, chart_type, metric, config):
                # Try fallback
                fallback_chart = apply_fallback_strategy(src_df, metric, chart_type, config)
                if fallback_chart:
                    chart_type = fallback_chart
                else:
                    # Force A2 as last resort - we want to visualize everything
                    logger.info(f"Forcing A2 visualization for {metric} despite rules")
                    chart_type = 'A2'
            
            # Generate visualization
            try:
                chart_result = _generate_chart(src_df, metric, chart_type, config, family_resolution)
                
                # Use ensure_triplet helper for consistent handling
                figure, cache_key, png_bytes = ensure_triplet(chart_result)
                
                # Track chart generation vs cache reuse
                if png_bytes is not None:
                    # New chart was generated
                    results['charts_generated'] += 1
                    logger.info(f"Generated new chart {chart_type} for {topic}/{group_name}/{metric}")
                    logger.info(f"Generated visualization: {chart_type} for {topic}/{group_name}/{metric}")  # Token for notebook
                else:
                    # Chart needs to be fetched from cache
                    png_bytes = get_cached_figure(cache_key)
                    if png_bytes:
                        results['figures_cached'] += 1
                        ctx.figures_cached += 1
                        logger.info(f"Using cached chart {chart_type} for {topic}/{group_name}/{metric}")
                        logger.info(f"Generated visualization: {chart_type} for {topic}/{group_name}/{metric} (cached)")  # Token for notebook
                
                # Track figure artifact in context
                figure_data = {
                    'metric': metric,
                    'chart_type': chart_type,
                    'dpi': config.render.figure_dpi,  # Will be actual DPI from figure
                    'has_labels': chart_type == 'A3',  # A3 should always have labels
                    'annotation_count': 0,  # Will be updated by chart function
                    'cache_key': cache_key
                }
                ctx.add_figure(figure_data, chart_type)
                
                # Build data card
                composition_base = None
                if chart_type == 'A3':
                    composition_base = select_composition_base(src_df, metric)
                
                data_card = build_data_card(src_df, metric, chart_type, composition_base=composition_base)
                
                # Generate narrative
                narrative = llm_narrate(data_card, config)
                results['narratives_generated'] += 1
                
                # QC check for avg_acct_per_cnsmr contradictions
                if 'avg_acct_per_cnsmr' in metric.lower():
                    # Look for total columns
                    tot_acct_col = None
                    tot_cnsmr_col = None
                    delta_col = None
                    
                    for col in src_df.columns:
                        if 'tot_acct' in col.lower() or 'total_acct' in col.lower():
                            tot_acct_col = col
                        elif 'tot_cnsmr' in col.lower() or 'total_cnsmr' in col.lower():
                            tot_cnsmr_col = col
                        elif metric in col and any(d in col for d in ['_yoy_pct', '_qoq_pct', '_mom_pct']):
                            delta_col = col
                    
                    if tot_acct_col and tot_cnsmr_col and delta_col:
                        period_col = 'period'  # Define period column name
                        qc_alert = _qc_implied_avg_alert(
                            src_df, period_col, tot_acct_col, tot_cnsmr_col, metric, delta_col
                        )
                        if qc_alert:
                            logger.warning(f"QC Alert: {qc_alert}")
                            # Add alert to narrative bullets
                            if 'bullets' in narrative and isinstance(narrative['bullets'], list):
                                narrative['bullets'].append(qc_alert)
                
                # Build data-bound headline to ensure consistency with chart
                from synthesis_agent.utils import pretty_label, fmt_value
                headline = None
                chart_title = None
                try:
                    if 'period' in src_df.columns and len(src_df) > 0:
                        latest_period = str(src_df['period'].max())
                        latest_row = src_df[src_df['period'] == latest_period]
                        
                        # Update context with latest period if more recent
                        if ctx.latest_period is None or latest_period > ctx.latest_period:
                            ctx.latest_period = latest_period
                        
                        if len(latest_row) > 0 and metric in latest_row.columns:
                            latest_val = latest_row[metric].iloc[0]
                            
                            # Check if this is a delta chart
                            if any(d in str(analysis_hint).lower() for d in ['qoq', 'yoy', 'mom']):
                                delta_type = (family_resolution or {}).get('short_delta', 'yoy')
                                delta_col, _ = select_delta_col(metric, delta_type, src_df)
                                logger.info(
                                    f"[AxisUnits] {metric} uses {delta_col} → {'pp/bps' if is_rate_metric(metric) else '%'} on Y-axis"
                                )
                                if delta_col in src_df.columns and delta_col in latest_row.columns:
                                    delta_val = latest_row[delta_col].iloc[0]
                                    if not pd.isna(delta_val):
                                        sign = '+' if delta_val >= 0 else ''
                                        if is_rate_metric(metric):
                                            unit = 'pp'
                                            if abs(delta_val) < 1.0:
                                                unit = 'bps'
                                                delta_val *= 100
                                        else:
                                            unit = '%'
                                        headline = f"{pretty_label(metric)} {sign}{delta_val:.1f}{unit} {delta_type.upper()}"
                            
                            # If no delta headline, use trend headline
                            if not headline:
                                headline = f"{pretty_label(metric)} at {fmt_value(latest_val, metric)} ({latest_period})"
                        
                        chart_title = None
                        descriptive_only = getattr(config.features, 'descriptive_titles_only', False)
                        if headline and not descriptive_only:
                            chart_title = headline
                            logger.debug(f"Computed data-bound headline: {headline}")
                        if chart_title is None:
                            focus_for_title = focus_upper if (periodic_enabled and focus_upper and is_quarterly) else None
                            chart_title = _descriptive_chart_title(
                                metric,
                                chart_type,
                                short_delta,
                                focus_for_title,
                            )

                except Exception as e:
                    logger.debug(f"Could not generate data-bound headline: {e}")

                if not chart_title:
                    chart_title = _descriptive_chart_title(
                        metric,
                        chart_type,
                        short_delta,
                        None,
                    )

                if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(chart_title):
                    chart_title = _sanitize_descriptive_title(chart_title)
                
                # Track narrative artifact
                ctx.add_narrative({
                    'metric': metric,
                    'title': narrative.get('title', ''),
                    'bullets': narrative.get('bullets', []),
                    'strapline': narrative.get('strapline', '')
                })
                
                # Generate speaker notes
                notes = generate_speaker_notes(data_card, narrative, config)
                
                # Determine footer
                # Determine the analysis type for this chart
                chart_analysis_type = 'TRENDS'
                if chart_type == 'A3':
                    chart_analysis_type = 'SNAPSHOT'
                elif chart_type == 'A4':
                    chart_analysis_type = short_delta.upper()
                
                has_missing_base, _ = check_delta_base_availability(src_df, chart_analysis_type)
                footer_type = get_appropriate_footer(
                    chart_analysis_type,
                    has_missing_base,
                    False,  # sparse_data
                    'utilization' in metric.lower()
                )
                
                # Add slide with deduplication check
                if presentation and not ctx.dry_run:
                    if config.features.chart_engine == 'pptx':
                        insight_title = (narrative.get('title') or '').strip() or 'Key Insight'
                        if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(insight_title):
                            sanitized_title = _sanitize_descriptive_title(insight_title)
                            if sanitized_title:
                                insight_title = sanitized_title
                            else:
                                insight_title = 'Key Insight'
                        tier_col = 'score_curr_tier' if 'score_curr_tier' in df_snapshot.columns else 'score_tier'
                        tiers_to_use = (
                            allowed_tiers
                            or df_snapshot.get(tier_col, pd.Series()).astype(str).str.upper().dropna().unique().tolist()
                        )

                        periodic_cfg = (config.viz_defaults or {}).get('periodic_trend', {}) or {}
                        apply_tags = {str(tag).lower() for tag in periodic_cfg.get('apply_to', [])}
                        group_lower = group_name.lower()
                        modes = analysis_modes or {}
                        focus_q = modes.get('focus_quarter')
                        default_focus = getattr(config.logic, 'default_focus_quarter', None)
                        apply_default = False
                        if any(tag in apply_tags for tag in ('delinquency', 'delinquencies')) and 'delinquen' in group_lower:
                            apply_default = True
                        if not apply_default and any(tag in apply_tags for tag in ('trend', 'trends')):
                            apply_default = True
                        if not focus_q and apply_default and default_focus:
                            focus_q = default_focus
                        periodic_enabled = bool(focus_q) and (
                            bool(modes.get('periodic_trend')) or apply_default
                        )
                        if analysis_modes is not None:
                            if focus_q and not analysis_modes.get('focus_quarter'):
                                analysis_modes['focus_quarter'] = focus_q
                            if periodic_enabled and not analysis_modes.get('periodic_trend'):
                                analysis_modes['periodic_trend'] = True
                        focus_upper = focus_q.upper() if focus_q else None
                        is_quarterly = (
                            (family_resolution or {}).get('granularity', 'quarterly').lower()
                            == 'quarterly'
                        )

                        if group_lower.startswith('delinquen'):
                            # Handle delinquency charts as a special group: build
                            # separate slides for account-, consumer-, and
                            # balance-level views.  Only trigger once on the
                            # first metric (30-day account rate).
                            if metric != 'deliq_30_acct_rate':
                                continue
                            trend_frame = df_trend.copy()
                            if 'period' in trend_frame.columns:
                                trend_frame['period'] = trend_frame['period'].astype(str)
                            per_periods: List[str] = []
                            if (
                                periodic_enabled
                                and focus_upper
                                and is_quarterly
                                and 'period' in trend_frame.columns
                            ):
                                data_periods = trend_frame['period'].astype(str).unique().tolist()
                                per_periods = [
                                    p for p in data_periods if str(p).upper().endswith(focus_upper)
                                ]
                                per_periods = canonical_sort_periods(per_periods)
                                if per_periods:
                                    period_upper = {p.upper() for p in per_periods}
                                    trend_frame = trend_frame[
                                        trend_frame['period'].astype(str).str.upper().isin(period_upper)
                                    ]
                            snapshot_frame = df_snapshot.copy()
                            if 'period' in snapshot_frame.columns:
                                snapshot_frame['period'] = snapshot_frame['period'].astype(str)
                            current_period_str = str(current_period) if current_period is not None else None
                            latest_period = per_periods[-1] if per_periods else current_period_str
                            latest_df = (
                                snapshot_frame[snapshot_frame['period'] == latest_period]
                                if latest_period is not None and 'period' in snapshot_frame.columns
                                else snapshot_frame.head(0)
                            )
                            if latest_df is None or latest_df.empty:
                                # Fallback to latest available snapshot period
                                latest_p = (
                                    snapshot_frame['period'].max()
                                    if 'period' in snapshot_frame.columns and not snapshot_frame.empty
                                    else None
                                )
                                if latest_p is not None:
                                    latest_df = snapshot_frame[snapshot_frame['period'] == latest_p]
                                else:
                                    latest_df = snapshot_frame.head(0)
                            granularity = (family_resolution or {}).get('granularity', 'quarterly')
                            series_sets = [
                                (['deliq_30_acct_rate','deliq_60_acct_rate','deliq_90_acct_rate'], 'Account'),
                                (['cnsmr_cnts_w_deliq_bal_30_rate','cnsmr_cnts_w_deliq_bal_60_rate','cnsmr_cnts_w_deliq_bal_90_rate'], 'Consumer'),
                                (['deliq_30_acct_bal_rate','deliq_60_acct_bal_rate','deliq_90_acct_bal_rate'], 'Balance'),
                            ]
                            for cols, label in series_sets:
                                avail = [c for c in cols if c in trend_frame.columns]
                                if not avail:
                                    continue
                                prepared_trend = prepare_dpd_trend_dataframe(trend_frame, 'period', avail)
                                if prepared_trend is None or prepared_trend.empty:
                                    continue
                                series_tag = _delinquency_series_tag(avail)
                                series_phrase = f"{series_tag} DPD" if series_tag else "Delinquency"
                                base_phrase = " ".join(part for part in [label, series_phrase] if part)
                                delinquency_headline = _delinquency_headline_multi(
                                    prepared_trend,
                                    'period',
                                    avail,
                                    granularity,
                                    label,
                                )
                                if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(delinquency_headline):
                                    delinquency_headline = _sanitize_descriptive_title(delinquency_headline)

                                trend_spec = build_dpd_trend_spec(
                                    prepared_trend,
                                    'period',
                                    avail,
                                    f"{base_phrase} Rates",
                                    config,
                                )
                                trend_chart_title = f"{base_phrase} Trend"
                                trend_card = build_data_card(prepared_trend, avail[0], 'A2')
                                trend_narrative = llm_narrate(trend_card, config)
                                add_chart_slide(
                                    presentation,
                                    delinquency_headline,
                                    trend_chart_title,
                                    trend_spec,
                                    config,
                                )
                                ctx.add_slide({
                                    'topic': topic,
                                    'group': group_name,
                                    'metric': avail[0],
                                    'title': delinquency_headline,
                                    'chart_title': trend_chart_title,
                                    'subtitle_rendered': bool(getattr(config.features, 'show_subtitle', False) and trend_chart_title),
                                    'variant': 'delinquency_trend',
                                    'content_count': 1,
                                })
                                ctx.add_narrative({
                                    'metric': avail[0],
                                    'title': trend_narrative.get('title', ''),
                                    'bullets': trend_narrative.get('bullets', []),
                                    'strapline': trend_narrative.get('strapline', ''),
                                })
                                coverage_ledger.mark_coverage(topic, group_name, avail[0], 'A2')
                                chart_types_generated.add('A2')
                                results['slides_created'] += 1

                                sev_spec = build_dpd_severity_spec(
                                    latest_df, tier_col, avail, tiers_to_use, config
                                )
                                split_title = _delinquency_split_insight(latest_df, tier_col, avail)
                                if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(split_title):
                                    split_title = _sanitize_descriptive_title(split_title)
                                tier_source = tiers_to_use or (
                                    latest_df[tier_col]
                                    .astype(str)
                                    .str.upper()
                                    .dropna()
                                    .unique()
                                    .tolist()
                                    if tier_col in latest_df.columns
                                    else []
                                )
                                severity_caption = _format_tier_caption(tier_source)
                                severity_chart_title = (
                                    f"{label} Delinquency Severity by Tier — {severity_caption}"
                                )
                                severity_card = build_data_card(
                                    latest_df,
                                    avail[0],
                                    'A3',
                                    composition_base='credit_tier',
                                )
                                severity_narrative = llm_narrate(severity_card, config)
                                add_chart_slide(
                                    presentation,
                                    split_title,
                                    severity_chart_title,
                                    sev_spec,
                                    config,
                                )
                                ctx.add_slide({
                                    'topic': topic,
                                    'group': group_name,
                                    'metric': avail[0],
                                    'title': split_title,
                                    'chart_title': severity_chart_title,
                                    'subtitle_rendered': bool(getattr(config.features, 'show_subtitle', False) and severity_chart_title),
                                    'variant': 'delinquency_severity',
                                    'content_count': 1,
                                })
                                ctx.add_narrative({
                                    'metric': avail[0],
                                    'title': severity_narrative.get('title', ''),
                                    'bullets': severity_narrative.get('bullets', []),
                                    'strapline': severity_narrative.get('strapline', ''),
                                })
                                coverage_ledger.mark_coverage(topic, group_name, avail[0], 'A3')
                                chart_types_generated.add('A3')
                                results['slides_created'] += 1
                            ctx.topic_has_image[topic] = False
                            continue
                        else:
                            fq = focus_upper if (periodic_enabled and focus_upper and is_quarterly) else None
                            per_periods: List[str] = []
                            df_per = None
                            if fq and 'period' in df_trend.columns:
                                data_periods = df_trend['period'].astype(str).unique().tolist()
                                per_periods = [p for p in data_periods if str(p).upper().endswith(fq)]
                                per_periods = canonical_sort_periods(per_periods)
                                if per_periods:
                                    df_per = df_trend[df_trend['period'].astype(str).isin(per_periods)]

                            include_baseline = True
                            only_periodic_flag = getattr(config.features, 'only_periodic_line_trends', False)
                            if (
                                only_periodic_flag
                                and df_per is not None
                                and len(df_per) >= config.logic.min_periods_for_trend
                            ):
                                include_baseline = False

                            multi_periods_for_tiers: List[str] = []
                            multi_focus_label: Optional[str] = None

                            if include_baseline:
                                top_spec = build_trend_line_spec(
                                    df_trend, 'period', [metric], chart_title, config
                                )
                                add_chart_slide(presentation, insight_title, chart_title, top_spec, config)
                                ctx.add_slide({
                                    'topic': topic,
                                    'group': group_name,
                                    'metric': metric,
                                    'title': narrative.get('title', ''),
                                    'chart_title': chart_title,
                                    'subtitle_rendered': bool(getattr(config.features, 'show_subtitle', False) and chart_title),
                                    'variant': 'trend_baseline',
                                    'content_count': 1
                                })
                                coverage_ledger.mark_coverage(topic, group_name, metric, 'A2')
                                chart_types_generated.add('A2')
                                results['slides_created'] += 1

                            if (
                                df_per is not None
                                and len(df_per) >= config.logic.min_periods_for_trend
                                and fq
                            ):
                                periodic_chart_title = _descriptive_chart_title(
                                    metric,
                                    'A2',
                                    short_delta,
                                    fq,
                                )
                                per_spec = build_trend_line_spec(
                                    df_per, 'period', [metric], periodic_chart_title, config
                                )
                                add_chart_slide(
                                    presentation,
                                    insight_title,
                                    periodic_chart_title,
                                    per_spec,
                                    config,
                                )
                                ctx.add_slide(
                                    {
                                        'topic': topic,
                                        'group': group_name,
                                        'metric': metric,
                                        'title': narrative.get('title', ''),
                                        'chart_title': periodic_chart_title,
                                        'subtitle_rendered': bool(getattr(config.features, 'show_subtitle', False) and periodic_chart_title),
                                        'variant': 'trend_periodic',
                                        'content_count': 1,
                                    }
                                )
                                coverage_ledger.mark_coverage(topic, group_name, metric, 'A2')
                                chart_types_generated.add('A2')
                                results['slides_created'] += 1
                                multi_periods_for_tiers = list(per_periods)
                                multi_focus_label = fq
                            if (
                                not multi_periods_for_tiers
                                and modes.get('comparison') == 'YOY'
                                and is_quarterly
                                and current_period is not None
                                and 'period' in df_trend.columns
                            ):
                                focus_suffix = _focus_suffix_from_period(current_period)
                                if focus_suffix:
                                    data_periods = df_trend['period'].astype(str).tolist()
                                    yoy_candidates = [
                                        p for p in data_periods if str(p).upper().endswith(focus_suffix)
                                    ]
                                    yoy_candidates = canonical_sort_periods(yoy_candidates)
                                    ordered_periods: List[str] = []
                                    seen_periods: Set[str] = set()
                                    for p in yoy_candidates:
                                        if p not in seen_periods:
                                            ordered_periods.append(p)
                                            seen_periods.add(p)
                                    min_required = max(config.logic.min_periods_for_trend, 3)
                                    if len(ordered_periods) >= min_required:
                                        multi_periods_for_tiers = ordered_periods
                                        multi_focus_label = focus_suffix.upper()

                            if (
                                multi_periods_for_tiers
                                and tier_col
                                and len(df_snapshot) > 0
                            ):
                                snapshot_for_tiers = df_snapshot.copy()
                                if 'period' in snapshot_for_tiers.columns:
                                    snapshot_for_tiers['period'] = snapshot_for_tiers['period'].astype(str)
                                tiers_for_chart = tiers_to_use or getattr(config.logic, 'allowed_tiers', [])
                                is_rate = is_rate_metric(metric)
                                multi_tier_spec = build_multi_period_grouped_by_tier_spec(
                                    df=snapshot_for_tiers,
                                    metric=metric,
                                    period_col='period',
                                    tier_col=tier_col,
                                    tiers=tiers_for_chart or [],
                                    periods=multi_periods_for_tiers,
                                    config=config,
                                    is_rate=is_rate,
                                )
                                focus_words = _quarter_label_to_words(multi_focus_label) if multi_focus_label else 'Year-over-year'
                                subtitle_label = f"{pretty_label(metric)} — {focus_words} tier mix across years"
                                if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(subtitle_label):
                                    subtitle_label = _sanitize_descriptive_title(subtitle_label)
                                add_chart_slide(
                                    presentation,
                                    insight_title,
                                    subtitle_label,
                                    multi_tier_spec,
                                    config,
                                )
                                ctx.add_slide({
                                    'topic': topic,
                                    'group': group_name,
                                    'metric': metric,
                                    'title': narrative.get('title', ''),
                                    'chart_title': subtitle_label,
                                    'subtitle_rendered': bool(getattr(config.features, 'show_subtitle', False) and subtitle_label),
                                    'variant': 'tier_periodic',
                                    'content_count': 1
                                })
                                coverage_ledger.mark_coverage(topic, group_name, metric, 'A4')
                                chart_types_generated.add('A4')
                                results['slides_created'] += 1

                            if comparison_period:
                                bottom_spec = build_compare_grouped_by_tier_spec(
                                    df_snapshot,
                                    metric,
                                    'period',
                                    tier_col or 'score_curr_tier',
                                    tiers_to_use,
                                    current_period,
                                    comparison_period,
                                    config,
                                    is_rate_metric(metric),
                                )
                                add_chart_slide(presentation, insight_title, chart_title, bottom_spec, config)
                                ctx.add_slide({
                                    'topic': topic,
                                    'group': group_name,
                                    'metric': metric,
                                    'title': narrative.get('title', ''),
                                    'chart_title': chart_title,
                                    'subtitle_rendered': bool(getattr(config.features, 'show_subtitle', False) and chart_title),
                                    'variant': 'tier_compare',
                                    'content_count': 1
                                })
                                coverage_ledger.mark_coverage(topic, group_name, metric, 'A4')
                                chart_types_generated.add('A4')
                                results['slides_created'] += 1
                            ctx.topic_has_image[topic] = False
                            continue

                    # Check for duplicate slides using cache key
                    dedupe_key = f"{topic}::{metric}::{chart_type}::{short_delta}::{cache_key}"
                    if cache_key and dedupe_key in ctx.slide_cache_keys:
                        logger.info(f"Skipping duplicate slide for {dedupe_key}")
                        continue

                    # Check for duplicate content using content hash
                    import hashlib
                    import json
                    content_hash = hashlib.sha256(json.dumps({
                        "t": narrative.get('title'),
                        "b": narrative.get('bullets'),
                        "s": narrative.get('strapline'),
                    }, sort_keys=True).encode()).hexdigest()

                    if content_hash in ctx.slide_content_hashes:
                        logger.info(f"Skipping duplicate narrative slide with identical content for {metric}")
                        continue

                    # Use PNG bytes directly or get from cache
                    figure_bytes = png_bytes or get_cached_figure(cache_key)

                    if not figure_bytes:
                        logger.warning(f"Skipping slide for {metric} – no chart image available")
                        continue

                    insight_title = (narrative.get('title') or '').strip() or 'Key Insight'
                    if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(insight_title):
                        sanitized_title = _sanitize_descriptive_title(insight_title)
                        if sanitized_title:
                            insight_title = sanitized_title
                        else:
                            insight_title = 'Key Insight'
                    add_insight_slide(
                        presentation,
                        insight_title,
                        narrative.get('bullets', []),
                        narrative.get('strapline', ''),
                        figure_bytes,
                        footer_type,
                        config,
                        chart_title,
                        notes_text=notes  # Pass the speaker notes we already generated
                    )

                    # Track this slide to prevent duplicates
                    if cache_key:
                        ctx.slide_cache_keys.add(dedupe_key)
                    ctx.slide_content_hashes.add(content_hash)

                    # Log presence of image and mark coverage
                    logger.info(f"Added insight slide with image for {metric}")
                    ctx.topic_has_image[topic] = True
                    chart_type_for_coverage = 'A2' if chart_type == 'A5' else chart_type
                    coverage_ledger.mark_coverage(topic, group_name, metric, chart_type_for_coverage)
                    chart_types_generated.add(chart_type)
                    results['slides_created'] += 1

                    # Track slide artifact
                    ctx.add_slide({
                        'topic': topic,
                        'group': group_name,
                        'metric': metric,
                        'title': narrative.get('title', ''),
                        'chart_title': chart_title,
                        'subtitle_rendered': bool(getattr(config.features, 'show_subtitle', False) and chart_title),
                        'variant': 'insight_image',
                        'content_count': 1  # Single content enforced
                    })

                    # If YoY delta is requested, generate additional slide
                    if config.features.chart_engine != 'pptx' and (family_resolution or {}).get('use_yoy', False) and short_delta != 'yoy':
                        delta_type = 'yoy'
                        src_df = compute_missing_deltas(
                            src_df,
                            metric,
                            delta_type,
                            (family_resolution or {}).get('granularity', 'quarterly'),
                            min_periods=config.logic.min_data_points,
                        )
                        qa_assert_delta_sanity(src_df, metric, delta_type)
                        yoy_chart_type = 'A4'
                        chart_result = _generate_chart(
                            src_df,
                            metric,
                            yoy_chart_type,
                            config,
                            family_resolution,
                            force_delta_type=delta_type,
                        )
                        figure2, cache_key2, png_bytes2 = ensure_triplet(chart_result)
                        if png_bytes2 is not None:
                            results['charts_generated'] += 1
                            logger.info(
                                f"Generated new chart {yoy_chart_type} for {topic}/{group_name}/{metric} (YOY)"
                            )
                        else:
                            png_bytes2 = get_cached_figure(cache_key2)
                            if png_bytes2:
                                results['figures_cached'] += 1
                                ctx.figures_cached += 1
                                logger.info(
                                    f"Using cached chart {yoy_chart_type} for {topic}/{group_name}/{metric} (YOY)"
                                )
                        figure_data = {
                            'metric': metric,
                            'chart_type': yoy_chart_type,
                            'dpi': config.render.figure_dpi,
                            'has_labels': False,
                            'annotation_count': 0,
                            'cache_key': cache_key2
                        }
                        ctx.add_figure(figure_data, yoy_chart_type)

                        data_card2 = build_data_card(src_df, metric, yoy_chart_type)
                        narrative2 = llm_narrate(data_card2, config)

                        # Compute headline for YoY
                        from synthesis_agent.utils import pretty_label, fmt_value
                        headline2 = None
                        chart_title2 = None
                        try:
                            if 'period' in src_df.columns and len(src_df) > 0:
                                latest_period = str(src_df['period'].max())
                                latest_row = src_df[src_df['period'] == latest_period]
                                if ctx.latest_period is None or latest_period > ctx.latest_period:
                                    ctx.latest_period = latest_period
                                if len(latest_row) > 0 and metric in latest_row.columns:
                                    latest_val = latest_row[metric].iloc[0]
                                    delta_col2, _ = select_delta_col(metric, delta_type, src_df)
                                    logger.info(
                                        f"[AxisUnits] {metric} uses {delta_col2} → {'pp/bps' if is_rate_metric(metric) else '%'} on Y-axis"
                                    )
                                    if delta_col2 in src_df.columns and delta_col2 in latest_row.columns:
                                        delta_val = latest_row[delta_col2].iloc[0]
                                        if not pd.isna(delta_val):
                                            sign = '+' if delta_val >= 0 else ''
                                            if is_rate_metric(metric):
                                                unit = 'pp'
                                                if abs(delta_val) < 1.0:
                                                    unit = 'bps'
                                                    delta_val *= 100
                                            else:
                                                unit = '%'
                                            headline2 = f"{pretty_label(metric)} {sign}{delta_val:.1f}{unit} {delta_type.upper()}"
                                    if not headline2:
                                        headline2 = f"{pretty_label(metric)} at {fmt_value(latest_val, metric)} ({latest_period})"
                                chart_title2 = headline2 if headline2 else f"{pretty_label(metric)} — {data_card2.get('period_range', '')}".strip()
                        except Exception as e:
                            logger.debug(f"Could not generate data-bound headline: {e}")
                        if not chart_title2 or getattr(config.features, 'descriptive_titles_only', False):
                            chart_title2 = _descriptive_chart_title(
                                metric,
                                yoy_chart_type,
                                delta_type,
                                focus_upper if 'focus_upper' in locals() else None,
                            )
                        if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(chart_title2):
                            chart_title2 = _sanitize_descriptive_title(chart_title2)

                        chart_analysis_type = delta_type.upper()
                        has_missing_base, _ = check_delta_base_availability(src_df, chart_analysis_type)
                        footer_type2 = get_appropriate_footer(
                            chart_analysis_type,
                            has_missing_base,
                            False,
                            'utilization' in metric.lower(),
                        )

                        if presentation and not ctx.dry_run:
                            dedupe_key2 = f"{topic}::{metric}::{yoy_chart_type}::{delta_type}::{cache_key2}"
                            if cache_key2 and dedupe_key2 in ctx.slide_cache_keys:
                                logger.info(f"Skipping duplicate slide for {dedupe_key2}")
                            else:
                                import hashlib
                                import json
                                content_hash2 = hashlib.sha256(json.dumps({
                                    "t": narrative2.get('title'),
                                    "b": narrative2.get('bullets'),
                                    "s": narrative2.get('strapline'),
                                }, sort_keys=True).encode()).hexdigest()
                                if content_hash2 in ctx.slide_content_hashes:
                                    logger.info(f"Skipping duplicate narrative slide with identical content for {metric}")
                                else:
                                    figure_bytes2 = png_bytes2 or get_cached_figure(cache_key2)
                                    if not figure_bytes2:
                                        logger.warning(f"Skipping YoY slide for {metric} – no chart image available")
                                    else:
                                        insight_title2 = (narrative2.get('title') or '').strip() or 'Key Insight'
                                        if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(insight_title2):
                                            sanitized_title2 = _sanitize_descriptive_title(insight_title2)
                                            if sanitized_title2:
                                                insight_title2 = sanitized_title2
                                            else:
                                                insight_title2 = _descriptive_chart_title(
                                                    metric,
                                                    yoy_chart_type,
                                                    delta_type,
                                                    focus_upper if 'focus_upper' in locals() else None,
                                                )
                                        add_insight_slide(
                                            presentation,
                                            insight_title2,
                                            narrative2.get('bullets', []),
                                            narrative2.get('strapline', ''),
                                            figure_bytes2,
                                            footer_type2,
                                            config,
                                            chart_title2,
                                            notes_text=notes,
                                        )
                                        if cache_key2:
                                            ctx.slide_cache_keys.add(dedupe_key2)
                                        ctx.slide_content_hashes.add(content_hash2)
                                        logger.info(f"Added insight slide with image for {metric} (YOY)")
                                        ctx.topic_has_image[topic] = True
                                        chart_type_for_coverage = 'A2' if yoy_chart_type == 'A5' else yoy_chart_type
                                        coverage_ledger.mark_coverage(
                                            topic, group_name, metric, chart_type_for_coverage
                                        )
                                        chart_types_generated.add(yoy_chart_type)
                                        results['slides_created'] += 1
                                        ctx.add_slide(
                                            {
                                                'topic': topic,
                                                'group': group_name,
                                                'metric': metric,
                                                'title': narrative2.get('title', ''),
                                                'chart_title': chart_title2,
                                                'subtitle_rendered': bool(getattr(config.features, 'show_subtitle', False) and chart_title2),
                                                'variant': 'insight_image_yoy',
                                                'content_count': 1,
                                            }
                                        )
                
                metrics_processed.add(metric_key)
                
            except Exception as e:
                logger.exception(f"Error processing {topic}/{group_name}/{metric}: {e}")
                results['warnings'].append(f"Failed to process {metric}: {str(e)}")
                
                # Continue with a stub narrative even if chart fails
                try:
                    # Create minimal data card
                    data_card = build_data_card(src_df, metric, 'A2')
                    
                    # Generate narrative anyway
                    narrative = llm_narrate(data_card, config)
                    results['narratives_generated'] += 1
                    
                    # Add slide without image (with deduplication check)
                    if presentation and not ctx.dry_run:
                        # Check for duplicate content
                        import hashlib
                        import json
                        content_hash = hashlib.sha256(json.dumps({
                            "t": narrative.get('title', f'{metric} Analysis'),
                            "b": narrative.get('bullets', ['Chart generation failed']),
                            "s": narrative.get('strapline', ''),
                        }, sort_keys=True).encode()).hexdigest()
                        
                        if content_hash not in ctx.slide_content_hashes:
                            chart_title = _descriptive_chart_title(
                                metric,
                                chart_type,
                                (family_resolution or {}).get('short_delta'),
                                None,
                            )
                            if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(chart_title):
                                chart_title = _sanitize_descriptive_title(chart_title)
                            insight_title = (narrative.get('title') or '').strip() or 'Key Insight'
                            if getattr(config.features, 'descriptive_titles_only', False) and _contains_numeric_tokens(insight_title):
                                sanitized_title = _sanitize_descriptive_title(insight_title)
                                if sanitized_title:
                                    insight_title = sanitized_title
                                else:
                                    insight_title = chart_title
                            add_insight_slide(
                                presentation,
                                insight_title,
                                narrative.get('bullets', ['Chart generation failed']),
                                narrative.get('title', ''),
                                None,  # No image
                                'standard',
                                config,
                                chart_title,
                                notes_text=""  # Empty notes for fallback
                            )
                            ctx.slide_content_hashes.add(content_hash)
                            ctx.add_slide({
                                'topic': topic,
                                'group': group_name,
                                'metric': metric,
                                'title': insight_title,
                                'chart_title': chart_title,
                                'subtitle_rendered': bool(getattr(config.features, 'show_subtitle', False) and chart_title),
                                'variant': 'insight_stub',
                                'content_count': 1,
                            })
                        else:
                            logger.info(f"Skipping duplicate narrative-only slide for {metric}")
                        logger.info(f"Added narrative-only slide for {metric} after chart failure")
                        results['slides_created'] += 1
                except Exception as narr_e:
                    logger.error(f"Also failed to generate narrative for {metric}: {narr_e}")
        
        # After processing all metrics, check if we have all chart types
        if available_metrics and not chart_types_generated:
            logger.warning(f"No charts generated for {group_name} despite having {len(available_metrics)} metrics")
        elif available_metrics:
            missing_types = {'A2', 'A3', 'A4'} - chart_types_generated
            if missing_types:
                logger.debug(f"Group {group_name}: Generated {chart_types_generated}, missing {missing_types}")
    
    return results


def _select_chart_type(df, metric: str, analysis_type: str, 
                      family_resolution: Optional[Dict] = None) -> str:
    """Select appropriate chart type for metric, respecting family resolution."""
    # Check for composition data (explicitly share/mix intents)
    tier_cols = extract_tier_columns(df)
    if tier_cols and 'snapshot' in analysis_type:
        tokens = (metric + ' ' + analysis_type).lower()
        comp_keys = ['share', 'mix', 'composition', '% of', 'proportion']
        if any(k in tokens for k in comp_keys):
            return 'A3'  # Composition chart
    elif any(d in analysis_type for d in ['yoy', 'qoq', 'mom']):
        # Apply strict delta family gating
        if family_resolution:
            allowed_deltas = [family_resolution['short_delta']]
            if family_resolution.get('use_yoy', False) and 'yoy' not in allowed_deltas:
                allowed_deltas.append('yoy')
            
            # Check if this delta type is allowed
            delta_found = False
            for delta in allowed_deltas:
                if delta in analysis_type:
                    delta_found = True
                    break
            
            if not delta_found:
                logger.debug(f"Skipping {analysis_type} - not in allowed family {allowed_deltas}")
                return 'A2'  # Fall back to trend
        
        return 'A4'  # Delta
    elif '_bal' in metric and f"{metric.replace('_bal', '_vol')}" in df.columns:
        return 'A5'  # Dual axis
    else:
        return 'A2'  # Trend


def _generate_chart(df, metric: str, chart_type: str, config: SynthesisConfig,
                   family_resolution: Optional[Dict] = None,
                   force_delta_type: Optional[str] = None):
    """Generate chart based on type."""
    period_col = 'period'

    if config.features.chart_engine == 'pptx':
        return {}
    
    if chart_type == 'A2':
        return line_trend_A2(df, metric, period_col, config)
    elif chart_type == 'A3':
        return stacked100_A3(df, metric, period_col, config)
    elif chart_type == 'A4':
        delta_type = force_delta_type or (family_resolution['short_delta'] if family_resolution else 'yoy')
        if force_delta_type is None and delta_type == 'qoq':
            return counts_deltas_A4(df, metric, period_col, config, delta_type=delta_type)
        return delta_over_time_A4(df, metric, period_col, config, delta_type=delta_type)
    elif chart_type == 'A5':
        metric2 = metric.replace('_bal', '_vol') if '_bal' in metric else metric.replace('_vol', '_bal')
        return dual_axis_A5(df, metric, metric2, period_col, config)
    else:
        return line_trend_A2(df, metric, period_col, config)


def run_acceptance_checks(ctx: PipelineContext) -> Tuple[bool, List[str]]:
    """
    Run acceptance checks on generated outputs.
    Inspects actual artifacts, not just config values.
    
    Args:
        ctx: Pipeline context with artifacts
    
    Returns:
        Tuple of (all_passed, list_of_failures)
    """
    failures = []
    config = ctx.config

    if config.features.chart_engine != 'pptx':
        if config.render.figure_dpi != 220:
            failures.append(f"Config: DPI is {config.render.figure_dpi}, expected 220")
        if not config.render.a3_labels_always_on:
            failures.append("Config: A3 labels not always ON")
        if config.render.max_annotations_per_chart > 2:
            failures.append(f"Config: Max annotations is {config.render.max_annotations_per_chart}, expected ≤2")
        if not config.features.single_content_only:
            failures.append("Config: single_content_only not enforced")
        # Check actual figure DPIs
        for i, fig in enumerate(ctx.figures):
            if fig.get('dpi') and fig['dpi'] != 220:
                failures.append(f"Figure {i}: DPI is {fig['dpi']}, expected 220")
        # Check A3 figures have labels ON
        for i, a3_fig in enumerate(ctx.a3_figures):
            if not a3_fig.get('has_labels', False):
                failures.append(f"A3 Figure {i}: Labels not ON")
        # Check annotation counts on figures
        for i, fig in enumerate(ctx.figures):
            annotation_count = fig.get('annotation_count', 0)
            if annotation_count > 2:
                failures.append(f"Figure {i}: {annotation_count} annotations, expected ≤2")
        if not ctx.all_slides_single_content:
            failures.append("Artifacts: Some slides have multiple content areas")
        for topic, has_image in ctx.topic_has_image.items():
            if not has_image:
                failures.append(f"No image slides produced for topic '{topic}'")
        topics_with_images = sum(1 for has_img in ctx.topic_has_image.values() if has_img)
        logger.info(f"Topics with image slides: {topics_with_images}/{len(ctx.topic_has_image)}")
    else:
        if not ctx.all_slides_single_content:
            failures.append("Artifacts: Some slides have multiple content areas")

    from synthesis_agent.config import STYLE_COLORS
    if STYLE_COLORS['SUBPRIME'] != '#E53935':
        failures.append(f"Config: SUBPRIME color is {STYLE_COLORS['SUBPRIME']}, expected #E53935")

    from synthesis_agent.config import ALLOW_HEATMAPS
    if ALLOW_HEATMAPS != False:
        failures.append(f"Config: ALLOW_HEATMAPS is {ALLOW_HEATMAPS}, expected False")

    # Check that we have at least some artifacts
    if ctx.strict_mode:
        if len(ctx.figures) == 0:
            failures.append("Strict mode: No figures generated")
        if len(ctx.slides) == 0:
            failures.append("Strict mode: No slides generated")
        if len(ctx.narratives) == 0:
            failures.append("Strict mode: No narratives generated")

    descriptive_only = getattr(config.features, 'descriptive_titles_only', False)
    if descriptive_only:
        for idx, slide in enumerate(ctx.slides):
            for key in ('title', 'chart_title'):
                value = slide.get(key)
                if value and _contains_numeric_tokens(value):
                    failures.append(
                        f"Slide {idx}: {key} contains numeric tokens while descriptive_titles_only is enabled"
                    )

    if not getattr(config.features, 'show_subtitle', True):
        for idx, slide in enumerate(ctx.slides):
            if slide.get('subtitle_rendered'):
                failures.append(f"Slide {idx}: subtitle rendered despite show_subtitle disabled")

    delinquency_titles: Dict[str, Dict[str, Set[str]]] = {}
    for slide in ctx.slides:
        group_name = str(slide.get('group', '') or '').lower()
        if 'delinquen' not in group_name:
            continue
        metric_name = slide.get('metric') or ''
        variant = slide.get('variant') or ''
        if variant not in {'delinquency_trend', 'delinquency_severity'}:
            continue
        delinquency_titles.setdefault(metric_name, {}).setdefault(variant, set()).add(slide.get('title', ''))

    for metric_name, variants in delinquency_titles.items():
        if {'delinquency_trend', 'delinquency_severity'}.issubset(variants.keys()):
            if variants['delinquency_trend'] & variants['delinquency_severity']:
                failures.append(
                    f"Delinquency titles for metric '{metric_name}' are identical between trend and severity slides"
                )

    all_passed = len(failures) == 0
    return all_passed, failures


def main():
    """Main entry point for synthesis agent."""
    parser = argparse.ArgumentParser(description='Agent 3: Synthesis Pipeline')
    parser.add_argument('--manifest', type=str, default='local_output/manifest.json',
                       help='Path to manifest.json')
    parser.add_argument('--engagement', type=str, default='spec_output/engagement.json',
                       help='Path to engagement.json for request-driven coverage')
    parser.add_argument('--output-dir', type=str, default='synthesis_output',
                       help='Output directory')
    parser.add_argument('--template-dir', type=str, default='templates',
                       help='Template directory')
    parser.add_argument('--client', nargs='+', default='Client',
                       help='Client name (allow spaces)')
    parser.add_argument('--config', type=str, help='Config file (YAML/JSON) or JSON overrides')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--strict', action='store_true', 
                       help='Fail on any spec violation (strict mode)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Test pipeline without generating output files')
    parser.add_argument('--log-level', type=str,
                       choices=['debug', 'info', 'warning', 'error', 'critical'],
                       help='Set logging level')
    
    args = parser.parse_args()
    
    # Normalize client name if provided as list (for names with spaces)
    if isinstance(args.client, list):
        args.client = ' '.join(args.client)
    
    # Initial logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Clear figure cache at the start of each run
    clear_figure_cache()
    
    # Load configuration
    config_overrides = None
    config_path = None
    
    if args.config:
        # Check if it's a file path or JSON string
        if args.config.endswith('.yaml') or args.config.endswith('.yml') or args.config.endswith('.json'):
            config_path = args.config
            logger.info(f"Loading config from file: {config_path}")
        else:
            # Assume it's a JSON string with overrides
            try:
                config_overrides = json.loads(args.config)
                logger.info("Applying config overrides from JSON")
            except json.JSONDecodeError:
                logger.error(f"Invalid config: not a valid file path or JSON string: {args.config}")
                sys.exit(1)
    
    config = load_config(overrides=config_overrides, config_path=config_path)

    # Finalize logging level based on CLI or config
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    if args.verbose:
        level = logging.DEBUG
    elif args.log_level:
        level = log_levels[args.log_level.lower()]
    else:
        level = log_levels.get(getattr(config.runtime, 'log_level', 'info').lower(), logging.INFO)
    logging.getLogger().setLevel(level)

    logger.info("=== Agent 3 Synthesis Pipeline Starting ===")
    logger.info(f"LLM temperature set to {getattr(config.runtime, 'temperature', 'N/A')}")
    
    # Set random seed for reproducibility
    random.seed(config.runtime.random_seed)
    
    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    
    manifest = load_manifest(manifest_path)
    
    # Load engagement spec for request-driven coverage
    engagement_data = None
    requested_metrics = {}
    engagement_path = Path(args.engagement)
    if engagement_path.exists():
        logger.info(f"Loading engagement spec from {engagement_path}")
        with engagement_path.open() as f:
            engagement_raw = json.load(f)
        # Unwrap if nested in 'specification' envelope
        engagement_data = engagement_raw.get('specification', engagement_raw)
        
        # Normalize engagement if enabled
        if config.logic.normalize_engagement:
            engagement_data = normalize_engagement(engagement_data)

        analysis_modes = resolve_analysis_modes(engagement_raw)
        periods_raw = engagement_raw["specification"]["client"]["time_frame"]["interpretation"]["periods"]
        gran = engagement_raw["specification"]["client"]["time_frame"]["interpretation"]["granularity"].lower()

        def _canon(p):
            p = str(p).upper()
            return p.replace("-", "") if "Q" in p else p

        periods = [_canon(p) for p in periods_raw]
        current_period = max(periods) if periods else None
        valid_periods = set(periods)

        comp_mode = analysis_modes.get("comparison")
        comparison_period = None
        if gran == "quarterly":
            if comp_mode == "QOQ":
                comparison_period = prev_quarter(current_period)
            elif comp_mode == "YOY":
                comparison_period = prev_year_same_quarter(current_period)
            if not comparison_period or comparison_period not in valid_periods:
                fallback = prev_year_same_quarter(current_period) if current_period else None
                if fallback in valid_periods:
                    comparison_period = fallback
                    comp_mode = "YOY"
        elif gran == "monthly" and comp_mode == "MOM":
            cp = prev_month(current_period)
            if cp in valid_periods:
                comparison_period = cp
        analysis_modes["comparison"] = comp_mode
        if comparison_period is None and periods:
            sorted_periods = canonical_sort_periods(periods)
            if len(sorted_periods) >= 2:
                comparison_period = sorted_periods[-2]

        
        # Build requested metrics structure (topics already normalized)
        from synthesis_agent.semantics import INSIGHT_GROUP_TO_METRICS
        
        for topic in engagement_data.get('topics', []):
            topic_name = topic.get('name', '')
            requested_metrics[topic_name] = {}
            
            for insight_group in topic.get('insight_groups', []):
                group_name = insight_group.get('name', '')
                # Expand metrics immediately for accurate startup logging
                metrics = []
                if group_name in INSIGHT_GROUP_TO_METRICS:
                    metrics = [m["column"] for m in INSIGHT_GROUP_TO_METRICS[group_name]["required_metrics"]]
                requested_metrics[topic_name][group_name] = metrics
        
        # Extract objectives for strapline tying
        objectives = engagement_data.get('objectives', [])
        config.brand.objectives = objectives  # Store for narrative use
    else:
        logger.warning(f"Engagement spec not found at {engagement_path}, using column-driven coverage")

    allowed_tiers = []
    try:
        raw = engagement_data.get('credit_tiers', []) if engagement_data else []
        canon = {"subprime": "SUBPRIME", "near prime": "NEAR_PRIME", "super prime": "SUPER_PRIME"}
        allowed_tiers = [canon.get(str(t).lower().strip(), str(t).upper()) for t in raw]
    except Exception:
        allowed_tiers = []
    logger.info(f"Allowed tiers from engagement: {allowed_tiers}")
    if allowed_tiers:
        config.logic.allowed_tiers = [t.upper() for t in allowed_tiers]
    
    # Create pipeline context
    ctx = PipelineContext(
        config=config,
        strict_mode=args.strict,
        dry_run=args.dry_run
    )
    
    # Scan for latest period BEFORE creating slides
    ctx.latest_period = _scan_latest_period(manifest)
    logger.info(f"Latest period detected: {ctx.latest_period or 'Not found'}")
    
    # Resolve families (CRITICAL: logs exact line)
    family_resolution = detect_family_resolution(manifest, engagement_data, config)
    
    # Initialize coverage ledger with requested metrics
    coverage_ledger = CoverageLedger(requested_metrics=requested_metrics)
    
    # Log startup matrix (per dev plan requirement)
    log_startup_matrix(engagement_data, requested_metrics)
    
    # Bind to template
    from synthesis_agent.deck import bind_template
    presentation = None
    template_report = {}
    
    if not args.dry_run:
        presentation, template_report = bind_template(
            config.brand.preferred_template,
            args.template_dir,
            config,
            config.brand.fallback_template,
        )
        
        # Log template readiness
        logger.info(f"Template readiness: {json.dumps(template_report, indent=2)}")
        
        # Add cover slide
        add_cover_slide(
            presentation,
            f"{args.client} Portfolio Analysis",
            f"Synthesis Report - {family_resolution['granularity'].title()}",
            config,
            as_of=ctx.latest_period
        )
    else:
        logger.info("DRY RUN mode - skipping template binding and slide creation")
    
    # Group files by topic
    allowed_grans = ['monthly','quarterly']
    chosen = family_resolution['granularity']  # 'quarterly', 'monthly' or 'both'
    
    topics_data = {}
    for file_info in manifest.get('files', []):
        # Filter by resolved family (unless 'both' is chosen)
        if chosen != 'both' and file_info.get('granularity') != chosen:
            continue  # old behavior
        
        topic = file_info.get('topic', 'unknown')
        if topic not in topics_data:
            topics_data[topic] = []
        topics_data[topic].append(file_info)
    
    # Add agenda slide
    topics = list(topics_data.keys())
    if presentation and not args.dry_run:
        add_agenda_slide(presentation, topics, config)
    
    # Process each topic
    all_results = []
    for topic, topic_files in topics_data.items():
        # Get topic spec from normalized engagement
        topic_spec = None
        if engagement_data and engagement_data.get('topics'):
            for t in engagement_data['topics']:
                if t.get('name', '').lower() == topic.lower():
                    topic_spec = t
                    break
        
        results = process_topic(
            topic,
            topic_files,
            config,
            coverage_ledger,
            presentation,
            ctx,
            topic_spec,
            family_resolution,
            allowed_tiers,
            analysis_modes,
            current_period,
            comparison_period,
        )
        all_results.append(results)

    logger.info(f"Sections added for topics: {sorted(ctx.sections_added)}")
    
    # Add appendix
    if presentation and not args.dry_run:
        appendix_content = {
            'data_sources': [args.manifest],
            'methodology': 'Automated synthesis using Agent 3 pipeline',
            'glossary': {
                'YoY': 'Year-over-Year',
                'QoQ': 'Quarter-over-Quarter',
                'MoM': 'Month-over-Month'
            }
        }
        add_appendix_slide(presentation, appendix_content, config)
        
        # Add thank-you slide (per dev plan requirement)
        add_thank_you_slide(presentation, config, client_name=args.client)
    
    # Export deck
    output_path = None
    if not args.dry_run:
        output_name = f"{slugify_client(args.client)}_{family_resolution['granularity']}_synthesis"
        output_path = Path(args.output_dir) / f"{output_name}.pptx"
        export_pptx(presentation, output_path, config)
        
        # Optional PDF export
        if config.features.enable_pdf_export:
            export_pdf(output_path, config=config)
    else:
        logger.info("DRY RUN mode - skipping deck export")
    
    # Print coverage report
    coverage_ledger.print_summary()
    
    # Validate plan consistency
    is_valid, issues = validate_plan_consistency(coverage_ledger, config)
    if not is_valid:
        logger.warning(f"Plan validation issues: {issues}")
    
    # Run acceptance checks (now with context containing artifacts)
    checks_passed, check_failures = run_acceptance_checks(ctx)
    if not checks_passed:
        logger.error(f"Acceptance checks failed: {check_failures}")
        if args.strict:
            logger.error("STRICT mode enabled - exiting with failure")
            sys.exit(1)
    
    # Pairing audit: Check for orphaned content
    logger.info("=== Pairing Audit ===")
    
    # Build sets of metrics that have charts and narratives
    metrics_with_charts = set()
    metrics_with_narratives = set()
    
    for slide in ctx.slides:
        metric = slide.get('metric', '')
        if metric:
            metrics_with_charts.add(metric)
    
    for narrative in ctx.narratives:
        metric = narrative.get('metric', '')
        if metric:
            metrics_with_narratives.add(metric)
    
    # Find orphans
    insights_without_charts = metrics_with_narratives - metrics_with_charts
    charts_without_insights = metrics_with_charts - metrics_with_narratives
    
    if insights_without_charts:
        logger.warning(f"Insights with NO matching chart: {list(insights_without_charts)}")
    else:
        logger.info("✓ All insights have matching charts")
    
    if charts_without_insights:
        logger.warning(f"Charts with NO matching insight: {list(charts_without_insights)}")
    else:
        logger.info("✓ All charts have matching insights")
    
    # Print summary
    total_slides = sum(r['slides_created'] for r in all_results)
    total_charts = sum(r.get('charts_generated', 0) for r in all_results)
    total_cached = sum(r['figures_cached'] for r in all_results)
    
    logger.info(f"""
=== Synthesis Complete ===
Mode: {'DRY RUN' if args.dry_run else 'NORMAL'} {'(STRICT)' if args.strict else ''}
Topics processed: {len(topics_data)}
Slides created: {total_slides}
Charts generated: {total_charts}
Figures cached/reused: {total_cached}
Output: {output_path if output_path else 'N/A (dry run)'}
Family: {family_resolution['granularity']} / {family_resolution['short_delta']}
Artifacts tracked: {len(ctx.figures)} figures, {len(ctx.slides)} slides, {len(ctx.narratives)} narratives
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
