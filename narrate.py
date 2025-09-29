"""
Narrative generation for Agent 3 synthesis pipeline.
Implements data cards, LLM narratives with limits, and QA checks.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from synthesis_agent.config import (
    NARRATIVE_LIMITS, SynthesisConfig
)
from synthesis_agent.utils import (
    fmt_currency, fmt_percent, fmt_delta, setup_logging, pretty_label
)
# Import metric-aware aggregation from charts module
try:
    from synthesis_agent.charts import _aggregate_metric
except ImportError:
    # Fallback if charts module not available
    def _aggregate_metric(df, period_col, metric):
        # Simple fallback aggregation
        return df.groupby(period_col, as_index=False, observed=True)[metric].mean()

logger = setup_logging("narrate")


def build_data_card(df: pd.DataFrame,
                   metric: str,
                   chart_type: str,
                   period_col: str = 'period',
                   composition_base: Optional[str] = None) -> Dict[str, Any]:
    """
    Build data card with facts appropriate for the chart type.
    CRITICAL: A3 charts must focus on tier analysis, A2 charts on trends.
    
    Args:
        df: DataFrame with data
        metric: Metric name
        chart_type: Type of chart (A2, A3, etc.)
        period_col: Period column name
        composition_base: Base period for composition (A3 only)
    
    Returns:
        Data card dictionary with chart-specific analysis
    """
    # Strictly classify visualization type based on chart type
    # This ensures appropriate narrative generation later
    visualization_type = 'generic'
    
    # Map chart_type to standard types for consistent handling
    chart_type_normalized = chart_type.upper() if isinstance(chart_type, str) else ''
    
    if chart_type_normalized in ['A2', 'LINE_TREND_A2']:
        visualization_type = 'trend'
    elif chart_type_normalized in ['A3', 'STACKED100_A3']:
        visualization_type = 'tier_distribution'
    elif chart_type_normalized in ['A4', 'DELTA_OVER_TIME_A4', 'COUNTS_DELTAS_A4']:
        visualization_type = 'delta_comparison'
    elif chart_type_normalized in ['A5', 'DUAL_AXIS_A5']:
        visualization_type = 'trend_with_comparison'
    elif '30_60_90' in str(chart_type_normalized):
        visualization_type = 'delinquency_comparison'
    
    # Log the chart type and visualization type for debugging
    logger.debug(f"Chart type '{chart_type}' mapped to visualization type '{visualization_type}'")
    
    card = {
        'visualization_type': visualization_type,  # Add visualization type
        'metric': metric,
        'chart_type': chart_type,
        'period_range': None,
        'latest_period': None,
        'previous_period': None,
        'earliest_period': None,
        'latest_value': None,
        'earliest_value': None,
        'min_value': None,
        'max_value': None,
        'mean_value': None,
        'trend': None,
        'trend_direction': None,
        'trend_value': None,
        'volatility': None,
        'composition_base': composition_base,  # CRITICAL for A3
        'view_metadata': {},
        'key_facts': []
    }
    
    # Extract period range
    if period_col in df.columns:
        periods = df[period_col].dropna()
        if len(periods) > 0:
            sorted_periods = sorted(set(periods.tolist()))
            earliest_period = sorted_periods[0]
            latest_period = sorted_periods[-1]
            previous_period = sorted_periods[-2] if len(sorted_periods) > 1 else None
            card['period_range'] = f"{earliest_period} to {latest_period}"
            card['earliest_period'] = str(earliest_period)
            card['latest_period'] = str(latest_period)
            if previous_period is not None:
                card['previous_period'] = str(previous_period)
    
    # Extract metric statistics
    if metric in df.columns and period_col in df.columns:
        # Use metric-aware aggregation to match chart display
        # Check if we have duplicate periods (multiple tiers)
        dup = df[period_col].duplicated(keep=False).any()
        if dup:
            # Aggregate using the same logic as charts
            agg_df = _aggregate_metric(df, period_col, metric)
            series = agg_df.set_index(period_col)[metric].dropna()
        else:
            # No duplicates, use the data as-is
            series = df.set_index(period_col)[metric].dropna()
        if len(series) > 0:
            card['latest_value'] = float(series.iloc[-1])
            card['earliest_value'] = float(series.iloc[0])
            card['min_value'] = float(series.min())
            card['max_value'] = float(series.max())
            card['mean_value'] = float(series.mean())

            # Type-aware formatting
            is_rate = ("rate" in metric.lower()) or metric.lower().endswith(("_pct", "_pp"))

            def _fmt_value(v):
                if v is None:
                    return "—"
                if is_rate:
                    return f"{(v*100 if abs(v) <= 1 else v):.1f}%"
                if any(k in metric.lower() for k in ("bal", "amount")):
                    return fmt_currency(v)
                return f"{v:,.0f}"

            card['latest_value_formatted'] = _fmt_value(series.iloc[-1])
            card['earliest_value_formatted'] = _fmt_value(series.iloc[0])
            
            # Calculate trend
            if len(series) > 1:
                latest_val = series.iloc[-1]
                earliest_val = series.iloc[0]
                absolute_change = latest_val - earliest_val if pd.notna(latest_val) and pd.notna(earliest_val) else None
                if is_rate:
                    change = (latest_val - earliest_val)
                    scale_factor = 100.0 if abs(earliest_val) <= 1 else 1.0
                    scaled_change = change * scale_factor if change is not None else None
                    if scaled_change is not None:
                        card['trend'] = (
                            f"{scaled_change:+.1f} bps" if abs(scaled_change) < 1 else f"{scaled_change:+.1f} pp"
                        )
                    card['trend_value'] = scaled_change
                else:
                    base = earliest_val
                    change_pct = ((latest_val / base) - 1) * 100 if base else None
                    card['trend'] = (
                        f"{change_pct:+.1f}%" if change_pct is not None and not pd.isna(change_pct) else None
                    )
                    card['trend_value'] = change_pct

                if absolute_change is not None:
                    if abs(absolute_change) < 1e-9:
                        card['trend_direction'] = 'flat'
                    elif absolute_change > 0:
                        card['trend_direction'] = 'increase'
                    else:
                        card['trend_direction'] = 'decrease'
            
            # Calculate volatility (coefficient of variation)
            if series.std() > 0 and series.mean() > 0:
                card['volatility'] = series.std() / series.mean()
            
            # Add formatted values to key facts for validation
            card['key_facts'].append(f"Latest: {card['latest_value_formatted']}")
            card['key_facts'].append(f"Initial: {card['earliest_value_formatted']}")
            if card.get('trend'):
                card['key_facts'].append(f"Change: {card['trend']}")
    
    # Add chart-specific facts
    if chart_type == 'A3' and composition_base:
        card['key_facts'].append(f"Composition based on {composition_base}")
    
    if chart_type == 'A4':
        delta_snapshot = {}
        delta_cols = [
            c for c in df.columns
            if metric in c and any(d in c for d in ['_yoy_pct', '_yoy_pp', '_qoq_pct', '_qoq_pp', '_mom_pct', '_mom_pp'])
        ]
        if delta_cols:
            latest_idx = df[period_col].dropna().index[-1] if period_col in df.columns and not df[period_col].dropna().empty else -1
            for col in delta_cols:
                latest_delta = df[col].iloc[latest_idx] if latest_idx >= 0 else df[col].iloc[-1]
                if pd.notna(latest_delta):
                    suffix = col.replace(metric + '_', '')
                    delta_snapshot[suffix] = float(latest_delta)
            if delta_snapshot:
                card['view_metadata']['deltas'] = delta_snapshot
                # Prefer YoY/QoQ for headline fact ordering
                for key in sorted(delta_snapshot.keys(), key=lambda x: (0 if 'yoy' in x else 1 if 'qoq' in x else 2, x)):
                    card['key_facts'].append(f"Latest change ({key}): {fmt_delta(delta_snapshot[key])}")
    
    # Add tier information if present
    from synthesis_agent.io_normalize import extract_tier_columns
    tier_cols = extract_tier_columns(df)
    tier_snapshot = {}

    if tier_cols:
        # Wide format with tier columns already present
        latest_idx = df[period_col].dropna().index[-1] if period_col in df.columns and not df[period_col].dropna().empty else -1
        for tier in tier_cols:
            if tier in df.columns:
                value = df.loc[latest_idx, tier] if latest_idx >= 0 else df[tier].iloc[-1]
                if pd.notna(value):
                    tier_snapshot[tier] = float(value)
    else:
        # Long format – look for explicit tier column and aggregate latest period
        for candidate in ('score_curr_tier', 'score_tier', 'score_band', 'tier'):
            if candidate in df.columns:
                if period_col in df.columns and card.get('latest_period'):
                    latest_rows = df[df[period_col].astype(str) == str(card['latest_period'])]
                else:
                    latest_rows = df
                if metric in df.columns:
                    grouped = latest_rows.groupby(candidate)[metric].sum(min_count=1)
                    tier_snapshot = {
                        str(k).upper(): float(v)
                        for k, v in grouped.dropna().items()
                    }
                break

    if tier_snapshot:
        prev_snapshot = {}
        previous_period = card.get('previous_period')
        if previous_period:
            if tier_cols:
                prev_row = df[df[period_col].astype(str) == str(previous_period)] if period_col in df.columns else pd.DataFrame()
                if not prev_row.empty:
                    for tier in tier_cols:
                        if tier in prev_row.columns and pd.notna(prev_row.iloc[0][tier]):
                            prev_snapshot[tier] = float(prev_row.iloc[0][tier])
            else:
                for candidate in ('score_curr_tier', 'score_tier', 'score_band', 'tier'):
                    if candidate in df.columns and metric in df.columns:
                        prev_rows = df[df[period_col].astype(str) == str(previous_period)] if period_col in df.columns else df
                        grouped_prev = prev_rows.groupby(candidate)[metric].sum(min_count=1)
                        prev_snapshot = {
                            str(k).upper(): float(v)
                            for k, v in grouped_prev.dropna().items()
                        }
                        break

        sorted_tiers = sorted(tier_snapshot.items(), key=lambda kv: kv[1], reverse=True)
        dominant_tier = sorted_tiers[0][0]
        dominant_value = sorted_tiers[0][1]

        tier_changes = []
        tier_shifts = {}
        tier_rankings = {}
        
        # Create tier analysis dictionary for tier distribution charts
        tier_analysis = {
            'tier_shares': tier_snapshot,
            'dominant_tier': dominant_tier,
            'dominant_tier_value': dominant_value,
            'tier_order': [t[0] for t in sorted_tiers]
        }
        
        if prev_snapshot:
            for tier, latest_val in tier_snapshot.items():
                prev_val = prev_snapshot.get(tier)
                if prev_val is not None:
                    tier_changes.append((tier, latest_val - prev_val))

        card['view_metadata']['tiers'] = {
            'latest': sorted_tiers,
            'previous': sorted(prev_snapshot.items(), key=lambda kv: kv[1], reverse=True) if prev_snapshot else None,
            'changes': tier_changes if tier_changes else None,
            'dominant': dominant_tier
        }

        card['key_facts'].append(f"Dominant tier: {dominant_tier} ({dominant_value:.1f})")
        
        # Add tier analysis to card for tier distribution charts
        if card.get('visualization_type') == 'tier_distribution':
            card['tier_analysis'] = tier_analysis
            
            # Add tier shifts if available
            if tier_shifts:
                card['tier_analysis']['tier_shifts'] = tier_shifts
                
            # Add tier rankings if available
            if tier_rankings:
                card['tier_analysis']['tier_rankings'] = tier_rankings
    
    logger.info(f"Built data card for {metric}/{chart_type} with {len(card['key_facts'])} facts")
    return card


def llm_narrate(data_card: Dict[str, Any],
               config: SynthesisConfig,
               narrative_type: str = 'insight') -> Dict[str, str]:
    """
    Generate LLM narrative with strict limits based on chart type.
    CRITICAL: Enforce character/bullet limits and re-prompt on violation.
    
    Args:
        data_card: Data card with facts
        config: Synthesis configuration
        narrative_type: Type of narrative (insight, summary, notes)
    
    Returns:
        Dictionary with title, bullets, and strapline
    """
    # Check narrative engine configuration
    engine = getattr(config.features, 'narrative_engine', 'vertex')
    
    if engine == 'deterministic':
        # Use deterministic, data-bound narrative (not a fallback)
        return stub_llm_narrative(data_card, config, narrative_type)
    elif engine != 'vertex':
        raise ValueError(f"Unknown narrative_engine: {engine}")
    
    # Prepare prompt based on narrative type
    if narrative_type == 'insight':
        prompt = _build_insight_prompt(data_card, config)
    elif narrative_type == 'summary':
        prompt = _build_summary_prompt(data_card, config)
    else:
        prompt = _build_notes_prompt(data_card, config)
    
    # Determine visualization type for this narrative
    viz_type = data_card.get('visualization_type', 'generic')
    
    # Call LLM (Gemini Flash via Vertex AI)
    llm_used = True
    try:
        narrative = _call_llm(prompt, config)
        # Include visualization type in the narrative metadata
        if isinstance(narrative, dict):
            narrative['viz_type'] = viz_type
            
        if isinstance(narrative, dict) and "title" in narrative and "headline" in narrative:
            narrative["insight_title"] = narrative["title"]
            narrative["metric_headline"] = narrative["headline"]
    except Exception as e:
        llm_used = False
        logger.warning(
            "Falling back to deterministic narrative for %s/%s due to %s",
            data_card.get('metric'),
            data_card.get('chart_type'),
            e,
        )
        narrative = stub_llm_narrative(data_card, config, narrative_type)

    # Validate and enforce limits (including numeric guard)
    validated = _validate_narrative(narrative, config, data_card)
    
    # Re-prompt if validation failed (only when LLM path succeeded)
    retry_count = 0
    while llm_used and not validated['is_valid'] and retry_count < config.runtime.max_retries:
        logger.warning(f"Narrative validation failed: {validated['issues']}")
        
        # Add correction instructions to prompt
        correction_prompt = prompt + f"\n\nCORRECTION REQUIRED:\n" + "\n".join(validated['issues'])
        correction_prompt += "\n\nIMPORTANT: Only use numbers that appear in the data card facts provided."
        narrative = _call_llm(correction_prompt, config)
        validated = _validate_narrative(narrative, config, data_card)
        retry_count += 1
    
    if not validated['is_valid']:
        logger.error(f"Failed to generate valid narrative after {retry_count} retries")
        # Fail with clear error instead of using fallback
        raise RuntimeError(f"LLM returned invalid narrative after {retry_count} retries: {validated['issues']}")
    
    # Sanitize LLM output to remove debug/reasoning text
    def _strip_reasoning(s: str) -> str:
        """Remove reasoning and debug text from LLM output."""
        if not s:
            return s
        for bad in ("Reasoning:", "Let's think", "Chain of thought", "Step 1:", "Step 2:", 
                   "First,", "Second,", "Third,", "Analysis:", "Note:"):
            s = s.replace(bad, "")
        return s.strip()
    
    # Remove unexpected debug keys from narrative
    for k in list(narrative.keys()):
        if k not in ('title', 'bullets', 'strapline', 'speaker_notes'):
            narrative.pop(k, None)
            logger.debug(f"Removed unexpected key '{k}' from narrative")
    
    # Clean reasoning text from string fields
    for k in ('title', 'strapline'):
        if k in narrative and isinstance(narrative[k], str):
            narrative[k] = _strip_reasoning(narrative[k])
    
    # Clean bullets list
    if 'bullets' in narrative and isinstance(narrative['bullets'], list):
        narrative['bullets'] = [_strip_reasoning(b) if isinstance(b, str) else b 
                               for b in narrative['bullets']]
    
    return narrative


def narrative_qc(narratives: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Quality check narratives with reordering and contradiction scan.
    
    Args:
        narratives: List of narrative dictionaries
    
    Returns:
        QC'd and reordered narratives
    """
    # Reorder by importance (based on metric type)
    reordered = _reorder_narratives(narratives)
    
    # Scan for contradictions
    contradictions = _scan_contradictions(reordered)
    
    if contradictions:
        logger.warning(f"Found {len(contradictions)} potential contradictions")
        # Flag contradictions for review
        for i, j, reason in contradictions:
            reordered[i]['qc_flag'] = f"Potential contradiction with item {j}: {reason}"
            reordered[j]['qc_flag'] = f"Potential contradiction with item {i}: {reason}"
    
    # Check for duplicate insights
    seen_insights = set()
    for narrative in reordered:
        key_insight = narrative.get('strapline', '')
        if key_insight in seen_insights:
            narrative['qc_flag'] = "Duplicate insight"
        seen_insights.add(key_insight)
    
    logger.info(f"QC completed on {len(narratives)} narratives")
    return reordered


def generate_speaker_notes(data_card: Dict[str, Any],
                          narrative: Dict[str, str],
                          config: SynthesisConfig) -> str:
    """
    Generate speaker notes with 2-3 bullets.
    
    Args:
        data_card: Data card with facts
        narrative: Generated narrative
        config: Synthesis configuration
    
    Returns:
        Speaker notes text
    """
    notes_bullets = []
    
    # Add key metric insight
    if data_card.get('trend'):
        notes_bullets.append(f"Trend shows {data_card['trend']} change over the period")
    
    # Add volatility insight if significant
    if data_card.get('volatility') and data_card['volatility'] > 0.2:
        notes_bullets.append(f"Note the high volatility (CV={data_card['volatility']:.2f})")
    
    # Add composition insight if A3
    if data_card.get('composition_base'):
        notes_bullets.append(f"Composition analysis based on {data_card['composition_base']}")
    
    # Limit to configured number of bullets
    max_bullets = NARRATIVE_LIMITS.get('speaker_notes_bullets', 3)
    notes_bullets = notes_bullets[:max_bullets]
    
    # Format as bullet points
    notes = "\n".join([f"• {bullet}" for bullet in notes_bullets])
    
    return notes


# Helper functions

def _build_insight_prompt(data_card: Dict[str, Any], config: SynthesisConfig) -> str:
    """Build prompt for insight narrative based on chart type."""
    limits = NARRATIVE_LIMITS
    
    # Check visualization type to route to specialized prompt builder
    viz_type = data_card.get('visualization_type', 'generic')
    chart_type = data_card.get('chart_type', 'unknown')
    
    # Log chart type and viz type to help with debugging
    logger.debug(f"Building insight prompt for chart_type={chart_type}, viz_type={viz_type}")
    
    # Route to specialized prompt builders based on visualization type
    if viz_type == 'tier_distribution':
        return _build_tier_insight_prompt(data_card, config)
    elif viz_type == 'trend':
        return _build_trend_insight_prompt(data_card, config)
    elif viz_type == 'delta_comparison':
        return _build_delta_insight_prompt(data_card, config)
    
    # Default generic prompt if no specific visualization type
    # Use formatted values if available, fallback to raw
    latest_value = data_card.get('latest_value_formatted', data_card.get('latest_value', 'N/A'))
    earliest_value = data_card.get('earliest_value_formatted', data_card.get('earliest_value', 'N/A'))
    
    prompt = f"""
Generate an executive insight for {data_card['metric']} with the following constraints:

STRICT REQUIREMENTS:
- Title: Maximum {limits['title_max_chars']} characters
- Bullets: Exactly {limits['bullet_min']}-{limits['bullet_max']} bullets, NO trailing periods
- Strapline: MUST BE UNDER {limits['strapline_max_chars']} CHARACTERS (currently max 140)
- Use ONLY the exact numbers provided below - do not calculate or format new numbers

DATA CARD:
Metric: {data_card['metric']}
Chart Type: {data_card['chart_type']}
Period: {data_card['period_range']}
Latest Value: {latest_value}
Initial Value: {earliest_value}
Trend: {data_card.get('trend', 'N/A')}
Key Facts: {json.dumps(data_card['key_facts'])}

IMPORTANT: 
- The strapline MUST be concise (under 140 characters)
- Use the formatted values provided (e.g., "302.5B" not "302450048944")
- Reference periods as quarters (e.g., "Q4 2022" from "2022Q4")

FORMAT:
{{
  "title": "...",
  "bullets": ["bullet1", "bullet2", "bullet3"],
  "strapline": "concise key insight under 140 chars"
}}
"""

    if data_card.get('composition_base'):
        prompt += (
            f"\nIMPORTANT: Composition based on {data_card['composition_base']}"
            "\nTITLE MUST identify the top two or three tiers by share and state whether each rose, fell, or held steady versus the comparison period."
            " Use concise tier names such as 'Prime', 'Near-Prime', 'Subprime'."
        )

    metric_lower = str(data_card.get('metric', '')).lower()
    if any(token in metric_lower for token in ('deliq_30', 'deliq_60', 'deliq_90')):
        prompt += (
            "\nTITLE REQUIREMENT: Summarize 30+, 60+, and 90+ delinquency in one line and indicate whether each bucket is rising,"
            " falling, or stable (use ↑, ↓, or ↔ when space is tight)."
        )

    return prompt


def _build_tier_insight_prompt(data_card: Dict[str, Any], config: SynthesisConfig) -> str:
    """Build specialized prompt for tier distribution chart insights."""
    limits = NARRATIVE_LIMITS
    metric_name = pretty_label(data_card.get('metric', 'metric'))
    
    # Extract tier-specific data if available
    tier_analysis = data_card.get('tier_analysis', {})
    dominant_tier = tier_analysis.get('dominant_tier', 'Unknown')
    tier_shares = tier_analysis.get('tier_shares', {})
    
    # Get tier breakdown if available
    tier_breakdown = ""
    if tier_shares:
        tier_breakdown = "Tier breakdown:\n"
        for tier, share in tier_shares.items():
            tier_breakdown += f"- {tier}: {share:.1f}%\n"
    
    prompt = f"""
    Generate an executive insight SPECIFICALLY ABOUT THE CREDIT TIER DISTRIBUTION for {metric_name}.

    CRITICAL REQUIREMENTS:
    - ALWAYS begin title with "Credit Tier Analysis:" or "Tier Distribution:" 
    - EVERY bullet point MUST mention specific tiers (SUPER_PRIME, PRIME, NEAR_PRIME, SUBPRIME, etc.)
    - Focus EXCLUSIVELY on the tier distribution - DO NOT discuss overall trends
    - Highlight the dominant tier ({dominant_tier}) and explain its significance
    - Compare proportions across different tiers and note which are growing/shrinking
    - Note any risk concentration in specific tiers
    - Be specific with tier percentages when possible
    - Keep the title under {limits.get('title_max_chars', 80)} characters
    - Include 2-4 bullet points, each under {limits.get('bullet_max_chars', 120)} characters
    - Keep the strapline under {limits.get('strapline_max_chars', 100)} characters
    
    {tier_breakdown}
    
    DATA FACTS:
    {data_card['key_facts']}
    
    OUTPUT FORMAT:
    {{
        "title": "Credit Tier Analysis: [specific insight about tier mix]",
        "strapline": "One-sentence highlighting tier-specific findings",
        "bullets": [
            "SUPER_PRIME accounts for X% of [metric], indicating...",
            "SUBPRIME tier shows significant change of X%, suggesting...",
            "Distribution across tiers reveals concentration in X tier..."
        ]
    }}
    """
    
    return prompt


def _build_trend_insight_prompt(data_card: Dict[str, Any], config: SynthesisConfig) -> str:
    """Build specialized prompt for trend chart insights."""
    limits = NARRATIVE_LIMITS
    metric_name = pretty_label(data_card.get('metric', 'metric'))
    
    # Extract trend-specific data
    latest_value = data_card.get('latest_value_formatted', data_card.get('latest_value', 'N/A'))
    earliest_value = data_card.get('earliest_value_formatted', data_card.get('earliest_value', 'N/A'))
    change_pct = data_card.get('change_pct_formatted', data_card.get('change_pct', 'N/A'))
    
    prompt = f"""
    Generate an executive insight about the OVERALL TREND for {metric_name}.

    CRITICAL REQUIREMENTS:
    - ALWAYS begin title with "Trend Analysis:" or "Performance Trend:" 
    - NEVER mention credit tiers (SUPER_PRIME, PRIME, NEAR_PRIME, SUBPRIME) - this is an aggregate view only
    - Focus EXCLUSIVELY on the time-series trend of the TOTAL metric across all periods
    - Highlight key turning points, acceleration, or deceleration in the trend
    - Identify any seasonality or cyclical patterns if evident
    - Compare the latest value ({latest_value}) with the earliest value ({earliest_value})
    - Note the overall change ({change_pct}) and its business significance
    - Keep the title under {limits.get('title_max_chars', 80)} characters
    - Include 2-4 bullet points, each under {limits.get('bullet_max_chars', 120)} characters
    - Keep the strapline under {limits.get('strapline_max_chars', 100)} characters
    
    DATA FACTS:
    {data_card['key_facts']}
    
    OUTPUT FORMAT:
    {{
        "title": "Trend Analysis: [specific insight about overall trend]",
        "strapline": "One-sentence summary capturing the trend direction and magnitude",
        "bullets": [
            "Overall trend shows [direction] of [X%] from [start period] to [end period]",
            "Notable [acceleration/deceleration/inflection] observed during [specific period]",
            "Business implication of this trend suggests [strategic insight]"
        ]
    }}
    """
    
    return prompt


def _build_delta_insight_prompt(data_card: Dict[str, Any], config: SynthesisConfig) -> str:
    """Build specialized prompt for delta comparison chart insights."""
    limits = NARRATIVE_LIMITS
    metric_name = pretty_label(data_card.get('metric', 'metric'))
    
    # Extract delta-specific information
    delta_type = data_card.get('delta_type', 'YOY').upper()
    delta_value = data_card.get('delta_formatted', data_card.get('delta', 'N/A'))
    
    prompt = f"""
    Generate an executive insight about the {delta_type} CHANGE in {metric_name}.

    CRITICAL REQUIREMENTS:
    - ALWAYS begin title with "{delta_type} Change:" or "{delta_type} Comparison:" 
    - This chart shows CHANGE between periods, not trends or tier breakdowns
    - Focus EXCLUSIVELY on period-over-period changes and their significance
    - Highlight the magnitude and direction of the most recent change ({delta_value})
    - Compare recent changes to historical patterns or expectations
    - Note any acceleration or deceleration in the rate of change
    - Analyze what these changes reveal about portfolio performance
    - Keep the title under {limits.get('title_max_chars', 80)} characters
    - Include 2-4 bullet points, each under {limits.get('bullet_max_chars', 120)} characters
    - Keep the strapline under {limits.get('strapline_max_chars', 100)} characters
    
    DATA FACTS:
    {data_card['key_facts']}
    
    OUTPUT FORMAT:
    {{
        "title": "{delta_type} Change: [specific insight about period-over-period change]",
        "strapline": "One-sentence summary about the significance of the observed changes",
        "bullets": [
            "{delta_type} change of [specific value/percentage] represents [acceleration/deceleration]",
            "Changes are [consistent/inconsistent/volatile] compared to historical patterns",
            "These changes suggest [specific business implication or forward-looking insight]"
        ]
    }}
    """
    
    return prompt


def _build_summary_prompt(data_card: Dict[str, Any], config: SynthesisConfig) -> str:
    """Build prompt for summary narrative."""

    return f"""
Summarize the key findings for {data_card['metric']}:

Data: {json.dumps(data_card, default=str)}

Provide a 2-3 sentence summary focusing on the most important trend or change.
"""


def _build_notes_prompt(data_card: Dict[str, Any], config: SynthesisConfig) -> str:
    """Build prompt for speaker notes."""
    return f"""
Generate {NARRATIVE_LIMITS['speaker_notes_bullets']} speaker note bullets for {data_card['metric']}:

Key points to cover:
- Main trend or change
- Any notable patterns
- Business implications

Keep each bullet concise and actionable.
"""


def generate_narrative(data_card: Dict[str, Any], config: SynthesisConfig, 
                       narrative_type: str = 'insight', llm_func=None) -> Dict[str, str]:
    """
    Generate narrative with pluggable LLM backend.
    
    Args:
        data_card: Data card with facts
        config: Synthesis configuration
        narrative_type: Type of narrative (insight, summary, notes)
        llm_func: Optional LLM function to use (defaults to llm_narrate)
    
    Returns:
        Dictionary with title, bullets, and strapline
    """
    if llm_func is None:
        # Use default LLM narrative function
        return llm_narrate(data_card, config, narrative_type)
    else:
        # Use provided LLM function (for testing/stubbing)
        return llm_func(data_card, config, narrative_type)


def _call_llm(prompt: str, config: SynthesisConfig) -> Dict[str, str]:
    """
    Call LLM (Gemini Flash) for narrative generation.
    """
    try:
        # Try to import Vertex AI SDK
        from vertexai.generative_models import GenerativeModel
        import vertexai
        
        # Initialize Vertex AI
        vertexai.init(
            project=config.runtime.project_id,
            location=config.runtime.location
        )
        
        # Create model instance
        model = GenerativeModel(config.runtime.generative_model_name)
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": getattr(config.runtime, "temperature", 0.6),
                "max_output_tokens": config.runtime.max_output_tokens,
                "response_mime_type": "application/json"
            }
        )
        
        # Parse JSON response
        import json
        result = json.loads(response.text)
        
        # Validate structure
        if not all(k in result for k in ['title', 'bullets', 'strapline']):
            raise ValueError("Invalid response structure from LLM")
        
        return result
        
    except ImportError as exc:
        logger.warning("Vertex AI SDK not available: %s", exc)
        raise RuntimeError("vertex_sdk_unavailable") from exc
        
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        # Fail fast - do not return generic boilerplate
        raise RuntimeError(f"LLM narrative generation failed: {e}")


def _extract_numbers_from_text(text: str) -> List[str]:
    """Extract all numeric values from text."""
    # Pattern to match numbers with various formats
    patterns = [
        r'\d+\.?\d*[KMBTkm]?',  # Numbers with suffixes
        r'\d+,\d{3}(?:,\d{3})*',  # Numbers with commas
        r'\d+\.?\d*%',  # Percentages
        r'\d+\.?\d*pp',  # Percentage points
        r'\d+\.?\d*bps',  # Basis points
        r'[\+\-−]\d+\.?\d*',  # Numbers with signs
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.extend(matches)
    
    # Also get plain numbers
    plain_numbers = re.findall(r'\b\d+\.?\d*\b', text)
    numbers.extend(plain_numbers)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_numbers = []
    for num in numbers:
        # Normalize for comparison (remove commas, convert K/M/B)
        normalized = num.replace(',', '').replace('−', '-')
        if normalized not in seen:
            seen.add(normalized)
            unique_numbers.append(num)
    
    return unique_numbers


def _verify_numbers_against_data_card(numbers: List[str], data_card: Dict[str, Any]) -> List[str]:
    """Verify that numbers in narrative appear in data card."""
    violations = []
    
    # Extract all numbers from data card
    card_numbers = set()
    
    # Get numbers from key facts
    for fact in data_card.get('key_facts', []):
        fact_numbers = _extract_numbers_from_text(fact)
        card_numbers.update(fact_numbers)
    
    # Get numbers from formatted fields (these are what LLM should use)
    for field in ['latest_value_formatted', 'earliest_value_formatted', 'trend']:
        if field in data_card and data_card[field]:
            value_str = str(data_card[field])
            card_numbers.add(value_str)
            # Extract numbers from formatted strings
            nums = _extract_numbers_from_text(value_str)
            card_numbers.update(nums)
    
    # Extract years/quarters from period_range
    if 'period_range' in data_card and data_card['period_range']:
        period_str = str(data_card['period_range'])
        # Extract years and quarters
        import re
        years = re.findall(r'\b(20\d{2})\b', period_str)
        card_numbers.update(years)
        quarters = re.findall(r'\b(Q[1-4])\b', period_str)
        card_numbers.update(quarters)
        # Also add the full period string
        card_numbers.add(period_str)
    
    # Get numbers from other fields
    for field in ['latest_value', 'min_value', 'max_value', 'change_pct', 'volatility', 'trend_value']:
        if field in data_card and data_card[field]:
            value_str = str(data_card[field])
            card_numbers.add(value_str)
            # Also add formatted versions
            if isinstance(data_card[field], (int, float)):
                card_numbers.add(f"{data_card[field]:.1f}")
                card_numbers.add(f"{data_card[field]:.2f}")
                card_numbers.add(f"{int(data_card[field])}")
    
    # Normalize card numbers for comparison
    normalized_card = set()
    for num in card_numbers:
        normalized = num.replace(',', '').replace('−', '-')
        normalized_card.add(normalized)
        # Also add without % or other suffixes for matching
        base = re.sub(r'[%KMBTpp\s]', '', normalized)
        if base:
            normalized_card.add(base)
    
    # Check each number in narrative
    for num in numbers:
        normalized_num = num.replace(',', '').replace('−', '-')
        base_num = re.sub(r'[%KMBTpp\s]', '', normalized_num)
        
        # Skip very small numbers (like 1, 2, 3 which might be bullet counts)
        if base_num.isdigit() and int(base_num) <= 3:
            continue
        
        # Check if number appears in data card
        found = False
        for card_val in normalized_card:
            if normalized_num in card_val or base_num in card_val:
                found = True
                break
        
        if not found:
            violations.append(f"Number '{num}' not found in data card")
    
    return violations


def stub_llm_narrative(data_card: Dict[str, Any], config: SynthesisConfig,
                      narrative_type: str = 'insight') -> Dict[str, str]:
    """Deterministic narrative that mirrors LLM output when Vertex AI is unavailable."""

    if narrative_type != 'insight':
        # Preserve legacy behaviour for summaries/notes
        metric_name = pretty_label(data_card.get('metric', 'Metric'))
        period = data_card.get('period_range') or 'latest period'
        return {
            'title': f"{metric_name} Overview"[:NARRATIVE_LIMITS['title_max_chars']],
            'bullets': [f"Coverage spans {period}", "Deterministic narrative mode", "Refer to charts for detail"][:3],
            'strapline': f"{metric_name} insight generated offline"[:NARRATIVE_LIMITS['strapline_max_chars']]
        }

    return _deterministic_insight(data_card)


def _deterministic_insight(data_card: Dict[str, Any]) -> Dict[str, str]:
    chart_type = (data_card.get('chart_type') or '').upper()
    if chart_type == 'A3':
        return _build_tier_insight(data_card)
    if chart_type == 'A4':
        return _build_delta_insight(data_card)
    return _build_trend_insight(data_card)


def _build_trend_insight(data_card: Dict[str, Any]) -> Dict[str, str]:
    metric_name = pretty_label(data_card.get('metric', 'Metric'))
    latest_period = _humanize_period(data_card.get('latest_period'))
    period_range = data_card.get('period_range')
    direction = data_card.get('trend_direction') or 'flat'
    direction_map = {
        'increase': 'rises',
        'decrease': 'declines',
        'flat': 'holds steady'
    }
    verb = direction_map.get(direction, 'trend update')

    if latest_period and verb != 'trend update':
        title = f"{metric_name} {verb} into {latest_period}"
    elif latest_period:
        title = f"{metric_name} trend update for {latest_period}"
    else:
        title = f"{metric_name} performance update"

    latest_value = data_card.get('latest_value_formatted') or _format_numeric(data_card.get('latest_value'), data_card.get('metric'))
    earliest_period = _humanize_period(data_card.get('earliest_period'))
    earliest_value = data_card.get('earliest_value_formatted') or _format_numeric(data_card.get('earliest_value'), data_card.get('metric'))
    trend_text = data_card.get('trend')
    bullets = []

    if latest_period and latest_value:
        bullets.append(f"{latest_period}: {latest_value}")
    if trend_text:
        bullets.append(f"Change vs start: {trend_text}")
    elif direction != 'flat':
        bullets.append(f"Direction: {direction.title()}")
    if earliest_period and earliest_value and earliest_period != latest_period:
        bullets.append(f"{earliest_period}: {earliest_value}")

    bullets = _finalize_bullets(bullets, data_card)

    strapline_direction = {
        'increase': 'uptrend',
        'decrease': 'downtrend',
        'flat': 'stable trend'
    }.get(direction, 'trend')
    strapline = f"{metric_name} {strapline_direction}"
    strapline_period = latest_period or (period_range.split(' to ')[-1] if period_range else '')
    if strapline_period:
        strapline += f" through {strapline_period}"

    return {
        'title': _limit_title(title),
        'bullets': bullets,
        'strapline': _limit_strapline(strapline)
    }


def _build_tier_insight(data_card: Dict[str, Any]) -> Dict[str, str]:
    metric_name = pretty_label(data_card.get('metric', 'Metric'))
    latest_period = _humanize_period(data_card.get('latest_period'))
    previous_period = _humanize_period(data_card.get('previous_period'))
    tier_meta = (data_card.get('view_metadata') or {}).get('tiers') or {}
    latest_tiers = tier_meta.get('latest') or []
    if not latest_tiers:
        return _build_trend_insight(data_card)

    top_tier, top_value = latest_tiers[0]
    top_name = _format_tier_name(top_tier)
    title = f"{metric_name} mix led by {top_name}"

    changes = {tier: change for tier, change in (tier_meta.get('changes') or [])}
    second_line = None
    if len(latest_tiers) > 1:
        second_tier, second_val = latest_tiers[1]
        second_line = f"{_format_tier_name(second_tier)}: {_format_share(second_val)}"

    bullets = []
    share_text = _format_share(top_value)
    if latest_period:
        bullets.append(f"{top_name}: {share_text} in {latest_period}")
    else:
        bullets.append(f"{top_name}: {share_text}")

    change_val = changes.get(top_tier)
    if change_val is not None and abs(change_val) >= 1e-4 and previous_period:
        direction_word = 'gains' if change_val > 0 else 'loses'
        bullets.append(f"{top_name} {direction_word} {fmt_delta(change_val)} vs {previous_period}")
    elif previous_period:
        bullets.append(f"{top_name} steady vs {previous_period}")

    if second_line:
        bullets.append(second_line)

    bullets = _finalize_bullets(bullets, data_card)

    strapline = f"{metric_name} mix highlights {top_name} leadership"
    strapline_period = latest_period or data_card.get('period_range')
    if strapline_period:
        strapline += f" in {strapline_period}"

    return {
        'title': _limit_title(title),
        'bullets': bullets,
        'strapline': _limit_strapline(strapline)
    }


def _build_delta_insight(data_card: Dict[str, Any]) -> Dict[str, str]:
    metric_name = pretty_label(data_card.get('metric', 'Metric'))
    latest_period = _humanize_period(data_card.get('latest_period'))
    deltas = (data_card.get('view_metadata') or {}).get('deltas') or {}
    if not deltas:
        return _build_trend_insight(data_card)

    preferred_order = ['yoy', 'qoq', 'mom']
    selected_key = None
    for basis in preferred_order:
        for key in deltas.keys():
            if basis in key.lower():
                selected_key = key
                break
        if selected_key:
            break
    if selected_key is None:
        selected_key = next(iter(deltas))

    delta_value = deltas[selected_key]
    basis_label = selected_key.upper().replace('_PCT', '').replace('_PP', '')
    basis_clean = basis_label.replace('YOY', 'YoY').replace('QOQ', 'QoQ').replace('MOM', 'MoM')
    delta_text = fmt_delta(delta_value)
    title = f"{metric_name} {basis_clean} change {delta_text}"

    latest_value = data_card.get('latest_value_formatted') or _format_numeric(data_card.get('latest_value'), data_card.get('metric'))
    previous_period = _humanize_period(data_card.get('previous_period'))
    trend_text = data_card.get('trend')

    bullets = []
    if latest_period and latest_value:
        bullets.append(f"{latest_period}: {latest_value}")
    bullets.append(f"{basis_clean} change: {delta_text}")
    if trend_text:
        bullets.append(f"Trend vs start: {trend_text}")
    elif previous_period and latest_value:
        bullets.append(f"Compared with {previous_period}")

    bullets = _finalize_bullets(bullets, data_card)

    direction = 'improves' if delta_value and delta_value > 0 else 'softens' if delta_value and delta_value < 0 else 'holds'
    strapline = f"{basis_clean} view {direction}"
    if metric_name:
        strapline += f" for {metric_name}"

    return {
        'title': _limit_title(title),
        'bullets': bullets,
        'strapline': _limit_strapline(strapline)
    }


def _humanize_period(period: Optional[Any]) -> Optional[str]:
    if period in (None, '', pd.NA):
        return None
    text = str(period)
    try:
        if 'Q' in text.upper():
            per = pd.Period(text.upper(), freq='Q')
            return f"Q{per.quarter} {per.year}"
        if '-' in text:
            per = pd.Period(text, freq='M')
            return per.strftime('%b %Y')
    except Exception:
        pass
    return text


def _format_share(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return 'N/A'
    return fmt_percent(float(value), decimals=1)


def _format_tier_name(name: Optional[str]) -> str:
    if not name:
        return 'Tier'
    return str(name).replace('_', ' ').title()


def _format_numeric(value: Optional[float], metric: Optional[str]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    metric_lower = (metric or '').lower()
    if any(tok in metric_lower for tok in ('bal', 'amt', 'amount', 'dollar')):
        return fmt_currency(value)
    if any(tok in metric_lower for tok in ('rate', 'pct', 'percent')):
        return fmt_percent(value, decimals=1)
    return f"{value:,.1f}"


def _clean_bullet(text: str) -> str:
    cleaned = text.strip()
    if cleaned.endswith('.'):
        cleaned = cleaned[:-1]
    return cleaned


def _finalize_bullets(bullets: List[str], data_card: Dict[str, Any]) -> List[str]:
    cleaned: List[str] = []
    for bullet in bullets:
        if not bullet:
            continue
        text = _clean_bullet(bullet)
        if text and text not in cleaned:
            cleaned.append(text)

    if data_card.get('period_range') and len(cleaned) < 2:
        cleaned.append(f"Range: {data_card['period_range']}")
    while len(cleaned) < 2:
        cleaned.append("Monitoring continues")

    return cleaned[:3]


def _limit_title(title: str) -> str:
    title = title.strip()
    if len(title) <= NARRATIVE_LIMITS['title_max_chars']:
        return title
    return title[:NARRATIVE_LIMITS['title_max_chars']].rstrip()


def _limit_strapline(text: str) -> str:
    text = text.strip()
    max_len = NARRATIVE_LIMITS['strapline_max_chars']
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip()


def _generate_fallback_narrative(data_card: Dict[str, Any], narrative_type: str) -> Dict[str, str]:
    """
    Generate fallback narrative when LLM fails.
    Uses stub_llm_narrative as the fallback.
    
    Args:
        data_card: Data card with facts
        narrative_type: Type of narrative
    
    Returns:
        Fallback narrative dictionary
    """
    # Use stub narrative as fallback
    return stub_llm_narrative(data_card, SynthesisConfig(), narrative_type)


def _validate_narrative(narrative: Dict[str, str], config: SynthesisConfig, 
                       data_card: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate narrative against limits and data card."""
    issues = []
    
    # Check title length
    title = narrative.get('title', '')
    if len(title) > NARRATIVE_LIMITS['title_max_chars']:
        issues.append(f"Title too long: {len(title)} > {NARRATIVE_LIMITS['title_max_chars']}")
    
    # Check bullet count and format
    bullets = narrative.get('bullets', [])
    if len(bullets) < NARRATIVE_LIMITS['bullet_min']:
        issues.append(f"Too few bullets: {len(bullets)} < {NARRATIVE_LIMITS['bullet_min']}")
    if len(bullets) > NARRATIVE_LIMITS['bullet_max']:
        issues.append(f"Too many bullets: {len(bullets)} > {NARRATIVE_LIMITS['bullet_max']}")
    
    # Check for trailing periods
    for bullet in bullets:
        if bullet.endswith('.'):
            issues.append(f"Bullet has trailing period: '{bullet}'")
    
    # Check strapline length
    strapline = narrative.get('strapline', '')
    if len(strapline) > NARRATIVE_LIMITS['strapline_max_chars']:
        issues.append(f"Strapline too long: {len(strapline)} > {NARRATIVE_LIMITS['strapline_max_chars']}")
    
    # STRICT NUMERIC GUARD: Check all numbers against data card
    if data_card:
        all_text = title + ' ' + ' '.join(bullets) + ' ' + strapline
        numbers_in_text = _extract_numbers_from_text(all_text)
        
        if numbers_in_text:
            violations = _verify_numbers_against_data_card(numbers_in_text, data_card)
            issues.extend(violations)
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues
    }


def _reorder_narratives(narratives: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Reorder narratives by importance."""
    # Priority order: Balance > Originations > Supply > Others
    priority_map = {
        'bal': 1,
        'orig': 2,
        'new': 2,
        'supply': 3,
        'cr_line': 3,
        'otb': 3
    }
    
    def get_priority(narrative):
        metric = narrative.get('metric', '').lower()
        for key, priority in priority_map.items():
            if key in metric:
                return priority
        return 99
    
    return sorted(narratives, key=get_priority)


def _scan_contradictions(narratives: List[Dict[str, str]]) -> List[Tuple[int, int, str]]:
    """Scan for contradictory statements."""
    contradictions = []
    
    for i in range(len(narratives)):
        for j in range(i + 1, len(narratives)):
            # Check for opposite trends
            if 'increase' in narratives[i].get('strapline', '').lower() and \
               'decrease' in narratives[j].get('strapline', '').lower():
                if narratives[i].get('metric') == narratives[j].get('metric'):
                    contradictions.append((i, j, "Opposite trends for same metric"))
            
            # Check for conflicting risk assessments
            if 'high risk' in narratives[i].get('strapline', '').lower() and \
               'low risk' in narratives[j].get('strapline', '').lower():
                contradictions.append((i, j, "Conflicting risk assessments"))
    
    return contradictions
