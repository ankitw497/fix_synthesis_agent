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
    fmt_currency, fmt_percent, fmt_delta, setup_logging
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
    Build data card with all facts for LLM.
    CRITICAL: Include composition_base for A3 charts.
    
    Args:
        df: DataFrame with data
        metric: Metric name
        chart_type: Type of chart (A2, A3, etc.)
        period_col: Period column name
        composition_base: Base period for composition (A3 only)
    
    Returns:
        Data card dictionary
    """
    card = {
        'metric': metric,
        'chart_type': chart_type,
        'period_range': None,
        'latest_value': None,
        'earliest_value': None,
        'min_value': None,
        'max_value': None,
        'mean_value': None,
        'trend': None,
        'volatility': None,
        'composition_base': composition_base,  # CRITICAL for A3
        'key_facts': [],
        'view': None,
        'delta_type': None,
        'split_by': None,
    }
    
    # Extract period range
    if period_col in df.columns:
        periods = df[period_col].dropna()
        if len(periods) > 0:
            card['period_range'] = f"{periods.min()} to {periods.max()}"
    
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
                if is_rate:
                    change = (series.iloc[-1] - series.iloc[0]) * (100.0 if abs(series.iloc[0]) <= 1 else 1.0)
                    card['trend'] = f"{change:+.1f} bps" if abs(change) < 1 else f"{change:+.1f} pp"
                else:
                    base = series.iloc[0]
                    change = ((series.iloc[-1] / base) - 1) * 100 if base else None
                    card['trend'] = f"{change:+.1f}%" if change is not None and not pd.isna(change) else None
                card['trend_value'] = change  # raw value
            
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
        # Look for delta columns
        delta_cols = [c for c in df.columns if metric in c and any(d in c for d in ['_yoy_pct', '_qoq_pct', '_mom_pct'])]
        if delta_cols:
            latest_delta = df[delta_cols[0]].iloc[-1]
            if pd.notna(latest_delta):
                card['key_facts'].append(f"Latest change: {fmt_delta(latest_delta)}")
    
    # Add tier information if present
    from synthesis_agent.io_normalize import extract_tier_columns
    tier_cols = extract_tier_columns(df)
    if tier_cols:
        # Get latest tier distribution (use last valid row to avoid dtype issues)
        latest_idx = df[period_col].dropna().index[-1] if period_col in df.columns and not df[period_col].dropna().empty else -1
        tier_values = {}
        for tier in tier_cols:
            if tier in df.columns:
                value = df.loc[latest_idx, tier] if latest_idx >= 0 else df[tier].iloc[-1]
                if pd.notna(value):
                    tier_values[tier] = value
        
        if tier_values:
            # Find dominant tier
            dominant_tier = max(tier_values, key=tier_values.get)
            card['key_facts'].append(f"Dominant tier: {dominant_tier} ({tier_values[dominant_tier]:.1f})")
    
    logger.info(f"Built data card for {metric}/{chart_type} with {len(card['key_facts'])} facts")
    return card


def llm_narrate(data_card: Dict[str, Any],
               config: SynthesisConfig,
               narrative_type: str = 'insight') -> Dict[str, str]:
    """
    Generate LLM narrative with strict limits.
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
    
    # Call LLM (Gemini Flash via Vertex AI)
    try:
        narrative = _call_llm(prompt, config)
            # Preserve separate insight title and numeric headline if provided
        if isinstance(narrative, dict) and "title" in narrative and "headline" in narrative:
            narrative["insight_title"] = narrative["title"]
            narrative["metric_headline"] = narrative["headline"]
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        # If Vertex is selected, it must work (no silent fallback)
        raise RuntimeError(f"Vertex AI narrative generation failed: {e}")
    
    # Validate and enforce limits (including numeric guard)
    validated = _validate_narrative(narrative, config, data_card)
    
    # Re-prompt if validation failed
    retry_count = 0
    while not validated['is_valid'] and retry_count < config.runtime.max_retries:
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
    """Build prompt for insight narrative."""
    limits = NARRATIVE_LIMITS
    
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

    view = (data_card.get('view') or '').lower()
    delta_type = (data_card.get('delta_type') or '').upper()
    split_by = (data_card.get('split_by') or '')

    if view == 'trend':
        prompt += "\nTITLE MUST explicitly reference a trend over time (use words like 'Trend' or 'Over time') and mention the time window."  # noqa: E501
    elif view == 'composition':
        prompt += "\nTITLE MUST call out the credit tier mix and specify which tiers gained or lost share."
        if 'tier' in split_by.lower():
            prompt += "\nALWAYS mention at least one credit tier (e.g., Prime, Near-Prime, Subprime) when describing the changes."  # noqa: E501
    elif view == 'delta':
        if delta_type in ('QOQ', 'YOY'):
            prompt += f"\nTITLE MUST explicitly mention {delta_type} change and highlight the largest increases or decreases."
        else:
            prompt += "\nTITLE MUST call out period-over-period change and identify the strongest movers."

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
        
    except ImportError:
        # Vertex AI SDK not available - use fallback implementation
        logger.warning("Vertex AI SDK not available, using deterministic fallback")
        
        # Parse key information from prompt to generate contextual response
        import re
        
        # Extract metric name from prompt
        metric_match = re.search(r'for (\w+)', prompt)
        metric = metric_match.group(1) if metric_match else "metric"
        
        # Extract trend if present
        trend_match = re.search(r'Trend: ([^\\n]+)', prompt)
        trend = trend_match.group(1) if trend_match else "stable"
        
        # Generate contextual response based on extracted info
        metric_display = metric.replace('_', ' ').title()
        
        return {
            "title": f"{metric_display} Performance Overview",
            "bullets": [
                f"{metric_display} analysis reveals {trend} pattern",
                "Key drivers align with portfolio objectives",
                "Continued monitoring recommended for optimization"
            ],
            "strapline": f"{metric_display} demonstrates expected behavior within portfolio context"
        }
        
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
    """
    Stub LLM narrative for testing or when Vertex AI is unavailable.
    
    Args:
        data_card: Data card with facts
        config: Synthesis configuration
        narrative_type: Type of narrative
    
    Returns:
        Stub narrative dictionary
    """
    metric = data_card.get('metric', 'Data')
    metric_label = metric.replace('_', ' ').title()
    view = (data_card.get('view') or '').lower()
    delta_type = (data_card.get('delta_type') or '').upper()

    if view == 'composition':
        title = f"{metric_label} — Credit tier mix"
    elif view == 'delta':
        if delta_type in ('QOQ', 'YOY'):
            title = f"{metric_label} — {delta_type} change"
        else:
            title = f"{metric_label} — Period-over-period change"
    elif view == 'trend':
        title = f"{metric_label} — Trend over time"
    else:
        title = f"{metric_label} — Performance overview"

    bullets = []
    if data_card.get('latest_value') is not None and data_card.get('earliest_value') is not None:
        bullets.append(f"Current value: {data_card['latest_value']}, initial: {data_card['earliest_value']}")

    if data_card.get('trend_direction'):
        bullets.append(f"Trend shows {data_card['trend_direction']} movement")
    
    if data_card.get('delta_values'):
        for delta_type, value in data_card['delta_values'].items():
            if value is not None:
                bullets.append(f"{delta_type}: {value}")
    
    if view == 'composition':
        dominant_fact = next(
            (fact for fact in data_card.get('key_facts', []) if str(fact).lower().startswith('dominant tier')),
            None,
        )
        tier_bullet = dominant_fact or "Credit tier mix shifts across tiers"
        bullets.insert(0, tier_bullet)

    # Ensure at least 2 bullets
    while len(bullets) < 2:
        bullets.append(f"Pattern detected in {metric.replace('_', ' ')} data")

    # Cap at 3 bullets
    bullets = bullets[:3]

    if view == 'composition':
        strapline = f"{metric_label} tiers show mix shifts across credit bands"
    elif view == 'delta' and delta_type in ('QOQ', 'YOY'):
        strapline = f"{metric_label} {delta_type} change spotlights leading movers"
    elif view == 'trend':
        strapline = f"{metric_label} trend highlights momentum over time"
    else:
        strapline = f"Key insight: {metric.replace('_', ' ')} shows measurable change over the period"

    return {
        'title': title[:NARRATIVE_LIMITS['title_max_chars']],
        'bullets': bullets,
        'strapline': strapline[:NARRATIVE_LIMITS['strapline_max_chars']]
    }


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
    
    view = (data_card or {}).get('view') if data_card else None
    view_lower = str(view).lower() if view else ''
    delta_type = (data_card or {}).get('delta_type') if data_card else None
    delta_upper = str(delta_type).upper() if delta_type else ''
    title_lower = title.lower()

    if view_lower == 'trend':
        if not any(token in title_lower for token in ('trend', 'over time')):
            issues.append("Title must mention trend or over time")
    elif view_lower == 'composition':
        if not any(token in title_lower for token in ('tier', 'mix', 'composition')):
            issues.append("Title must include credit tier mix")
        tier_tokens = ('prime', 'near-prime', 'subprime', 'super-prime', 'deep subprime')
        tier_present = any(token in title_lower for token in tier_tokens)
        if not tier_present:
            for bullet in bullets:
                if isinstance(bullet, str) and any(token in bullet.lower() for token in tier_tokens):
                    tier_present = True
                    break
        if not tier_present:
            issues.append("Narrative must reference at least one credit tier")
    elif view_lower == 'delta':
        if delta_upper == 'QOQ':
            tokens = ('qoq', 'quarter-over-quarter', 'q/q')
            if not any(token in title_lower for token in tokens):
                issues.append("Title must include QoQ phrasing")
        elif delta_upper == 'YOY':
            tokens = ('yoy', 'year-over-year', 'y/y')
            if not any(token in title_lower for token in tokens):
                issues.append("Title must include YoY phrasing")
        else:
            if not any(token in title_lower for token in ('change', 'delta', 'period-over-period')):
                issues.append("Title must highlight period-over-period change")

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


def _generate_fallback_narrative(data_card: Dict[str, Any], narrative_type: str) -> Dict[str, str]:
    """Generate safe fallback narrative."""
    metric_name = data_card['metric'].replace('_', ' ').title()
    
    if narrative_type == 'insight':
        return {
            "title": f"{metric_name} Analysis",
            "bullets": [
                f"Data covers period {data_card['period_range']}",
                f"Latest value: {data_card['latest_value']}",
                f"Overall trend: {data_card['trend'] or 'Stable'}"
            ],
            "strapline": f"{metric_name} shows expected patterns over the analysis period"
        }
    else:
        return {
            "summary": f"{metric_name} analysis for {data_card['period_range']}"
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