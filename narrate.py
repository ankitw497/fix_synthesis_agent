"""
Narrative generation for Agent 3 synthesis pipeline.
Implements data cards, LLM narratives with limits, and QA checks.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

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
        """Fallback aggregation if charts module unavailable."""
        return df.groupby(period_col)[metric].sum().reset_index()
        

logger = setup_logging("narrate")


def _get_chart_context(chart_type: str, delta_type: Optional[str] = None, tier_col: Optional[str] = None) -> str:
    """Generate chart-specific context for narrative generation."""
    context_map = {
        'A2': 'trend analysis over time',
        'A3': 'composition breakdown by credit tiers',
        'A4': f'{delta_type.upper() if delta_type else "Period-over-period"} change analysis',
        'A5': 'dual-metric comparison analysis'
    }
    
    base_context = context_map.get(chart_type, 'performance analysis')
    
    if chart_type == 'A3' and tier_col:
        base_context += f' (grouped by {tier_col.replace("_", " ").title()})'
    elif chart_type == 'A4' and delta_type:
        base_context = f'{delta_type.upper()} change analysis'
    
    return base_context


def build_data_card(df: pd.DataFrame,
                   metric: str,
                   chart_type: str,
                   period_col: str = 'period',
                   composition_base: Optional[str] = None,
                   delta_type: Optional[str] = None,
                   tier_col: Optional[str] = None,
                   analysis_mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Build data card with all facts for LLM.
    CRITICAL: Include composition_base for A3 charts and chart-specific context.
    
    Args:
        df: DataFrame with data
        metric: Metric name
        chart_type: Type of chart (A2, A3, etc.)
        period_col: Period column name
        composition_base: Base period for composition (A3 only)
        delta_type: Type of delta analysis (yoy, qoq, mom)
        tier_col: Credit tier column name
        analysis_mode: Analysis mode (TRENDS, QOQ, YOY, etc.)
    
    Returns:
        Data card dictionary
    """
    card = {
        'metric': metric,
        'metric_full_name': pretty_label(metric),  # Full form of metric
        'chart_type': chart_type,
        'delta_type': delta_type,
        'tier_col': tier_col,
        'analysis_mode': analysis_mode,
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
        'chart_context': _get_chart_context(chart_type, delta_type, tier_col)
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
            
            # Basic trend analysis
            if len(series) >= 2:
                if card['latest_value'] > card['earliest_value']:
                    card['trend'] = 'increasing'
                elif card['latest_value'] < card['earliest_value']:
                    card['trend'] = 'decreasing'
                else:
                    card['trend'] = 'stable'
                
                # Volatility (coefficient of variation)
                std_val = series.std()
                mean_val = series.mean()
                if mean_val != 0:
                    card['volatility'] = float(abs(std_val / mean_val))
                
                # Period-over-period change facts
                pct_change = series.pct_change().dropna()
                if len(pct_change) > 0:
                    card['key_facts'].append(f"Average period change: {pct_change.mean():.2%}")
                    if pct_change.std() > 0.1:  # High volatility
                        card['key_facts'].append("High volatility observed")
                
                # Latest period insights
                if len(series) >= 3:
                    recent_trend = series.iloc[-3:].pct_change().mean()
                    if abs(recent_trend) > 0.05:  # 5% threshold
                        direction = "acceleration" if recent_trend > 0 else "deceleration"
                        card['key_facts'].append(f"Recent {direction} in trend")
    
    # Add chart-specific facts
    if chart_type == 'A3' and composition_base:
        card['key_facts'].append(f"Composition analysis based on {composition_base}")
    
    if chart_type == 'A4':
        delta_label = f"{delta_type.upper()} analysis" if delta_type else "Period-over-period analysis"
        card['key_facts'].append(delta_label)
        
        # Add delta-specific insights if delta columns exist
        if delta_type:
            delta_col_pct = f"{metric}_{delta_type}_pct"
            delta_col_pp = f"{metric}_{delta_type}_pp"
            
            if delta_col_pct in df.columns:
                delta_series = df[delta_col_pct].dropna()
                if len(delta_series) > 0:
                    latest_delta = delta_series.iloc[-1]
                    card['key_facts'].append(f"Latest {delta_type.upper()}: {latest_delta:.1f}%")
            elif delta_col_pp in df.columns:
                delta_series = df[delta_col_pp].dropna()
                if len(delta_series) > 0:
                    latest_delta = delta_series.iloc[-1]
                    card['key_facts'].append(f"Latest {delta_type.upper()}: {latest_delta:.1f}pp")
    
    # Add tier information if present
    from synthesis_agent.io_normalize import extract_tier_columns
    tier_cols = extract_tier_columns(df)
    if tier_cols and chart_type == 'A3':
        tier_col = tier_cols[0]
        if tier_col in df.columns:
            tier_values = df.groupby(tier_col)[metric].sum()
            if len(tier_values) > 0:
                # Find dominant tier
                dominant_tier = tier_values.idxmax() if len(tier_values) > 0 else None
                if dominant_tier:
                    card['key_facts'].append(f"Largest segment: {dominant_tier}")
                
                # Tier concentration
                total = tier_values.sum()
                if total > 0:
                    top_share = tier_values.max() / total
                    card['key_facts'].append(f"Top tier represents {top_share:.1%} of total")
    
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
        logger.warning(f"Unknown narrative engine: {engine}, using deterministic fallback")
        return stub_llm_narrative(data_card, config, narrative_type)
    
    # Prepare prompt based on narrative type
    if narrative_type == 'insight':
        prompt = _build_chart_specific_prompt(data_card, config)
    elif narrative_type == 'summary':
        prompt = _build_summary_prompt(data_card, config)
    else:
        prompt = _build_notes_prompt(data_card, config)
    
    # Call LLM (Gemini Flash via Vertex AI)
    try:
        narrative = _call_llm(prompt, config)
        logger.debug(f"Generated narrative for {data_card['metric']}")
    except Exception as e:
        logger.error(f"LLM call failed: {e}, using fallback")
        return stub_llm_narrative(data_card, config, narrative_type)
    
    # Validate and enforce limits (including numeric guard)
    validated = _validate_narrative(narrative, config, data_card)
    
    # Re-prompt if validation failed
    retry_count = 0
    while not validated['is_valid'] and retry_count < config.runtime.max_retries:
        retry_count += 1
        logger.warning(f"Narrative validation failed (attempt {retry_count}), re-prompting...")
        try:
            narrative = _call_llm(prompt + "\nIMPORTANT: Previous response failed validation. Please follow ALL character limits strictly.", config)
            validated = _validate_narrative(narrative, config, data_card)
        except Exception as e:
            logger.error(f"Re-prompt failed: {e}")
            break
    
    if not validated['is_valid']:
        logger.warning("Final narrative validation failed, using fallback")
        return stub_llm_narrative(data_card, config, narrative_type)
    
    # Sanitize LLM output to remove debug/reasoning text
    def _strip_reasoning(s: str) -> str:
        """Remove common LLM reasoning patterns."""
        patterns = [
            r'^(here\'s|this is|based on)',
            r'^(the chart shows|analysis reveals|data indicates)',
            r'(in summary|to conclude|in conclusion)$'
        ]
        for pattern in patterns:
            s = re.sub(pattern, '', s, flags=re.IGNORECASE).strip()
        return s
    
    # Remove unexpected debug keys from narrative
    for k in list(narrative.keys()):
        if k not in ['title', 'bullets', 'strapline']:
            del narrative[k]
    
    # Clean reasoning text from string fields
    for k in ['title', 'strapline']:
        if k in narrative and isinstance(narrative[k], str):
            narrative[k] = _strip_reasoning(narrative[k])
    
    if 'bullets' in narrative and isinstance(narrative['bullets'], list):
        narrative['bullets'] = [_strip_reasoning(b) if isinstance(b, str) else b
                               for b in narrative['bullets']]
    
    return narrative


def _build_chart_specific_prompt(data_card: Dict[str, Any], config: SynthesisConfig) -> str:
    """Build chart-type specific prompt for insight narrative."""
    limits = NARRATIVE_LIMITS
    chart_type = data_card.get('chart_type', 'A2')
    chart_context = data_card.get('chart_context', 'analysis')
    metric_full_name = data_card.get('metric_full_name', data_card.get('metric', 'metric'))
    delta_type = data_card.get('delta_type')
    tier_col = data_card.get('tier_col')
    
    # Use formatted values if available, fallback to raw
    latest_value = data_card.get('latest_value_formatted', data_card.get('latest_value', 'N/A'))
    earliest_value = data_card.get('earliest_value_formatted', data_card.get('earliest_value', 'N/A'))
    
    # Chart-specific prompt templates
    if chart_type == 'A2':
        # Trend analysis
        prompt_template = f"""
Generate an executive insight for {metric_full_name} TREND ANALYSIS with the following constraints:

CHART CONTEXT: This is a time series trend chart showing {metric_full_name} performance over time.

STRICT REQUIREMENTS:
- Title: Maximum {limits['title_max_chars']} characters, MUST mention "trend" or "trajectory" or "performance over time"
- Bullets: Exactly {limits['bullet_min']}-{limits['bullet_max']} bullets focusing on TREND PATTERNS, NO trailing periods
- Strapline: MUST BE UNDER {limits['strapline_max_chars']} CHARACTERS, summarize overall trend direction
- Use ONLY the exact numbers provided below - do not calculate or format new numbers

FOCUS ON:
- Overall trend direction (increasing/decreasing/stable)
- Rate of change and momentum
- Notable inflection points or patterns
- Period-over-period variations

"""
    elif chart_type == 'A3':
        # Composition analysis
        prompt_template = f"""
Generate an executive insight for {metric_full_name} TIER COMPOSITION ANALYSIS with the following constraints:

CHART CONTEXT: This is a composition chart showing {metric_full_name} breakdown by credit tiers/segments.

STRICT REQUIREMENTS:
- Title: Maximum {limits['title_max_chars']} characters, MUST mention "tier mix", "composition", or "segment breakdown"
- Bullets: Exactly {limits['bullet_min']}-{limits['bullet_max']} bullets focusing on TIER INSIGHTS, NO trailing periods
- Strapline: MUST BE UNDER {limits['strapline_max_chars']} CHARACTERS, summarize dominant tier or mix changes
- Use ONLY the exact numbers provided below - do not calculate or format new numbers

FOCUS ON:
- Which credit tiers dominate (Prime, Near-Prime, Subprime, etc.)
- Changes in tier composition over time
- Concentration vs diversification patterns
- Risk profile implications of the mix

"""
    elif chart_type == 'A4':
        # Delta analysis
        delta_label = delta_type.upper() if delta_type else "Period-over-period"
        prompt_template = f"""
Generate an executive insight for {metric_full_name} {delta_label} CHANGE ANALYSIS with the following constraints:

CHART CONTEXT: This is a delta chart showing {delta_label} changes in {metric_full_name}.
FOCUS: This chart shows CHANGES/DELTAS, not absolute values. Emphasize growth rates, accelerations, and change patterns.

STRICT REQUIREMENTS:
- Title: Maximum {limits['title_max_chars']} characters, MUST mention "{delta_label}", "change", "growth", or "delta"
- Bullets: Exactly {limits['bullet_min']}-{limits['bullet_max']} bullets focusing on CHANGE PATTERNS and MOMENTUM, NO trailing periods
- Strapline: MUST BE UNDER {limits['strapline_max_chars']} CHARACTERS, summarize change direction and acceleration/deceleration
- Use ONLY the exact numbers provided below - do not calculate or format new numbers

FOCUS ON:
- Magnitude and direction of {delta_label} changes (positive/negative growth)
- Acceleration or deceleration patterns (is growth speeding up or slowing down?)
- Volatility in change rates (consistent vs erratic changes)
- Recent momentum vs historical change patterns
- Business implications of the change trajectory

EXAMPLE PHRASES for {delta_label} analysis:
- "accelerating growth momentum"
- "decelerating expansion pace"
- "volatile change patterns"
- "consistent {delta_label} improvement"
- "growth trajectory stabilizing"

"""
    elif chart_type == 'A5':
        # Dual axis analysis
        prompt_template = f"""
Generate an executive insight for {metric_full_name} DUAL-METRIC COMPARISON with the following constraints:

CHART CONTEXT: This is a dual-axis chart comparing related metrics for comprehensive analysis.

STRICT REQUIREMENTS:
- Title: Maximum {limits['title_max_chars']} characters, MUST mention "relationship", "correlation", or "comparison"
- Bullets: Exactly {limits['bullet_min']}-{limits['bullet_max']} bullets focusing on METRIC RELATIONSHIPS, NO trailing periods
- Strapline: MUST BE UNDER {limits['strapline_max_chars']} CHARACTERS, summarize correlation or divergence
- Use ONLY the exact numbers provided below - do not calculate or format new numbers

FOCUS ON:
- Correlation or divergence between metrics
- Leading/lagging relationships
- Ratio analysis and efficiency metrics
- Combined performance insights

"""
    else:
        # Fallback generic template
        prompt_template = f"""
Generate an executive insight for {metric_full_name} with the following constraints:

STRICT REQUIREMENTS:
- Title: Maximum {limits['title_max_chars']} characters
- Bullets: Exactly {limits['bullet_min']}-{limits['bullet_max']} bullets, NO trailing periods
- Strapline: MUST BE UNDER {limits['strapline_max_chars']} CHARACTERS
- Use ONLY the exact numbers provided below - do not calculate or format new numbers

"""
    
    # Add common data section
    data_section = f"""
DATA CARD:
Metric: {metric_full_name} (original: {data_card['metric']})
Chart Type: {chart_type} ({chart_context})
Period: {data_card['period_range']}
Latest Value: {latest_value}
Initial Value: {earliest_value}
Trend: {data_card.get('trend', 'N/A')}
Key Facts: {json.dumps(data_card['key_facts'])}

IMPORTANT: 
- The strapline MUST be concise (under 140 characters)
- Use the formatted values provided (e.g., "302.5B" not "302450048944")
- Reference periods as quarters (e.g., "Q4 2022" from "2022Q4")
- Focus on insights specific to this chart type

FORMAT:
{{
  "title": "...",
  "bullets": ["bullet1", "bullet2", "bullet3"],
  "strapline": "concise key insight under 140 chars"
}}
"""
    
    return prompt_template + data_section


def narrative_qc(narratives: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Quality check narratives for consistency and accuracy.
    
    Args:
        narratives: List of narrative dictionaries
    
    Returns:
        Quality-checked narratives
    """
    if not narratives:
        return narratives
    
    # Check for duplicate titles
    titles_seen = set()
    for i, narrative in enumerate(narratives):
        title = narrative.get('title', '')
        if title in titles_seen:
            logger.warning(f"Duplicate narrative title detected: '{title}'")
            # Append index to make unique
            narrative['title'] = f"{title} ({i+1})"
        titles_seen.add(title)
    
    # Check for contradictory statements
    contradictions = _scan_contradictions(narratives)
    if contradictions:
        logger.warning(f"Found {len(contradictions)} potential contradictions in narratives")
        for i, j, reason in contradictions:
            logger.warning(f"Narratives {i} and {j}: {reason}")
    
    # Reorder narratives for logical flow
    narratives = _reorder_narratives(narratives)
    
    return narratives


def generate_speaker_notes(data_card: Dict[str, Any],
                          narrative: Dict[str, str],
                          config: SynthesisConfig) -> str:
    """
    Generate speaker notes for presenter.
    
    Args:
        data_card: Data card with chart facts
        narrative: Generated narrative
        config: Synthesis configuration
    
    Returns:
        Speaker notes text
    """
    chart_type = data_card.get('chart_type', 'A2')
    metric_full_name = data_card.get('metric_full_name', data_card.get('metric', 'metric'))
    
    notes = []
    
    # Chart-specific talking points
    if chart_type == 'A2':
        notes.append(f"This trend chart shows {metric_full_name} evolution over time")
        notes.append("Highlight the overall direction and any significant turning points")
    elif chart_type == 'A3':
        notes.append(f"This composition shows {metric_full_name} by credit tier segments")
        notes.append("Emphasize the tier mix and any shifts in risk profile")
    elif chart_type == 'A4':
        delta_type = data_card.get('delta_type', 'period-over-period')
        notes.append(f"This chart shows {delta_type.upper()} changes in {metric_full_name}")
        notes.append("Focus on the magnitude and consistency of changes")
    elif chart_type == 'A5':
        notes.append(f"This dual-axis chart compares related {metric_full_name} metrics")
        notes.append("Point out correlations or divergences between the metrics")
    
    # Add data context
    if data_card.get('period_range'):
        notes.append(f"Data covers {data_card['period_range']}")
    
    # Add key supporting facts
    key_facts = data_card.get('key_facts', [])
    if key_facts:
        notes.append("Supporting details: " + "; ".join(key_facts[:2]))  # Limit to 2 facts
    
    return " • ".join(notes[:3])  # Limit to 3 bullet points


# Helper functions

def _build_summary_prompt(data_card: Dict[str, Any], config: SynthesisConfig) -> str:
    """Build prompt for summary narrative."""
    return f"""
Summarize the key findings for {data_card['metric']} in 2-3 concise bullets.
Focus on the most important business insights from the data.
"""


def _build_notes_prompt(data_card: Dict[str, Any], config: SynthesisConfig) -> str:
    """Build prompt for speaker notes."""
    return f"""
Generate speaker notes for {data_card['metric']} chart presentation.
Include 3 key talking points for the presenter.
Keep technical but accessible for business audience.
"""


def generate_narrative(data_card: Dict[str, Any], config: SynthesisConfig, 
                       narrative_type: str = 'insight', llm_func=None) -> Dict[str, str]:
    """Legacy interface for narrative generation."""
    if llm_func:
        # Use custom LLM function if provided
        prompt = _build_chart_specific_prompt(data_card, config)
        return llm_func(prompt)
    else:
        # Use standard LLM narrative
        return llm_narrate(data_card, config, narrative_type)


def _call_llm(prompt: str, config: SynthesisConfig) -> Dict[str, str]:
    """Call LLM via Vertex AI."""
    try:
        # Import here to avoid startup dependencies
        from vertexai.generative_models import GenerativeModel
        import vertexai
        
        # Initialize Vertex AI
        vertexai.init(
            project=config.runtime.project_id,
            location=config.runtime.location
        )
        
        # Initialize model
        model = GenerativeModel(config.runtime.generative_model_name)
        
        # Configure generation parameters
        generation_config = {
            "temperature": config.runtime.temperature,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json"
        }
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Parse JSON response
        narrative = json.loads(response.text)
        
        # Ensure required keys exist
        if not isinstance(narrative, dict):
            raise ValueError("Response is not a dictionary")
            
        required_keys = ['title', 'bullets', 'strapline']
        for key in required_keys:
            if key not in narrative:
                narrative[key] = ''
        
        # Ensure bullets is a list
        if not isinstance(narrative.get('bullets'), list):
            narrative['bullets'] = []
            
        return narrative
        
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


def _extract_numbers_from_text(text: str) -> List[str]:
    """Extract numeric values from text for validation."""
    if not text:
        return []
    
    # Patterns for various number formats
    patterns = [
        r'\d+\.?\d*[BMK]',  # 123.4B, 56M, 78K
        r'\d+\.?\d*%',      # 12.3%
        r'\d+\.?\d*pp',     # 2.5pp (percentage points)
        r'\d+\.?\d*bps',    # 50bps (basis points)
        r'\$\d+\.?\d*[BMK]?', # $123.4B
        r'\d{4}Q[1-4]',     # 2022Q4
        r'\d+\.?\d+',       # General numbers
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers.extend(matches)
    
    return numbers


def _verify_numbers_against_data_card(numbers: List[str], data_card: Dict[str, Any]) -> List[str]:
    """Verify extracted numbers against data card values."""
    issues = []
    
    if not numbers:
        return issues
    
    # Get reference values from data card
    reference_values = []
    for key in ['latest_value', 'earliest_value', 'min_value', 'max_value']:
        val = data_card.get(key)
        if val is not None:
            reference_values.append(str(val))
    
    # Check for numbers that don't appear in reference
    for num in numbers:
        # Simple check - more sophisticated validation could be added
        if not any(ref for ref in reference_values if str(ref) in num or num in str(ref)):
            issues.append(f"Number '{num}' may not match data card values")
    
    return issues


def stub_llm_narrative(data_card: Dict[str, Any], config: SynthesisConfig, 
                      narrative_type: str = 'insight') -> Dict[str, str]:
    """Generate deterministic narrative based on data card."""
    metric = data_card.get('metric', 'metric')
    metric_display = data_card.get('metric_full_name', pretty_label(metric))
    chart_type = data_card.get('chart_type', 'A2')
    chart_context = data_card.get('chart_context', 'analysis')
    
    trend = data_card.get('trend', 'stable')
    latest_value = data_card.get('latest_value', 'N/A')
    
    # Chart-specific deterministic narratives
    if chart_type == 'A2':
        title = f"{metric_display} shows {trend} trend over time"
        bullets = [
            f"Overall trajectory indicates {trend} performance",
            f"Latest value stands at {latest_value}",
            "Trend analysis supports strategic planning"
        ]
        strapline = f"{metric_display} maintains {trend} trajectory"
    elif chart_type == 'A3':
        title = f"{metric_display} composition by credit tiers"
        bullets = [
            f"Tier breakdown reveals risk profile composition",
            f"Current mix supports portfolio strategy", 
            "Segment analysis enables targeted decisions"
        ]
        strapline = f"{metric_display} tier mix aligns with risk appetite"
    elif chart_type == 'A4':
        delta_type = data_card.get('delta_type', 'period-over-period')
        title = f"{metric_display} {delta_type.upper()} change analysis"
        bullets = [
            f"{delta_type.upper()} changes show performance momentum",
            f"Change patterns indicate {trend} direction",
            "Growth analysis supports forecasting models"
        ]
        strapline = f"{metric_display} {delta_type.upper()} changes reflect {trend} momentum"
    elif chart_type == 'A5':
        title = f"{metric_display} dual-metric relationship analysis"
        bullets = [
            f"Metric correlation reveals operational efficiency",
            f"Combined analysis shows balanced performance",
            "Relationship insights guide strategic decisions"
        ]
        strapline = f"{metric_display} metrics show correlated performance"
    else:
        title = f"{metric_display} {chart_context}"
        bullets = [
            f"{metric_display} analysis reveals {trend} pattern",
            "Key drivers align with portfolio objectives",
            "Continued monitoring recommended for optimization"
        ]
        strapline = f"{metric_display} performance supports strategic objectives"
    
    return {
        'title': title[:NARRATIVE_LIMITS['title_max_chars']],  # Truncate if needed
        'bullets': bullets[:NARRATIVE_LIMITS['bullet_max']],   # Limit bullet count
        'strapline': strapline[:NARRATIVE_LIMITS['strapline_max_chars']]  # Truncate if needed
    }


def _validate_narrative(narrative: Dict[str, str], config: SynthesisConfig, 
                       data_card: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate narrative against limits and data card."""
    issues = []
    limits = NARRATIVE_LIMITS
    
    # Check title length
    title = narrative.get('title', '')
    if len(title) > limits['title_max_chars']:
        issues.append(f"Title too long: {len(title)} > {limits['title_max_chars']}")
    
    # Check bullet count and content
    bullets = narrative.get('bullets', [])
    if not isinstance(bullets, list):
        issues.append("Bullets must be a list")
    elif len(bullets) < limits['bullet_min'] or len(bullets) > limits['bullet_max']:
        issues.append(f"Bullet count {len(bullets)} not in range {limits['bullet_min']}-{limits['bullet_max']}")
    
    # Check strapline length
    strapline = narrative.get('strapline', '')
    if len(strapline) > limits['strapline_max_chars']:
        issues.append(f"Strapline too long: {len(strapline)} > {limits['strapline_max_chars']}")
    
    # Check for trailing periods in bullets
    for i, bullet in enumerate(bullets):
        if isinstance(bullet, str) and bullet.endswith('.'):
            issues.append(f"Bullet {i+1} has trailing period")
    
    # Verify numbers against data card if provided
    if data_card:
        all_text = ' '.join([title, strapline] + [str(b) for b in bullets])
        numbers = _extract_numbers_from_text(all_text)
        number_issues = _verify_numbers_against_data_card(numbers, data_card)
        issues.extend(number_issues)
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues
    }


def _generate_fallback_narrative(data_card: Dict[str, Any], narrative_type: str) -> Dict[str, str]:
    """Generate simple fallback narrative when LLM fails."""
    metric = data_card.get('metric_full_name', data_card.get('metric', 'Metric'))
    chart_type = data_card.get('chart_type', 'A2')
    
    if chart_type == 'A3':
        title = f"{metric} by Credit Tiers"
        bullets = [
            f"Composition breakdown shows tier distribution",
            f"Risk profile reflected in segment mix"
        ]
        strapline = f"{metric} composition supports risk management"
    elif chart_type == 'A4':
        delta_type = data_card.get('delta_type', 'YOY')
        title = f"{metric} {delta_type.upper()} Changes"
        bullets = [
            f"{delta_type.upper()} analysis shows performance shifts",
            f"Change patterns indicate trend direction"
        ]
        strapline = f"{metric} {delta_type.upper()} changes reflect business momentum"
    else:
        title = f"{metric} Performance Analysis"
        bullets = [
            f"Data covers period {data_card['period_range']}",
            f"Latest value: {data_card['latest_value']}",
            f"Overall trend: {data_card['trend']}"
        ]
        strapline = f"{metric} shows stable performance trajectory"
    
    return {
        'title': title,
        'bullets': bullets,
        'strapline': strapline
    }


def _reorder_narratives(narratives: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Reorder narratives for logical presentation flow."""
    if len(narratives) <= 1:
        return narratives
    
    # Simple reordering: trends first, then composition, then deltas
    chart_priority = {'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4}
    
    # If narratives have chart_type info, sort by that
    # Otherwise keep original order
    try:
        return sorted(narratives, key=lambda n: chart_priority.get(n.get('chart_type', 'A2'), 5))
    except:
        return narratives


def _scan_contradictions(narratives: List[Dict[str, str]]) -> List[Tuple[int, int, str]]:
    """Scan narratives for contradictory statements."""
    contradictions = []
    
    # Simple contradiction detection
    # Look for opposing trend words
    opposing_pairs = [
        ('increasing', 'decreasing'),
        ('rising', 'falling'), 
        ('up', 'down'),
        ('growth', 'decline'),
        ('improvement', 'deterioration')
    ]
    
    for i, narr1 in enumerate(narratives):
        for j, narr2 in enumerate(narratives[i+1:], i+1):
            text1 = ' '.join([narr1.get('title', ''), narr1.get('strapline', '')] + narr1.get('bullets', []))
            text2 = ' '.join([narr2.get('title', ''), narr2.get('strapline', '')] + narr2.get('bullets', []))
            
            for word1, word2 in opposing_pairs:
                if word1.lower() in text1.lower() and word2.lower() in text2.lower():
                    contradictions.append((i, j, f"Opposing trends: {word1} vs {word2}"))
    
    return contradictions
