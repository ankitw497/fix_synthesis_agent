"""
Validation and coverage tracking for Agent 3 synthesis pipeline.
Implements CoverageLedger, product gating, and delta base availability checks.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

from synthesis_agent.utils import prev_quarter, prev_year_same_quarter, prev_month

def check_comparison_period(granularity: str, mode: str, current: str, valid: List[str]):
    expected = None
    if mode == "QOQ" and granularity == "quarterly":
        expected = prev_quarter(current)
    elif mode == "YOY" and granularity == "quarterly":
        expected = prev_year_same_quarter(current)
    elif mode == "MOM" and granularity == "monthly":
        expected = prev_month(current)
    if not expected:
        return True, None
    if expected not in valid:
        return False, f"Comparison period {expected} missing"
    return True, None

from synthesis_agent.config import (
    REVOLVING_ONLY_METRICS, DECK_FOOTERS, SynthesisConfig
)
from synthesis_agent.utils import setup_logging

logger = setup_logging("validate_plan")

# Constants for long-form tier detection
LONG_FORM_TIER_CANDIDATES = ["score_curr_tier", "score_tier", "tier", "risk_tier", "score_band"]


@dataclass
class CoverageEntry:
    """Single coverage ledger entry."""
    topic: str
    group: str
    metric: str
    chart_type: Optional[str] = None  # Specific chart type
    has_trend: bool = False
    has_delta: bool = False
    has_composition: bool = False
    has_30_60_90: bool = False
    slide_count: int = 0
    fallback_used: bool = False
    skip_reason: Optional[str] = None
    # Track chart-specific narrative status
    has_tier_narrative: bool = False  # Whether tier-specific narrative is present
    has_trend_narrative: bool = False  # Whether trend-specific narrative is present
    has_delta_narrative: bool = False  # Whether delta-specific narrative is present


class CoverageLedger:
    """
    Track coverage of metrics across topics.
    Ensures ≥1 slide per metric or logs reason.
    Tracks coverage by chart type to ensure appropriate narrative for each visualization.
    """
    
    def __init__(self, requested_metrics: Optional[Dict] = None):
        self.entries: Dict[Tuple[str, str, str, Optional[str]], CoverageEntry] = {}
        self.warnings: List[str] = []
        self.requested_metrics = requested_metrics or {}
        
        # Pre-populate entries for all requested metrics
        if requested_metrics:
            for topic, groups in requested_metrics.items():
                for group, metrics in groups.items():
                    for metric in metrics:
                        self.add_entry(topic, group, metric)
    
    def add_entry(self, topic: str, group: str, metric: str, chart_type: Optional[str] = None) -> CoverageEntry:
        """Add or get coverage entry with optional chart type."""
        key = (topic, group, metric, chart_type)
        if key not in self.entries:
            self.entries[key] = CoverageEntry(topic, group, metric, chart_type)
        return self.entries[key]
    
    def mark_coverage(self, topic: str, group: str, metric: str,
                      chart_type: str, fallback: bool = False, 
                      narrative_type: Optional[str] = None):
        """
        Mark that a metric has coverage with specific chart and narrative type.
        
        Args:
            topic: Topic name
            group: Insight group
            metric: Metric name
            chart_type: Chart type (A2, A3, A4, A5)
            fallback: Whether fallback was used
            narrative_type: Type of narrative (tier, trend, delta)
        """
        # Mark in general entry without chart type
        general_entry = self.add_entry(topic, group, metric)
        general_entry.slide_count += 1
        
        # Also mark in chart-specific entry
        specific_entry = self.add_entry(topic, group, metric, chart_type)
        specific_entry.slide_count += 1
        
        if fallback:
            general_entry.fallback_used = True
            specific_entry.fallback_used = True
        
        # Update flags based on chart type
        if chart_type in ['A2', 'A5', 'trend', 'dual_axis', 'dual-axis']:
            general_entry.has_trend = True
            specific_entry.has_trend = True
            if narrative_type == 'trend':
                specific_entry.has_trend_narrative = True
        elif chart_type in ['A3', 'composition']:
            general_entry.has_composition = True
            specific_entry.has_composition = True
            if narrative_type == 'tier':
                specific_entry.has_tier_narrative = True
        elif chart_type in ['A4', 'delta']:
            general_entry.has_delta = True
            specific_entry.has_delta = True
            if narrative_type == 'delta':
                specific_entry.has_delta_narrative = True
        elif chart_type == '30_60_90':
            general_entry.has_30_60_90 = True
            specific_entry.has_30_60_90 = True
    
    def get_missing(self) -> List[CoverageEntry]:
        """Get entries with no coverage."""
        return [e for e in self.entries.values() if e.slide_count == 0]
    
    def is_requested_metric(self, topic: str, metric: str) -> bool:
        """Check if a metric was explicitly requested in engagement spec."""
        if not self.requested_metrics:
            return False
        
        topic_lower = topic.lower()
        if topic_lower not in self.requested_metrics:
            return False
        
        for group, metrics in self.requested_metrics[topic_lower].items():
            if metric in metrics:
                return True
        return False
    
    def get_requested_group(self, topic: str, metric: str) -> Optional[str]:
        """Get the insight group for a requested metric."""
        if not self.requested_metrics:
            return None
        
        topic_lower = topic.lower()
        if topic_lower not in self.requested_metrics:
            return None
        
        for group, metrics in self.requested_metrics[topic_lower].items():
            if metric in metrics:
                return group
        return None
    
    def get_coverage_table(self) -> pd.DataFrame:
        """Generate coverage table with archetype counts."""
        data = []
        for entry in self.entries.values():
            data.append({
                'Topic': entry.topic,
                'Group': entry.group,
                'Metric': entry.metric,
                'Slides': entry.slide_count,
                'Trend': '✓' if entry.has_trend else '',
                'Delta': '✓' if entry.has_delta else '',
                'Composition': '✓' if entry.has_composition else '',
                '30-60-90': '✓' if entry.has_30_60_90 else '',
                'Fallback': '✓' if entry.fallback_used else '',
                'Skip Reason': entry.skip_reason or ''
            })
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            df = df.sort_values(['Topic', 'Group', 'Metric'])
        return df
    
    def validate_coverage(self, min_slides_per_metric: int = 1) -> Tuple[bool, List[str]]:
        """Validate that coverage requirements are met."""
        issues = []
        
        for entry in self.entries.values():
            if entry.slide_count < min_slides_per_metric:
                if entry.skip_reason:
                    logger.info(f"Skipped {entry.topic}/{entry.metric}: {entry.skip_reason}")
                else:
                    issues.append(f"{entry.topic}/{entry.metric} has {entry.slide_count} slides (min: {min_slides_per_metric})")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def print_summary(self):
        """Print coverage summary."""
        total = len(self.entries)
        covered = sum(1 for e in self.entries.values() if e.slide_count > 0)
        skipped = sum(1 for e in self.entries.values() if e.skip_reason)
        
        logger.info(f"Coverage Summary: {covered}/{total} metrics covered, {skipped} skipped")
        
        # Print table
        df = self.get_coverage_table()
        if len(df) > 0:
            print("\nCoverage Ledger:")
            print(df.to_string(index=False))


def check_product_restrictions(df: pd.DataFrame, 
                              metric: str,
                              config: SynthesisConfig) -> Tuple[bool, Optional[str]]:
    """
    Check product restrictions using LOB from df.attrs.
    
    Args:
        df: DataFrame with LOB in attrs
        metric: Metric to validate
        config: Synthesis configuration
    
    Returns:
        Tuple of (is_allowed, restriction_reason)
    """
    if not config.logic.guard_product_restrictions:
        return True, None
    
    # Get LOB from DataFrame attrs
    lob = df.attrs.get('lob', 'unknown')
    
    # Check if metric is revolving-only
    if metric in REVOLVING_ONLY_METRICS:
        if lob != 'revolving':
            reason = f"Metric '{metric}' is revolving-only but LOB is '{lob}'"
            logger.warning(reason)
            return False, reason
    
    return True, None


def check_delta_base_availability(df: pd.DataFrame,
                                 delta_type: str,
                                 period_col: str = 'period') -> Tuple[bool, List[str]]:
    """
    Check if base periods are available for delta calculations.
    
    Args:
        df: DataFrame with delta columns
        delta_type: Type of delta (yoy, qoq, mom)
        period_col: Period column name
    
    Returns:
        Tuple of (has_all_bases, list_of_missing_periods)
    """
    missing_periods = []
    
    if period_col not in df.columns:
        return False, ["Period column not found"]
    
    suffixes = [f"_{delta_type}_pct", f"_{delta_type}_pp"]
    delta_cols = [col for col in df.columns if any(col.endswith(suf) for suf in suffixes)]

    if not delta_cols:
        return True, []
    
    # Check each period for NA values in delta columns
    for _, row in df.iterrows():
        period = row[period_col]
        
        # Check if any delta is NA
        has_na = any(pd.isna(row[col]) for col in delta_cols)
        
        if has_na:
            missing_periods.append(str(period))
    
    has_all_bases = len(missing_periods) == 0
    
    if not has_all_bases:
        logger.info(f"Missing base periods for {delta_type}: {missing_periods}")
    
    return has_all_bases, missing_periods


def get_appropriate_footer(analysis_type: str,
                          has_missing_base: bool,
                          has_sparse_data: bool,
                          has_utilization: bool = False) -> str:
    """
    Get appropriate footer based on data characteristics.
    
    Args:
        analysis_type: Type of analysis
        has_missing_base: Whether base periods are missing
        has_sparse_data: Whether data is sparse
        has_utilization: Whether utilization is shown
    
    Returns:
        Footer text from DECK_FOOTERS
    """
    if has_missing_base and (
        'delta' in analysis_type.lower() or analysis_type.upper() in {'QOQ','MOM','YOY'}
    ):
        return DECK_FOOTERS['missing_base']
    elif has_utilization:
        return DECK_FOOTERS['utilization']
    elif has_sparse_data:
        return DECK_FOOTERS['sparse']
    else:
        return DECK_FOOTERS['default']


def check_snapshot_completeness(df: pd.DataFrame,
                               tier_columns: List[str],
                               period_col: str = 'period') -> Tuple[bool, List[str]]:
    """
    Check snapshot data completeness and identify suppressed tiers.
    
    Args:
        df: DataFrame with tier data
        tier_columns: List of tier column names
        period_col: Period column name
    
    Returns:
        Tuple of (is_complete, list_of_suppressed_tiers)
    """
    if not tier_columns or period_col not in df.columns:
        return False, []
    
    # Get latest period
    latest_period = df[period_col].max()
    latest_data = df[df[period_col] == latest_period]
    
    if len(latest_data) == 0:
        return False, tier_columns
    
    suppressed_tiers = []
    for tier_col in tier_columns:
        if tier_col in latest_data.columns:
            # Check if tier has NA or 0 in latest period
            value = latest_data[tier_col].iloc[0]
            if pd.isna(value) or value == 0:
                suppressed_tiers.append(tier_col)
    
    is_complete = len(suppressed_tiers) == 0
    
    if suppressed_tiers:
        logger.info(f"Suppressed tiers in latest period: {suppressed_tiers}")
    
    return is_complete, suppressed_tiers


def check_per_consumer_materiality(df: pd.DataFrame,
                                  metric: str,
                                  config: SynthesisConfig) -> Tuple[bool, str]:
    """
    Check per-consumer materiality thresholds.
    Correlation ≤ 0.92 AND mean absolute % difference ≥ 15%
    
    Args:
        df: DataFrame with per-consumer data
        metric: Metric to check
        config: Synthesis configuration
    
    Returns:
        Tuple of (is_material, reason)
    """
    # Check if we have per-consumer columns
    total_col = metric
    per_consumer_col = f"{metric}_per_cnsmr"
    
    if per_consumer_col not in df.columns or total_col not in df.columns:
        return True, "No per-consumer data"
    
    # Get non-null values
    mask = df[total_col].notna() & df[per_consumer_col].notna()
    total_vals = df.loc[mask, total_col]
    per_consumer_vals = df.loc[mask, per_consumer_col]
    
    if len(total_vals) < 3:
        return True, "Insufficient data points"
    
    # Calculate correlation
    correlation = total_vals.corr(per_consumer_vals)
    
    # Calculate mean absolute percentage difference
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_diff = np.abs((total_vals - per_consumer_vals) / total_vals)
        pct_diff = pct_diff[np.isfinite(pct_diff)]
        mean_pct_diff = np.mean(pct_diff) if len(pct_diff) > 0 else 0
    
    # Check thresholds
    corr_threshold = config.logic.per_consumer_correlation_threshold
    diff_threshold = config.logic.per_consumer_difference_threshold
    
    is_material = (correlation <= corr_threshold) and (mean_pct_diff >= diff_threshold)
    
    reason = f"Correlation: {correlation:.3f} (≤{corr_threshold}), Mean % diff: {mean_pct_diff:.1%} (≥{diff_threshold:.0%})"
    
    if is_material:
        logger.info(f"Per-consumer metric '{per_consumer_col}' is material: {reason}")
    else:
        logger.debug(f"Per-consumer metric '{per_consumer_col}' not material: {reason}")
    
    return is_material, reason


def rule_applicable(df: pd.DataFrame,
                   rule_type: str,
                   metric: str,
                   config: SynthesisConfig) -> bool:
    """
    Check if a visualization rule is applicable.
    
    Args:
        df: DataFrame to check
        rule_type: Type of rule/chart
        metric: Metric name
        config: Synthesis configuration
    
    Returns:
        Whether the rule is applicable
    """
    # Check product restrictions first
    is_allowed, _ = check_product_restrictions(df, metric, config)
    if not is_allowed:
        return False
    
    # Check data availability
    if metric not in df.columns:
        return False
    
    # Check minimum data points
    non_null = df[metric].notna().sum()
    if non_null < config.logic.min_data_points:
        return False
    
    # Check sparsity
    sparsity = 1 - (non_null / len(df))
    if sparsity > config.logic.sparsity_threshold:
        return False
    
    # Rule-specific checks
    if rule_type == 'composition':
        # Need tier columns for composition
        from synthesis_agent.io_normalize import extract_tier_columns
        tier_cols = extract_tier_columns(df)
        has_long_form_tier = any(c in df.columns for c in LONG_FORM_TIER_CANDIDATES)
        if not tier_cols and not has_long_form_tier:
            return False
    
    elif rule_type == 'delta':
        # Need delta columns
        delta_cols = [c for c in df.columns if any(d in c for d in ['_yoy_pct', '_qoq_pct', '_mom_pct'])]
        if not delta_cols:
            return False
    
    elif rule_type == 'dual_axis':
        # Need two related metrics
        if '_bal' in metric:
            partner = metric.replace('_bal', '_vol')
        elif '_vol' in metric:
            partner = metric.replace('_vol', '_bal')
        else:
            return False
        
        if partner not in df.columns:
            return False
    
    return True


def apply_fallback_strategy(df: pd.DataFrame,
                           metric: str,
                           preferred_chart: str,
                           config: SynthesisConfig) -> Optional[str]:
    """
    Apply A2 fallback strategy when preferred chart isn't applicable.
    
    Args:
        df: DataFrame
        metric: Metric name
        preferred_chart: Preferred chart type
        config: Synthesis configuration
    
    Returns:
        Fallback chart type or None
    """
    if not config.logic.allow_a2_fallback:
        return None
    
    # Check if A2 (trend) is applicable
    if rule_applicable(df, 'trend', metric, config):
        logger.info(f"Using A2 fallback for {metric} (preferred: {preferred_chart})")
        return 'A2'
    
    return None


def validate_plan_consistency(coverage_ledger: CoverageLedger,
                             config: SynthesisConfig) -> Tuple[bool, List[str]]:
    """
    Validate overall plan consistency.
    
    Args:
        coverage_ledger: Coverage tracking
        config: Synthesis configuration
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check coverage requirements
    is_valid, coverage_issues = coverage_ledger.validate_coverage()
    issues.extend(coverage_issues)
    
    # Check that each metric has at least one slide
    missing = coverage_ledger.get_missing()
    for entry in missing:
        if not entry.skip_reason:
            issues.append(f"No coverage for {entry.topic}/{entry.metric}")
    
    # Validate archetype distribution
    df = coverage_ledger.get_coverage_table()
    if len(df) > 0:
        # Check that we have diverse chart types
        chart_types = ['Trend', 'Delta', 'Composition']
        for chart_type in chart_types:
            if df[chart_type].str.contains('✓').sum() == 0:
                logger.warning(f"No {chart_type} charts in plan")
    
    is_valid = len(issues) == 0
    return is_valid, issues
