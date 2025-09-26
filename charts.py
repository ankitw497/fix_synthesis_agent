"""
Visualization functions for Agent 3 synthesis pipeline.
Implements A2/A3/A4/A5 chart types with caching and strict visual standards.
"""

import io
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Handle matplotlib imports gracefully
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = None
    plt = None

# Handle data library imports
try:
    import numpy as np
    import pandas as pd
    HAS_DATA_LIBS = True
except ImportError:
    HAS_DATA_LIBS = False
    np = None
    pd = None

from synthesis_agent.config import (
    STYLE_COLORS, TIER_ORDER, DELTA_CLUSTERS, SynthesisConfig
)
from synthesis_agent.utils import (
    fmt_currency, fmt_percent, fmt_delta, get_cache_key,
    detect_period_gaps, setup_logging, pretty_label, fmt_value,
    canonical_sort_periods,
)
from synthesis_agent.io_normalize import (
    extract_tier_columns,
    compute_missing_deltas,
    is_rate_metric,
    infer_percent_scale,
)

logger = setup_logging("charts")

# Global figure cache
FIGURE_CACHE: Dict[str, bytes] = {}

# Business-friendly axis labels
BUSINESS_AXIS = {
    "tot_acct_bal": "Total Account Balance (B)",
    "avg_acct_bal": "Average Balance (K)",
    "tot_cnsmr_cnts": "Total Consumers",
    "deliq_30_acct_rate": "30-Day Delinquency Rate (%)",
    "deliq_60_acct_rate": "60-Day Delinquency Rate (%)",
    "deliq_90_acct_rate": "90-Day Delinquency Rate (%)",
    "cnsmr_cnts_w_deliq_bal_30_rate": "Consumer 30+ DPD Rate (%)",
    "cnsmr_cnts_w_deliq_bal_60_rate": "Consumer 60+ DPD Rate (%)",
    "cnsmr_cnts_w_deliq_bal_90_rate": "Consumer 90+ DPD Rate (%)",
    "deliq_30_acct_bal_rate": "30+ DPD Balance Rate (%)",
    "deliq_60_acct_bal_rate": "60+ DPD Balance Rate (%)",
    "deliq_90_acct_bal_rate": "90+ DPD Balance Rate (%)",
}


def _axis_title(metric: Optional[str], is_rate: bool) -> str:
    """Return a human friendly y-axis label."""
    if not metric:
        return "Value"

    m = (metric or "").lower()
    if m in BUSINESS_AXIS:
        return BUSINESS_AXIS[m]

    pretty = pretty_label(metric)
    if is_rate or m.endswith("_rate") or m.endswith("_pct"):
        if "%" in pretty:
            return pretty
        if "rate" in pretty.lower():
            return pretty if pretty.endswith(")") or pretty.endswith("%") else f"{pretty} (%)"
        return f"{pretty} (%)"

    currency_tokens = ("bal", "balance", "amt", "amount", "dollar", "open_to_buy", "line")
    if any(tok in m for tok in currency_tokens):
        return f"{pretty} ($)"

    count_tokens = ("cnt", "count", "volume", "vol", "num", "number", "acct", "accounts")
    if any(tok in m for tok in count_tokens):
        return pretty

    return pretty


def _legend_bottom(fig, ax, config: SynthesisConfig, handles=None, labels=None, ncol: Optional[int] = None) -> None:
    """Place a legend beneath the chart with reserved space."""
    if not HAS_MATPLOTLIB:
        return

    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    if not getattr(config.render, 'legend_bottom', True):
        ax.legend(handles, labels, frameon=False)
        return

    if ncol is None:
        series_count = len(handles)
        if series_count <= 2:
            ncol = series_count
        else:
            ncol = min(3, (series_count + 1) // 2)
    legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                       frameon=False, ncol=max(1, ncol))
    pad = getattr(config.render, 'legend_bottom_pad', 0.22)
    try:
        fig.subplots_adjust(bottom=pad)
    except Exception:
        pass
    return legend

# --- PPTX chart spec builders ---

def _spec(
    kind,
    categories,
    series,
    title=None,
    y_is_percent=False,
    color_overrides=None,
    box=None,
    x_axis_title=None,
    y_axis_title=None,
):
    return {
        "kind": kind,
        "categories": list(categories),
        "series": list(series),
        "title": title,
        "y_is_percent": y_is_percent,
        "color_overrides": color_overrides or {},
        "box": box,
        "x_axis_title": x_axis_title,
        "y_axis_title": y_axis_title,
    }

def build_trend_line_spec(df, period_col, series_cols, title, config):
    metric = series_cols[0] if series_cols else None
    agg = df.groupby(period_col, as_index=False)[series_cols].sum()
    cats = agg[period_col].tolist()
    series = []
    for col in series_cols:
        series.append({"name": col, "values": agg[col].tolist()})
    is_rate = is_rate_metric(metric) if metric else False
    return _spec(
        "line",
        cats,
        series,
        title=title,
        y_is_percent=is_rate,
        x_axis_title=None,
        y_axis_title=_axis_title(metric, is_rate),
    )

def build_compare_grouped_by_tier_spec(df, metric, period_col, tier_col, allowed_tiers,
                                       current_period, compare_period, config, is_rate):
    snap = df[df[period_col].isin([compare_period, current_period])].copy()
    snap[tier_col] = snap[tier_col].astype(str).str.upper()
    tiers = [t.upper() for t in (allowed_tiers or [])]
    if not tiers:
        tiers = snap[tier_col].dropna().unique().tolist()
    cats, cur_vals, cmp_vals = [], [], []
    for t in tiers:
        sub = snap[snap[tier_col] == t]
        cats.append(_clean_tier_name(t))
        cur = sub[sub[period_col] == current_period][metric].sum() if not sub.empty else None
        cm = sub[sub[period_col] == compare_period][metric].sum() if not sub.empty else None
        cur_vals.append(cur)
        cmp_vals.append(cm)
    co = {
        "Current Period": STYLE_COLORS.get("CURRENT_PERIOD"),
        "Comparison Period": STYLE_COLORS.get("PRIOR_PERIOD")
    }
    series = [
        {"name": "Comparison Period", "values": cmp_vals},
        {"name": "Current Period", "values": cur_vals},
    ]
    return _spec(
        "grouped_bar",
        cats,
        series,
        y_is_percent=is_rate,
        color_overrides=co,
        x_axis_title="Credit Tier",
        y_axis_title=_axis_title(metric, is_rate),
    )

def prepare_dpd_trend_dataframe(df, period_col: str, cols_30_60_90: List[str]):
    """Return a de-duplicated, chronologically sorted delinquency frame."""

    if not HAS_DATA_LIBS or df is None:
        return df

    if period_col not in df.columns:
        return df.iloc[0:0] if hasattr(df, "iloc") else df

    frame = df.copy()
    try:
        frame = frame.dropna(subset=[period_col])
    except Exception:
        pass

    try:
        frame[period_col] = frame[period_col].astype(str)
    except Exception:
        frame[period_col] = frame[period_col]

    value_cols = [c for c in cols_30_60_90 if c in frame.columns]
    if not value_cols:
        return frame[[period_col]].iloc[0:0]

    tier_col = next((c for c in ("score_curr_tier", "score_tier", "tier") if c in frame.columns), None)
    if tier_col:
        try:
            tiers = frame[tier_col].astype(str).str.upper()
            frame[tier_col] = tiers
            total_mask = tiers.isin({"TOTAL", "TOT", "ALL", "PORTFOLIO", "OVERALL"})
        except Exception:
            total_mask = None
        if total_mask is not None and total_mask.any():
            frame = frame.loc[total_mask, [period_col] + value_cols]
        else:
            frame = frame[[period_col] + value_cols]
    else:
        frame = frame[[period_col] + value_cols]

    try:
        frame = frame.groupby(period_col, as_index=False).mean(numeric_only=True)
    except TypeError:
        frame = frame.groupby(period_col, as_index=False).mean()

    cats = canonical_sort_periods(frame[period_col].tolist())
    if cats:
        try:
            frame = (
                frame.set_index(period_col)
                .reindex(cats)
                .reset_index()
            )
        except Exception:
            frame = frame.sort_values(period_col)

    for col in value_cols:
        try:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        except Exception:
            continue

    return frame[[period_col] + value_cols]


def build_dpd_trend_spec(df, period_col, cols_30_60_90, title, config):
    frame = prepare_dpd_trend_dataframe(df, period_col, cols_30_60_90)
    if not HAS_DATA_LIBS or frame is None or frame.empty:
        cats = df[period_col].tolist() if period_col in getattr(df, "columns", []) else []
    else:
        cats = frame[period_col].tolist()

    labels = ["30+ DPD", "60+ DPD", "90+ DPD"]
    co = {
        "30+ DPD": STYLE_COLORS.get("DPD_30"),
        "60+ DPD": STYLE_COLORS.get("DPD_60"),
        "90+ DPD": STYLE_COLORS.get("DPD_90"),
    }
    source = frame if HAS_DATA_LIBS and frame is not None and not frame.empty else df
    series = []
    for idx, col in enumerate(cols_30_60_90):
        if col not in getattr(source, "columns", []):
            continue
        values = source[col].tolist()
        if HAS_DATA_LIBS:
            values = [float(v) if pd.notna(v) else None for v in pd.to_numeric(values, errors="coerce")]
        label = labels[idx] if idx < len(labels) else col
        series.append({"name": label, "values": values})

    return _spec(
        "line",
        cats,
        series,
        title=title,
        y_is_percent=True,
        color_overrides=co,
        x_axis_title=None,
        y_axis_title="Rate (%)",
    )

def build_dpd_severity_spec(df_latest, tier_col, cols_30_60_90, allowed_tiers, config):
    if tier_col not in df_latest.columns:
        return _spec("grouped_bar", [], [], y_is_percent=True, x_axis_title="Credit Tier", y_axis_title="Rate (%)")

    tiers = [t.upper() for t in (allowed_tiers or [])]
    if not tiers:
        tiers = df_latest[tier_col].astype(str).str.upper().dropna().unique().tolist()

    cats = [_clean_tier_name(t) for t in tiers]
    column_map = [
        ("30+ DPD", cols_30_60_90[0] if len(cols_30_60_90) > 0 else None, "DPD_30"),
        ("60+ DPD", cols_30_60_90[1] if len(cols_30_60_90) > 1 else None, "DPD_60"),
        ("90+ DPD", cols_30_60_90[2] if len(cols_30_60_90) > 2 else None, "DPD_90"),
    ]

    series = []
    color_overrides = {}
    tier_series = df_latest.copy()
    tier_series[tier_col] = tier_series[tier_col].astype(str).str.upper()

    for label, column, color_key in column_map:
        if not column or column not in tier_series.columns:
            continue
        values = []
        for t in tiers:
            sub = tier_series[tier_series[tier_col] == t]
            if column in sub.columns and not sub.empty:
                values.append(float(sub[column].sum()))
            else:
                values.append(None)
        series.append({"name": label, "values": values})
        hx = STYLE_COLORS.get(color_key)
        if hx:
            color_overrides[label] = hx

    return _spec(
        "grouped_bar",
        cats,
        series,
        y_is_percent=True,
        color_overrides=color_overrides,
        x_axis_title="Credit Tier",
        y_axis_title="Rate (%)",
    )

def build_term_distribution_spec(df_latest, term_bucket_col, allowed_tiers, config):
    tiers = [t.upper() for t in (allowed_tiers or [])]
    if not tiers:
        tiers = df_latest['score_curr_tier'].astype(str).str.upper().dropna().unique().tolist()
    buckets = getattr(config, "viz_defaults", {}).get("term_buckets", {}).get("default", [])
    cats = buckets
    series = []
    for t in tiers:
        vals = []
        sub_t = df_latest[df_latest['score_curr_tier'].astype(str).str.upper() == t]
        for b in buckets:
            vals.append(float(sub_t.loc[sub_t[term_bucket_col] == b, 'tot_acct_vol'].sum()))
        series.append({"name": t.title(), "values": vals})
    return _spec(
        "stacked_col",
        cats,
        series,
        y_is_percent=False,
        x_axis_title="Term Bucket",
        y_axis_title="Volume",
    )

def build_score_histogram_spec(df, bin_col, value_col, is_percent, title, config):
    cats = df[bin_col].tolist()
    vals = df[value_col].tolist()
    series = [{"name": "Share of Consumers" if is_percent else "Consumers", "values": vals}]
    return _spec(
        "grouped_bar",
        cats,
        series,
        title=title,
        y_is_percent=is_percent,
        x_axis_title="Score Bin",
        y_axis_title="Share (%)" if is_percent else "Consumers",
    )


def _clean_tier_name(name: str) -> str:
    """Return a human readable tier label."""
    if name is None:
        return ""
    return str(name).replace("_", " ").title()


def build_multi_period_grouped_by_tier_spec(
    df,
    metric,
    period_col,
    tier_col,
    tiers,
    periods,
    config,
    is_rate,
):
    """Grouped bar chart comparing a metric across tiers for multiple periods."""

    frame = df.copy()
    if period_col in frame.columns:
        frame[period_col] = frame[period_col].astype(str)
    normalized_periods = [str(p) for p in (periods or [])]

    tiers = [t.upper() for t in (tiers or [])]
    if not tiers:
        tiers = frame[tier_col].astype(str).str.upper().dropna().unique().tolist()
    cats = [_clean_tier_name(t) for t in tiers]
    series = []
    for p in normalized_periods:
        vals = []
        snap = frame[frame[period_col] == p]
        for t in tiers:
            sub = snap[snap[tier_col].astype(str).str.upper() == t]
            vals.append(float(sub[metric].sum()) if not sub.empty and metric in sub else None)
        series.append({"name": p, "values": vals})
    return _spec(
        "grouped_bar",
        cats,
        series,
        y_is_percent=is_rate,
        x_axis_title="Credit Tier",
        y_axis_title=_axis_title(metric, is_rate),
    )


# Distinct tier palette (no grays)
TIER_COLORS = {
    "SUBPRIME":     "#E53935",
    "NEAR_PRIME":   "#F59E0B",
    "PRIME":        "#FBBF24",
    "PRIME_PLUS":   "#34D399",
    "SUPER_PRIME":  "#60A5FA",
    "UNSCORED":     "#CBD5E1",
    "OTHER":        "#94A3B8",
    "TOTAL":        "#475569"
}


def select_delta_col(metric: str, delta_type: str, df: pd.DataFrame) -> Tuple[str, str]:
    """Return (column name, y-axis label) for given metric and delta type."""
    if is_rate_metric(metric):
        col = f"{metric}_{delta_type}_pp"
        ylabel = f"{delta_type.upper()} Change (pp)"
    else:
        col = f"{metric}_{delta_type}_pct"
        ylabel = f"{delta_type.upper()} Change (%)"
    return col, ylabel


def _norm_tier_key(name: str) -> str:
    """Normalize tier name to match canonical keys in TIER_ORDER/STYLE_COLORS."""
    if name is None:
        return ""
    s = str(name).strip()
    # unify separators and case
    s = s.replace("-", " ").replace("/", " ")
    s = "_".join([t for t in s.split() if t])  # collapse spaces to underscores
    return s.upper()


def _fmt_billions(x, pos=None):
    """Format axis values in billions."""
    try:
        if abs(x) >= 1e9:
            return f"{x/1e9:,.1f}B"
        elif abs(x) >= 1e6:
            return f"{x/1e6:,.0f}M"
        elif abs(x) >= 1e3:
            return f"{x/1e3:,.0f}K"
        else:
            return f"{x:,.0f}"
    except Exception:
        return f"{x:,.0f}"


def _infer_scale(vmax: float) -> Tuple[int, str]:
    """Return (scale, suffix) based on magnitude."""
    av = abs(vmax)
    if av >= 1_000_000_000:
        return 1_000_000_000, "B"
    if av >= 1_000_000:
        return 1_000_000, "M"
    if av >= 1_000:
        return 1_000, "K"
    return 1, ""


def _metric_label(metric: str, series) -> Tuple[str, Any]:
    """Derive axis label and formatter from metric type and data."""
    from matplotlib.ticker import FuncFormatter

    name = metric.lower()
    pretty = pretty_label(metric)
    vmax = float(pd.Series(series).abs().max() or 0.0)

    # Rate metrics → percent formatting
    if is_rate_metric(metric):
        def _fmt_percent(v, _pos=None):
            return f"{(v*100 if abs(v) <= 1 else v):.0f}%"
        return f"{pretty} (%)", FuncFormatter(_fmt_percent)

    scale, suf = _infer_scale(vmax)

    if any(k in name for k in ("bal", "balance", "amt", "amount")):
        def _fmt_cur(v, _pos=None):
            return f"${v/scale:,.0f}{suf}"
        return f"{pretty} ({'$'+suf if suf else '$'})", FuncFormatter(_fmt_cur)

    if any(k in name for k in ("cnt", "count", "vol", "volume")):
        def _fmt_int(v, _pos=None):
            return f"{v/scale:,.0f}{suf}"
        label = f"{pretty} ({suf})" if suf else pretty
        return label, FuncFormatter(_fmt_int)

    def _fmt_default(v, _pos=None):
        return f"{v:,.0f}"
    return pretty, FuncFormatter(_fmt_default)


def _modernize_axes(ax, metric: str, series) -> None:
    """Apply axes formatting using data-driven labels."""
    from matplotlib.ticker import MaxNLocator

    # grid & spines
    grid_color = STYLE_COLORS.get("gridlines", "#E5E7EB")
    ax.grid(True, axis="y", color=grid_color, linewidth=0.3, alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=10)

    ylabel, formatter = _metric_label(metric, series)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(6))


def compute_delta_column(df: pd.DataFrame, metric: str, period_col: str, 
                         delta_type: str = 'yoy') -> pd.Series:
    """
    Compute delta column if missing from data.
    
    Args:
        df: DataFrame with metric data
        metric: Base metric name
        period_col: Period column name
        delta_type: Type of delta ('yoy', 'qoq', 'mom')
    
    Returns:
        Series with computed delta percentages
    """
    if metric not in df.columns:
        raise ValueError(f"Base metric {metric} not found in DataFrame")
    
    # Sort by period to ensure correct ordering
    df_sorted = df.sort_values(period_col)
    
    # Determine lag period based on delta type
    if delta_type == 'yoy':
        lag = 4 if 'Q' in str(df_sorted[period_col].iloc[0]) else 12
    elif delta_type == 'qoq':
        lag = 1
    elif delta_type == 'mom':
        lag = 1
    else:
        raise ValueError(f"Unknown delta type: {delta_type}")
    
    base_values = pd.to_numeric(df_sorted[metric], errors="coerce")
    lagged_values = base_values.shift(lag)

    if is_rate_metric(metric):
        # percentage-point change (series already in % or decimal)
        scale = infer_percent_scale(base_values)
        delta_pp = (base_values - lagged_values) * (100.0 if scale == 'decimal' else 1.0)
        return delta_pp

    delta_pct = ((base_values / lagged_values) - 1) * 100
    return delta_pct

# Tier column candidates for A3 charts
TIER_COL_CANDIDATES = ["score_curr_tier", "score_tier", "tier", "risk_tier"]

# Set consistent matplotlib defaults for clean visuals with IMPROVED READABILITY
if HAS_MATPLOTLIB:
    matplotlib.rcParams.update({
        "figure.dpi": 220,
        "font.size": 12,           # Increased from 11
        "axes.titlesize": 16,       # Increased from 12 for better visibility
        "axes.labelsize": 12,       # Increased from 11
        "xtick.labelsize": 11,      # Increased from 10
        "ytick.labelsize": 11,      # Increased from 10
        "legend.fontsize": 11,      # Ensure legends are readable
        "figure.titlesize": 18,     # Large titles for charts
    })


def _aggregate_metric(df: pd.DataFrame, period_col: str, metric: str) -> pd.DataFrame:
    """
    Aggregate metric values by period using metric-aware aggregation.
    Prefers TOTAL tier if present, uses mean for rates/averages, sum otherwise.
    
    Args:
        df: DataFrame with data
        period_col: Period column name
        metric: Metric column name
    
    Returns:
        Aggregated DataFrame
    """
    # First, prefer TOTAL tier if present
    if 'score_curr_tier' in df.columns:
        tot_df = df[df['score_curr_tier'].astype(str).str.upper() == 'TOTAL']
        if not tot_df.empty:
            return tot_df.groupby(period_col, as_index=False, observed=True)[metric].first()
    
    # Determine aggregation method based on metric name
    metric_lower = metric.lower()
    if (metric_lower.startswith('avg_') or 
        metric_lower.endswith('_rate') or 
        metric_lower.endswith('_pct') or
        'per_' in metric_lower or
        'utilization' in metric_lower):
        # Use mean for averages, rates, percentages
        agg_func = 'mean'
    else:
        # Use sum for totals (balances, counts, etc.)
        agg_func = 'sum'
    
    return df.groupby(period_col, as_index=False, observed=True)[metric].agg(agg_func)


def _find_tier_col(df: pd.DataFrame) -> Optional[str]:
    """
    Find tier column in DataFrame for composition charts.
    
    Args:
        df: DataFrame to search
    
    Returns:
        Tier column name or None
    """
    for c in TIER_COL_CANDIDATES:
        if c in df.columns:
            return c
    
    # Last resort: any categorical-like column with <10 unique values
    for c in df.columns:
        try:
            if df[c].dtype == "object" and df[c].nunique(dropna=True) <= 10:
                logger.info(f"Using {c} as tier column (categorical with {df[c].nunique()} values)")
                return c
        except Exception:
            pass
    
    return None


def _save_figure_to_cache(fig: Figure, cache_key: str, dpi: int) -> bytes:
    """
    Save figure to cache and return PNG bytes.
    
    Args:
        fig: Matplotlib figure
        cache_key: Cache key for the figure
        dpi: DPI for rendering
    
    Returns:
        PNG bytes
    """
    buf = io.BytesIO()
    # Remove bbox_inches="tight" to avoid inconsistent cropping
    fig.savefig(buf, format="png", dpi=dpi)
    png = buf.getvalue()
    FIGURE_CACHE[cache_key] = png
    plt.close(fig)
    logger.debug(f"Saved figure to cache: {cache_key}")
    return png


def clear_figure_cache():
    """Clear the figure cache."""
    FIGURE_CACHE.clear()
    logger.info("Cleared figure cache")


def get_cached_figure(cache_key: str) -> Optional[bytes]:
    """
    Get cached figure bytes.
    
    Args:
        cache_key: Cache key
    
    Returns:
        PNG bytes or None if not cached
    """
    return FIGURE_CACHE.get(cache_key)


def check_overlay_materiality(main_series: pd.Series, 
                             overlay_series: pd.Series,
                             config: SynthesisConfig) -> Tuple[bool, str]:
    """
    Check if per-consumer overlay meets materiality thresholds.
    
    Args:
        main_series: Main metric series
        overlay_series: Per-consumer overlay series
        config: Synthesis configuration
    
    Returns:
        Tuple of (is_material, reason)
    """
    if not HAS_DATA_LIBS:
        return False, "Data libraries not available"
        
    # Calculate correlation
    try:
        correlation = main_series.corr(overlay_series)
        if pd.isna(correlation):
            return False, "Cannot compute correlation"
    except:
        return False, "Correlation computation failed"
    
    # Check correlation threshold (must be ≤ 0.92)
    if abs(correlation) > config.logic.per_consumer_correlation_threshold:
        return False, f"Correlation {correlation:.3f} exceeds threshold {config.logic.per_consumer_correlation_threshold}"
    
    # Calculate mean absolute percentage difference
    try:
        # Align series
        aligned_main = main_series.dropna()
        aligned_overlay = overlay_series.reindex(aligned_main.index).dropna()
        
        # Calculate percentage differences
        pct_diffs = abs((aligned_overlay - aligned_main) / aligned_main) * 100
        mean_diff = pct_diffs.mean()
        
        if pd.isna(mean_diff):
            return False, "Cannot compute percentage difference"
    except:
        return False, "Difference computation failed"
    
    # Check difference threshold (must be ≥ 15%)
    if mean_diff < config.logic.per_consumer_difference_threshold * 100:
        return False, f"Mean difference {mean_diff:.1f}% below threshold {config.logic.per_consumer_difference_threshold * 100}%"
    
    # Both thresholds met
    logger.info(f"Per-consumer overlay is material: corr={correlation:.3f}, diff={mean_diff:.1f}%")
    return True, "Material"


def check_overlay_legibility(df: pd.DataFrame,
                            n_series: int,
                            config: SynthesisConfig) -> bool:
    """
    Check if overlay would be legible based on density.
    
    Args:
        df: DataFrame with data
        n_series: Number of series to plot
        config: Synthesis configuration
    
    Returns:
        True if legible, False if too dense
    """
    n_periods = len(df)
    density = n_series * n_periods
    
    # Check against legibility threshold
    # Threshold is a fraction, so scale appropriately
    max_density = config.render.overlay_legibility_threshold * 100
    
    if density > max_density:
        logger.warning(f"Overlay density {density} exceeds threshold {max_density}")
        return False
    
    return True


def line_trend_A2(df: pd.DataFrame,
                 metric: str,
                 period_col: str,
                 config: SynthesisConfig,
                 show_index: bool = False,
                 show_shares: bool = False,
                 add_per_consumer: bool = False) -> Tuple[Figure, str, bytes]:
    """
    A2: Trend chart. If tier columns exist, plot a multi-series "spaghetti" with per-tier colors.
    Otherwise, plot single series. Includes modern formatting and clean axes.
    
    Args:
        df: DataFrame with trend data
        metric: Metric column name
        period_col: Period column name
        config: Synthesis configuration
        show_index: Whether to show indexed values
        show_shares: Whether to show as shares
        add_per_consumer: Whether to add per-consumer overlay
    
    Returns:
        Tuple of (Figure, cache_key, png_bytes)
    """
    if not HAS_MATPLOTLIB or not HAS_DATA_LIBS:
        logger.error("Required libraries not available")
        return None, "libraries_unavailable", b""
    # Generate cache key
    cache_params = {'chart_type': 'A2', 'metric': metric}
    cache_key = get_cache_key(df[[c for c in df.columns if c in (period_col, metric)]], cache_params)
    
    # Check cache
    if cache_key in FIGURE_CACHE:
        logger.info(f"Using cached A2 figure for {metric}")
        return None, cache_key, get_cached_figure(cache_key)
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.render.figure_size, dpi=config.render.figure_dpi)
    
    plot_df = df.copy()
    if pd.api.types.is_period_dtype(plot_df[period_col]):
        plot_df[period_col] = plot_df[period_col].astype(str)

    # Detect tiers and draw multi-series if available
    tier_cols = extract_tier_columns(plot_df)
    allowed = set(getattr(config.logic, 'allowed_tiers', []))
    drew_tiers = False
    
    if tier_cols:
        # We have tier columns in wide format - plot directly
        # Filter to columns that contain our metric pattern
        metric_tier_cols = [col for col in tier_cols if metric in col]
        if allowed:
            metric_tier_cols = [c for c in metric_tier_cols
                                if _norm_tier_key(c.replace(metric + '_', '').replace('_' + metric, '')) in allowed]
        if metric_tier_cols:
            # Build mapping from normalized key -> original column
            col_map = {_norm_tier_key(c.replace(metric + '_', '').replace('_' + metric, '')): c
                      for c in metric_tier_cols}
            # Respect canonical order but use actual column names
            ordered = [col_map[k] for k in TIER_ORDER if k in col_map and (not allowed or k in allowed)]
            if not ordered:
                ordered = metric_tier_cols
            
            # Plot each tier series
            for tier_col in ordered[:getattr(config.render, 'max_spaghetti_series', 6)]:
                if tier_col in plot_df.columns:
                    series_df = plot_df[[period_col, tier_col]].dropna()
                    if len(series_df) >= config.logic.min_data_points:
                        # Extract tier name from column and normalize for color lookup
                        display_name = tier_col.replace(metric + '_', '').replace('_' + metric, '')
                        tier_key = _norm_tier_key(display_name)
                        
                        ax.plot(series_df[period_col], series_df[tier_col],
                               linewidth=2.4, marker='o', alpha=0.95,
                               label=display_name.replace('_', ' ').title(),
                               color=TIER_COLORS.get(tier_key, TIER_COLORS['OTHER']))
                        drew_tiers = True
    elif 'score_curr_tier' in plot_df.columns or 'score_tier' in plot_df.columns:
        # Long format with tiers - pivot and plot
        tier_col = 'score_curr_tier' if 'score_curr_tier' in plot_df.columns else 'score_tier'
        long_cols = [period_col, tier_col, metric]
        long_df = plot_df[[c for c in long_cols if c in plot_df.columns]].copy()

        if allowed:
            long_df[tier_col] = long_df[tier_col].astype(str)
            long_df = long_df[long_df[tier_col].str.upper().isin(allowed)]

        if len(long_df.dropna(subset=[metric])) >= config.logic.min_data_points:
            # Pivot to wide format
            wide = (long_df.pivot_table(index=period_col, columns=tier_col, values=metric, aggfunc='sum')
                          .fillna(np.nan))

            if allowed:
                wide = wide[[c for c in wide.columns if _norm_tier_key(c) in allowed]]

            # Build mapping from normalized key -> original column name
            col_map = {_norm_tier_key(c): c for c in wide.columns}
            # Respect canonical order but use actual column names
            ordered = [col_map[k] for k in TIER_ORDER if k in col_map and (not allowed or k in allowed)]
            if not ordered:
                ordered = list(wide.columns)

            # Plot each tier
            for tier in ordered[:getattr(config.render, 'max_spaghetti_series', 6)]:
                series = wide[tier].dropna()
                if len(series) >= 2:  # Need at least 2 points for a line
                    tier_key = _norm_tier_key(tier)
                    ax.plot(series.index, series.values,
                           linewidth=2.4, marker='o', alpha=0.95,
                           label=tier.replace('_', ' ').title(),
                           color=TIER_COLORS.get(tier_key, TIER_COLORS['OTHER']))
                    drew_tiers = True
    
    # If no tiers or tier plotting failed, fall back to single series
    if not drew_tiers:
        # Aggregate if needed
        base_df = plot_df[[period_col, metric]].copy()
        if base_df[period_col].duplicated(keep=False).any():
            base_df = _aggregate_metric(base_df, period_col, metric)

        base_df = base_df.dropna()
        if len(base_df) > 0:
            ax.plot(base_df[period_col], base_df[metric],
                   linewidth=2.6, marker='o', alpha=0.98,
                   label=pretty_label(metric),
                   color=TIER_COLORS.get('SUPER_PRIME', TIER_COLORS['TOTAL']))
    
    # Modern axes formatting with enhanced labels
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(STYLE_COLORS.get('axes', '#333333'))
    ax.spines['bottom'].set_color(STYLE_COLORS.get('axes', '#333333'))
    grid_color = STYLE_COLORS.get('gridlines', '#E5E7EB')
    ax.grid(True, axis='y', which='major', color=grid_color, alpha=0.2)
    try:
        ax.minorticks_on()
        ax.grid(True, axis='y', which='minor', color=grid_color, alpha=0.12)
    except Exception:
        pass
    
    # Smart labels based on metric name
    ax.set_xlabel("Period")
    y_series = base_df[metric] if 'base_df' in locals() and metric in base_df else plot_df[metric]
    _modernize_axes(ax, metric, y_series)
    
    # Smart tick formatting
    set_nice_ticks(ax, config.render.tick_steps_min, config.render.tick_steps_max)
    
    fig.tight_layout()
    _legend_bottom(fig, ax, config)

    # Cache figure and get PNG bytes
    png = _save_figure_to_cache(fig, cache_key, config.render.figure_dpi)
    
    # Explicit token for notebook metric counting
    logger.info(f"Generated visualization: A2 for {metric}")
    
    return fig, cache_key, png


def stacked100_A3(df: pd.DataFrame,
                 metric_base: str,
                 period_col: str,
                 config: SynthesisConfig) -> Tuple[Figure, str, bytes]:
    """
    Create A3 100% stacked area chart.
    CRITICAL: Labels ALWAYS ON inside segments.
    
    Args:
        df: DataFrame with composition data
        metric_base: Base metric name
        period_col: Period column name
        config: Synthesis configuration
    
    Returns:
        Tuple of (Figure, cache_key, png_bytes)
    """
    # Robust tier detection with aliasing
    tier_col = None
    for cand in ("score_curr_tier", "score_tier", "score_band", "tier"):
        if cand in df.columns:
            if cand != "score_curr_tier":
                df = df.rename(columns={cand: "score_curr_tier"})
            tier_col = "score_curr_tier"
            break
    
    # If no tier column found, try extract_tier_columns for wide format
    tier_cols = extract_tier_columns(df)
    allowed = set(getattr(config.logic, 'allowed_tiers', []))
    if not tier_cols and not tier_col:
        # FAIL FAST: No heuristic tier discovery
        raise ValueError("A3 requires an explicit tier column (score_curr_tier/score_tier/score_band/tier). "
                        "Configure 'tier_column' in config or ensure data has the correct column.")
    
    if not tier_cols and not tier_col:
        raise ValueError("A3 requires a score tier column (score_curr_tier/score_tier/score_band/tier).")
    
    # If we have long format data, pivot it
    if not tier_cols and tier_col:
        # Ensure we have valid data
        df = df.copy()
        df = df[df[metric_base].notna()]
        if period_col not in df.columns:
            raise ValueError("A3 requires a 'period' column. Use normalize_period first.")
        
        # FIX: Use ALL periods for time-series composition, not just one
        # Filter to periods with at least 2 tiers
        periods_with_tiers = (
            df.groupby(period_col)[tier_col]
              .nunique()
              .loc[lambda s: s >= 2]
              .index
        )
        if len(periods_with_tiers) == 0:
            raise ValueError("A3: no period has at least 2 tiers of data.")
        
        # Use ALL valid periods for composition trends
        valid_data = df[df[period_col].isin(periods_with_tiers)]
        
        # Pivot to wide format with ALL periods
        pivot = (valid_data.pivot_table(index=period_col,
                                        columns=tier_col,
                                        values=metric_base,
                                        aggfunc='sum')
                           .fillna(0)
                           .reset_index())
        
        # Order tier columns using canonical order if available
        from synthesis_agent.config import TIER_ORDER
        cols = [c for c in TIER_ORDER if c in pivot.columns]
        if allowed:
            cols = [c for c in cols if c in allowed]
        if cols:
            df = pivot[[period_col] + cols]
            tier_cols = cols
        else:
            # Fallback to all columns except period
            tier_cols = [c for c in pivot.columns if c != period_col and (not allowed or _norm_tier_key(c) in allowed)]
            df = pivot[[period_col] + tier_cols]
        logger.info(f"Auto-pivoted {tier_col} to wide format for A3 chart using {len(periods_with_tiers)} periods")
    
    # Select composition base
    # Filter any wide-format tier columns to allowed set
    if tier_cols and allowed:
        tier_cols = [c for c in tier_cols if _norm_tier_key(c) in allowed]
        df = df[[period_col] + tier_cols]

    from synthesis_agent.io_normalize import select_composition_base
    composition_base = select_composition_base(df, metric_base, period_col)
    logger.info(f"A3 chart using composition_base: {composition_base}")
    
    # Generate cache key
    cache_params = {
        'chart_type': 'A3',
        'metric_base': metric_base,
        'composition_base': composition_base,
        'tiers': tier_cols
    }
    cache_key = get_cache_key(df[tier_cols + [period_col]], cache_params)
    
    # Check cache
    if cache_key in FIGURE_CACHE:
        logger.info(f"Using cached A3 figure for {metric_base}")
        return None, cache_key, get_cached_figure(cache_key)
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.render.figure_size, dpi=config.render.figure_dpi)
    
    # Prepare data - calculate percentages
    plot_df = df[[period_col] + tier_cols].copy()
    
    # Convert Period objects to string for matplotlib compatibility
    if pd.api.types.is_period_dtype(plot_df[period_col]):
        plot_df[period_col] = plot_df[period_col].astype(str)
    
    plot_df = plot_df.set_index(period_col)
    
    # Convert to percentages
    row_sums = plot_df.sum(axis=1)
    for col in tier_cols:
        plot_df[col] = plot_df[col] / row_sums * 100
    
    # Order tiers by TIER_ORDER then latest value
    ordered_tiers = order_legend_items(tier_cols, plot_df.iloc[-1].to_dict())
    
    # Create stacked area chart
    ax.stackplot(plot_df.index, 
                *[plot_df[tier] for tier in ordered_tiers],
                labels=ordered_tiers,
                colors=[TIER_COLORS.get(_norm_tier_key(tier), TIER_COLORS['OTHER']) 
                       for tier in ordered_tiers])
    
    # CRITICAL: Add labels ALWAYS ON inside segments
    # No conditional - labels are ALWAYS added per spec
    add_a3_labels(ax, plot_df, ordered_tiers, config)
    
    # Format axes
    ax.set_xlabel('')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, color=STYLE_COLORS['gridlines'], alpha=0.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(STYLE_COLORS['axes'])
    ax.spines['bottom'].set_color(STYLE_COLORS['axes'])
    
    # Set tick counts
    set_nice_ticks(ax, config.render.tick_steps_min, config.render.tick_steps_max)
    fig.tight_layout()
    _legend_bottom(fig, ax, config)

    # Cache figure and get PNG bytes
    png = _save_figure_to_cache(fig, cache_key, config.render.figure_dpi)
    
    # Explicit token for notebook metric counting
    logger.info(f"Generated visualization: A3 for {metric_base}")
    
    return fig, cache_key, png


def dual_axis_A5(df: pd.DataFrame,
                metric1: str,
                metric2: str,
                period_col: str,
                config: SynthesisConfig) -> Tuple[Figure, str, bytes]:
    """
    Create A5 dual-axis chart.
    CRITICAL: Only when units differ (e.g., _bal vs _vol).
    
    Args:
        df: DataFrame with paired metrics
        metric1: First metric (left axis)
        metric2: Second metric (right axis)
        period_col: Period column name
        config: Synthesis configuration
    
    Returns:
        Tuple of (Figure, cache_key, png_bytes)
    """
    # Verify units differ
    units_differ = check_units_differ(metric1, metric2)
    if not units_differ:
        # FAIL FAST: No silent fallback to A2
        raise ValueError(f"A5 dual-axis requires differing units. Got: {metric1}, {metric2}. "
                        f"Consider pairing _bal with _vol or a rate metric.")
    
    # Generate cache key
    cache_params = {
        'chart_type': 'A5',
        'metric1': metric1,
        'metric2': metric2
    }
    cache_key = get_cache_key(df[[period_col, metric1, metric2]], cache_params)
    
    # Check cache
    if cache_key in FIGURE_CACHE:
        logger.info(f"Using cached A5 figure for {metric1}/{metric2}")
        return None, cache_key, get_cached_figure(cache_key)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=config.render.figure_size, dpi=config.render.figure_dpi)
    
    # Prepare data with Period conversion
    plot_df = df[[period_col, metric1, metric2]].copy()
    if pd.api.types.is_period_dtype(plot_df[period_col]):
        plot_df[period_col] = plot_df[period_col].astype(str)
    
    # Plot first metric on left axis
    color1 = STYLE_COLORS['TOTAL']
    ax1.set_xlabel('')
    ax1.set_ylabel(pretty_label(metric1), color=color1)
    line1 = ax1.plot(plot_df[period_col], plot_df[metric1], color=color1, 
                     linewidth=2, marker='o', label=metric1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis
    ax2 = ax1.twinx()
    color2 = STYLE_COLORS['highlight']
    ax2.set_ylabel(pretty_label(metric2), color=color2)
    line2 = ax2.plot(plot_df[period_col], plot_df[metric2], color=color2,
                     linewidth=2, marker='s', label=metric2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add growth annotation if space permits
    if len(plot_df) > 1:
        growth1 = (plot_df[metric1].iloc[-1] / plot_df[metric1].iloc[0] - 1) * 100
        growth2 = (plot_df[metric2].iloc[-1] / plot_df[metric2].iloc[0] - 1) * 100
        
        # Only add if under annotation limit
        annotation_count = 0
        if annotation_count < config.render.max_annotations_per_chart:
            ax1.text(0.02, 0.98, f"{metric1}: {growth1:+.1f}%",
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', color=color1)
            annotation_count += 1
        
        if annotation_count < config.render.max_annotations_per_chart:
            ax1.text(0.02, 0.93, f"{metric2}: {growth2:+.1f}%",
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', color=color2)
    
    # Grid and spines
    ax1.grid(True, color=STYLE_COLORS['gridlines'], alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color(STYLE_COLORS['axes'])
    ax1.spines['bottom'].set_color(STYLE_COLORS['axes'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(STYLE_COLORS['axes'])
    
    # Check if we should add utilization overlay (for revolving products)
    has_utilization = False
    if ('tot_open_to_buy' in [metric1, metric2] or 'tot_cr_lines' in [metric1, metric2]) and \
       'tot_open_to_buy' in df.columns and 'tot_cr_lines' in df.columns:
        # Check if LOB is revolving (if available)
        lob = df.attrs.get('lob', '').lower()
        if 'revolving' in lob or lob in ['bankcard', 'credit_card', 'heloc']:
            # Calculate and add utilization overlay
            mask = df['tot_cr_lines'] > 0
            if mask.any():
                utilization = 1 - (df.loc[mask, 'tot_open_to_buy'] / df.loc[mask, 'tot_cr_lines'])
                
                # Add as third axis (since we already have ax2)
                ax3 = ax1.twinx()
                # Offset the third axis to avoid overlap
                ax3.spines['right'].set_position(('outward', 60))
                ax3.plot(df.loc[mask, period_col], utilization * 100,
                        color=STYLE_COLORS['annotation'], linestyle='--',
                        alpha=0.7, label='Utilization %')
                ax3.set_ylabel('Utilization (%)', color=STYLE_COLORS['annotation'])
                ax3.tick_params(axis='y', labelcolor=STYLE_COLORS['annotation'])
                has_utilization = True
                logger.info("Added utilization overlay to A5 chart")
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    # Set tick counts
    set_nice_ticks(ax1, config.render.tick_steps_min, config.render.tick_steps_max)
    set_nice_ticks(ax2, config.render.tick_steps_min, config.render.tick_steps_max)

    fig.tight_layout()
    _legend_bottom(fig, ax1, config, handles=lines, labels=labels)

    # Cache figure and get PNG bytes
    png = _save_figure_to_cache(fig, cache_key, config.render.figure_dpi)
    
    # Explicit token for notebook metric counting
    logger.info(f"Generated visualization: A5 for {metric1}/{metric2}")
    
    # Return with PNG and utilization flag for footer selection
    return fig, cache_key, png


def deltas_bar_A4(df: pd.DataFrame,
                 cluster_name: str,
                 period_col: str,
                 config: SynthesisConfig,
                 delta_type: str = 'yoy',
                 family_resolution: Optional[Dict] = None) -> Tuple[Figure, str, bytes]:
    """
    Create A4 delta bar chart using EXACT DELTA_CLUSTERS.
    CRITICAL: Use bps for <1pp changes.
    Enforces short-delta discipline based on family resolution.
    
    Args:
        df: DataFrame with delta columns
        cluster_name: Name from DELTA_CLUSTERS
        period_col: Period column name
        config: Synthesis configuration
        delta_type: Type of delta (yoy, qoq, mom)
        family_resolution: Family resolution dict with short_delta and use_yoy
    
    Returns:
        Tuple of (Figure, cache_key, png_bytes)
    """
    # Get metrics for cluster or auto-discover
    if cluster_name in DELTA_CLUSTERS:
        metrics = DELTA_CLUSTERS[cluster_name]
    else:
        # Auto-discover delta columns when cluster_name is not predefined
        suffix = f"_{delta_type}_pct"
        delta_cols_found = [c for c in df.columns if c.endswith(suffix)]
        if not delta_cols_found:
            raise ValueError(f"A4: no delta cols ending with {suffix} found.")
        # Extract base metric names
        metrics = [c.replace(suffix, '') for c in delta_cols_found]
        logger.info(f"Auto-discovered {len(metrics)} delta metrics for A4 chart")
    
    # Apply short-delta discipline if family resolution provided
    if family_resolution:
        primary_delta = family_resolution.get('short_delta', delta_type)
        use_yoy = family_resolution.get('use_yoy', False)
        
        # Use primary delta
        delta_cols = [f"{m}_{primary_delta}_pct" for m in metrics]
        
        # Optionally add YoY if different and requested
        if use_yoy and primary_delta != 'yoy':
            yoy_cols = [f"{m}_yoy_pct" for m in metrics]
            # Check if YoY columns exist
            yoy_available = any(c in df.columns for c in yoy_cols)
            if yoy_available:
                logger.info(f"Including YoY deltas alongside {primary_delta} per family resolution")
                # Will create grouped bars for both delta types
                delta_cols = delta_cols + yoy_cols
    else:
        # Default behavior without family resolution
        delta_cols = [f"{m}_{delta_type}_pct" for m in metrics]
    
    # Filter to available columns
    available_cols = [c for c in delta_cols if c in df.columns]
    if not available_cols:
        raise ValueError(f"No delta columns found for cluster {cluster_name}")
    
    # Generate cache key
    cache_params = {
        'chart_type': 'A4',
        'cluster': cluster_name,
        'delta_type': delta_type,
        'metrics': available_cols
    }
    cache_key = get_cache_key(df[available_cols + [period_col]], cache_params)
    
    # Check cache
    if cache_key in FIGURE_CACHE:
        logger.info(f"Using cached A4 figure for {cluster_name}")
        return None, cache_key, get_cached_figure(cache_key)
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.render.figure_size, dpi=config.render.figure_dpi)
    
    # Prepare data with Period conversion
    plot_df = df.copy()
    if pd.api.types.is_period_dtype(plot_df[period_col]):
        plot_df[period_col] = plot_df[period_col].astype(str)
    
    # Get latest period data
    latest_idx = plot_df[period_col].idxmax()
    latest_data = plot_df.loc[latest_idx]
    
    # Prepare bar data
    bar_data = []
    labels = []
    colors = []
    
    for col in available_cols:
        value = latest_data[col]
        if pd.notna(value):
            bar_data.append(value)
            # Extract metric name
            metric_name = col.replace(f'_{delta_type}_pct', '')
            labels.append(pretty_label(metric_name))
            
            # Try to detect if metric name corresponds to a tier
            tier_key = _norm_tier_key(metric_name)
            if tier_key in TIER_COLORS:
                colors.append(TIER_COLORS[tier_key])
            else:
                # Color based on positive/negative
                colors.append(TIER_COLORS['PRIME'] if value >= 0 else TIER_COLORS['SUBPRIME'])
    
    # Create bars
    x_pos = np.arange(len(bar_data))
    bars = ax.bar(x_pos, bar_data, color=colors)
    
    # Add value labels using bps for <1pp
    for i, (bar, value) in enumerate(zip(bars, bar_data)):
        label = fmt_delta(value)
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                label, ha='center', va='bottom' if height >= 0 else 'top')
    
    # Format axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(f'{delta_type.upper()} Change')
    ax.axhline(y=0, color=STYLE_COLORS['axes'], linewidth=0.5)
    _modernize_axes(ax, cluster_name, bar_data)
    
    # Set tick counts
    set_nice_ticks(ax, config.render.tick_steps_min, config.render.tick_steps_max)
    
    plt.tight_layout()
    
    # Cache figure and get PNG bytes
    png = _save_figure_to_cache(fig, cache_key, config.render.figure_dpi)
    
    # Explicit token for notebook metric counting
    logger.info(f"Generated visualization: A4 for {cluster_name}")
    
    return fig, cache_key, png


def snapshot_by_tier_A4(df: pd.DataFrame,
                       metric: str,
                       period_col: str,
                       config: SynthesisConfig) -> Tuple[Figure, str, bytes]:
    """
    Create A4 snapshot chart showing latest values by tier.
    Used when historical data is limited or for latest-only views.
    
    Args:
        df: DataFrame with tier data
        metric: Metric to display
        period_col: Period column name
        config: Synthesis configuration
    
    Returns:
        Tuple of (Figure, cache_key, uses_snapshot_footer)
    """
    if not HAS_MATPLOTLIB or not HAS_DATA_LIBS:
        logger.error("Required libraries not available")
        return None, "libraries_unavailable", b""
        
    # Get latest period
    latest_period = df[period_col].max()
    latest_df = df[df[period_col] == latest_period].copy()
    
    # Extract tier columns
    tier_cols = extract_tier_columns(latest_df)
    allowed = set(getattr(config.logic, 'allowed_tiers', []))
    if allowed:
        tier_cols = [c for c in tier_cols if _norm_tier_key(c) in allowed]
    if not tier_cols:
        logger.warning(f"No tier columns found for snapshot")
        return None, "no_tier_columns", b""
    
    # Generate cache key
    cache_params = {
        'chart_type': 'A4_snapshot',
        'metric': metric,
        'period': str(latest_period)
    }
    cache_key = get_cache_key(latest_df[tier_cols], cache_params)
    
    # Check cache
    if cache_key in FIGURE_CACHE:
        logger.info(f"Using cached A4 snapshot for {metric}")
        return None, cache_key, get_cached_figure(cache_key)
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.render.figure_size, 
                          dpi=config.render.figure_dpi)
    
    # Prepare data - only include tiers with values
    tier_values = []
    tier_names = []
    suppressed_tiers = []

    order = [t for t in TIER_ORDER if (not allowed or t in allowed)]
    for tier in order:
        if tier in tier_cols and tier in latest_df.columns:
            value = latest_df[tier].iloc[0] if len(latest_df) > 0 else None
            if pd.notna(value):
                tier_values.append(value)
                tier_names.append(tier)
            else:
                suppressed_tiers.append(tier)
    
    if not tier_values:
        logger.warning(f"No tier values available for {metric} in {latest_period}")
        return None, f"no_tier_values_{metric}", b""
    
    # Create bar chart
    x_pos = np.arange(len(tier_names))
    colors = [STYLE_COLORS.get(tier, STYLE_COLORS['OTHER']) for tier in tier_names]
    
    bars = ax.bar(x_pos, tier_values, color=colors)
    
    # Add value labels on bars
    for bar, value in zip(bars, tier_values):
        height = bar.get_height()
        label = fmt_currency(value) if 'bal' in metric or 'lines' in metric else fmt_percent(value)
        ax.text(bar.get_x() + bar.get_width()/2, height,
                label, ha='center', va='bottom')
    
    # Format axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tier_names, rotation=45, ha='right')
    ax.set_ylabel(pretty_label(metric))
    ax.set_title(f"{pretty_label(metric)} - {latest_period}")
    ax.grid(True, axis='y', color=STYLE_COLORS['gridlines'], alpha=0.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(STYLE_COLORS['axes'])
    ax.spines['bottom'].set_color(STYLE_COLORS['axes'])
    
    # Set tick counts
    set_nice_ticks(ax, config.render.tick_steps_min, config.render.tick_steps_max)
    
    plt.tight_layout()
    
    # Cache figure and get PNG bytes
    png = _save_figure_to_cache(fig, cache_key, config.render.figure_dpi)
    
    # Return with flag indicating if snapshot footer is needed
    uses_snapshot_footer = len(suppressed_tiers) > 0
    if uses_snapshot_footer:
        logger.info(f"Snapshot chart suppressed tiers: {suppressed_tiers}")
    
    return fig, cache_key, png


def counts_deltas_A4(
    df: pd.DataFrame,
    metric: str,
    period_col: str,
    config: SynthesisConfig,
    delta_type: str = 'yoy',
) -> Tuple[Figure, str, bytes]:
    """Create A4 chart pairing level trend with deltas.

    Args:
        df: DataFrame with metric data
        metric: Metric name
        period_col: Period column name
        config: Synthesis configuration
        delta_type: Type of delta

    Returns:
        Tuple of (Figure, cache_key, png_bytes)
    """
    # Determine appropriate delta column
    delta_col, ylabel = select_delta_col(metric, delta_type, df)
    
    # Generate cache key
    cache_params = {
        'chart_type': 'A4_pair',
        'metric': metric,
        'delta_type': delta_type,
    }
    cache_key = get_cache_key(df[[period_col, metric]], cache_params)

    # Check cache
    if cache_key in FIGURE_CACHE:
        logger.info(f"Using cached A4 pair figure for {metric}")
        return None, cache_key, get_cached_figure(cache_key)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.render.figure_size[0]*1.5, 
                                                  config.render.figure_size[1]),
                                   dpi=config.render.figure_dpi)
    
    # Prepare data with Period conversion
    plot_df = df.copy()
    if pd.api.types.is_period_dtype(plot_df[period_col]):
        plot_df[period_col] = plot_df[period_col].astype(str)
    
    # Left: Level trend
    ax1.bar(plot_df[period_col], plot_df[metric], color=STYLE_COLORS['TOTAL'])
    ax1.set_title('Level')
    ax1.set_xlabel('')
    ax1.set_ylabel(pretty_label(metric))
    ax1.grid(True, axis='y', color=STYLE_COLORS['gridlines'], alpha=0.3)
    
    # Right: Delta - compute if not available
    if delta_col not in plot_df.columns:
        plot_df = compute_missing_deltas(plot_df, metric, delta_type,
                                         getattr(config.logic, 'granularity_family', 'quarterly'))

    if delta_col in plot_df.columns and plot_df[delta_col].notna().any():
        delta_series = pd.to_numeric(plot_df[delta_col], errors='coerce')
        colors = [STYLE_COLORS['PRIME'] if v >= 0 else STYLE_COLORS['SUBPRIME'] for v in delta_series]
        max_abs = delta_series.abs().max()
        unit_label = ylabel
        plot_vals = delta_series
        if is_rate_metric(metric) and pd.notna(max_abs) and max_abs < 1.0:
            plot_vals = delta_series * 100.0
            unit_label = f"{delta_type.upper()} Change (bps)"
        ax2.bar(plot_df[period_col], plot_vals, color=colors)
        ax2.set_title(f'{delta_type.upper()} Change')
        ax2.set_xlabel('')
        ax2.set_ylabel(unit_label)
        ax2.axhline(y=0, color=STYLE_COLORS['axes'], linewidth=0.5)
        ax2.grid(True, axis='y', color=STYLE_COLORS['gridlines'], alpha=0.3)
    else:
        # No delta could be computed - show error message with reason
        ax2.text(0.5, 0.5, f"Insufficient data for {delta_type.upper()} delta\n(need {4 if delta_type == 'yoy' else 2} periods)", 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, color=STYLE_COLORS['axes'])
        ax2.set_axis_off()
    
    # Format spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(STYLE_COLORS['axes'])
    ax1.spines['bottom'].set_color(STYLE_COLORS['axes'])
    set_nice_ticks(ax1, config.render.tick_steps_min, config.render.tick_steps_max)
    
    # Only format ax2 if it has data
    if delta_col in plot_df.columns and plot_df[delta_col].notna().any():
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(STYLE_COLORS['axes'])
        ax2.spines['bottom'].set_color(STYLE_COLORS['axes'])
        set_nice_ticks(ax2, config.render.tick_steps_min, config.render.tick_steps_max)
    
    plt.tight_layout()
    
    # Cache figure and get PNG bytes
    png = _save_figure_to_cache(fig, cache_key, config.render.figure_dpi)

    return fig, cache_key, png


def delta_pair_A4(df: pd.DataFrame,
                  metric: str,
                  period_col: str,
                  config: SynthesisConfig,
                  delta_type: str = 'qoq') -> Tuple[Figure, str, bytes]:
    """Create A4 chart with absolute and percentage deltas."""
    if not HAS_MATPLOTLIB or not HAS_DATA_LIBS:
        logger.error("Required libraries not available")
        return None, "libraries_unavailable", b""

    agg_df = _aggregate_metric(df, period_col, metric)
    agg_df = compute_missing_deltas(agg_df, metric, delta_type,
                                    getattr(config.logic, 'granularity_family', 'quarterly'))

    delta_col, ylabel = select_delta_col(metric, delta_type, agg_df)
    abs_col = f"{metric}_{delta_type}_abs"
    if is_rate_metric(metric):
        scale = infer_percent_scale(agg_df[metric])
        agg_df[abs_col] = agg_df[metric].diff() * (100.0 if scale == 'decimal' else 1.0)
    else:
        agg_df[abs_col] = agg_df[metric].diff()

    plot_df = agg_df[[period_col, abs_col, delta_col]].dropna(how='all', subset=[abs_col, delta_col])
    if pd.api.types.is_period_dtype(plot_df[period_col]):
        plot_df[period_col] = plot_df[period_col].astype(str)
    if plot_df.empty:
        return None, f"no_delta_{metric}", b""

    cache_params = {
        'chart_type': 'A4_delta_pair',
        'metric': metric,
        'delta_type': delta_type
    }
    cache_key = get_cache_key(plot_df, cache_params)
    if cache_key in FIGURE_CACHE:
        logger.info(f"Using cached A4 delta pair for {metric}")
        return None, cache_key, get_cached_figure(cache_key)

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(config.render.figure_size[0], config.render.figure_size[1]),
                                   dpi=config.render.figure_dpi)

    abs_vals = plot_df[abs_col].values
    colors = [STYLE_COLORS['PRIME'] if v >= 0 else STYLE_COLORS['SUBPRIME'] for v in abs_vals]
    ax1.bar(plot_df[period_col], abs_vals, color=colors)
    ax1.axhline(y=0, color=STYLE_COLORS['axes'], linewidth=0.5)
    ax1.set_xlabel('')
    abs_ylabel = f"{delta_type.upper()} Change"
    if is_rate_metric(metric):
        unit = 'pp'
        if np.nanmax(np.abs(abs_vals)) < 1:
            abs_vals = abs_vals * 100
            unit = 'bps'
            ax1.clear()
            ax1.bar(plot_df[period_col], abs_vals, color=colors)
            ax1.axhline(y=0, color=STYLE_COLORS['axes'], linewidth=0.5)
        ax1.set_ylabel(f"{delta_type.upper()} Change ({unit})")
    else:
        ax1.set_ylabel(abs_ylabel)
    set_nice_ticks(ax1, config.render.tick_steps_min, config.render.tick_steps_max)
    ax1.grid(True, axis='y', linewidth=0.6, alpha=0.35)
    for spine in ('top', 'right'):
        ax1.spines[spine].set_visible(False)
    ax1.spines['left'].set_color(STYLE_COLORS.get('axes', '#333333'))
    ax1.spines['bottom'].set_color(STYLE_COLORS.get('axes', '#333333'))

    delta_vals = pd.to_numeric(plot_df[delta_col], errors='coerce')
    unit = 'pp' if is_rate_metric(metric) else '%'
    if is_rate_metric(metric) and np.nanmax(np.abs(delta_vals)) < 1:
        delta_vals = delta_vals * 100
        unit = 'bps'
    colors2 = [STYLE_COLORS['PRIME'] if v >= 0 else STYLE_COLORS['SUBPRIME'] for v in delta_vals]
    ax2.bar(plot_df[period_col], delta_vals, color=colors2)
    ax2.axhline(y=0, color=STYLE_COLORS['axes'], linewidth=0.5)
    ax2.set_xlabel('')
    ax2.set_ylabel(f"{delta_type.upper()} Change ({unit})")
    from matplotlib.ticker import FuncFormatter
    if unit == 'bps':
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}"))
    elif unit == 'pp':
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}"))
    else:
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}%"))
    set_nice_ticks(ax2, config.render.tick_steps_min, config.render.tick_steps_max)
    ax2.grid(True, axis='y', linewidth=0.6, alpha=0.35)
    for spine in ('top', 'right'):
        ax2.spines[spine].set_visible(False)
    ax2.spines['left'].set_color(STYLE_COLORS.get('axes', '#333333'))
    ax2.spines['bottom'].set_color(STYLE_COLORS.get('axes', '#333333'))

    plt.tight_layout()
    png = _save_figure_to_cache(fig, cache_key, config.render.figure_dpi)
    logger.info(f"Generated visualization: A4 delta pair for {metric}")
    return fig, cache_key, png


def delta_over_time_A4(df: pd.DataFrame,
                       metric: str,
                       period_col: str,
                       config: SynthesisConfig,
                       delta_type: str = 'yoy') -> Tuple[Figure, str, bytes]:
    """Create A4 chart showing metric delta over time."""
    if not HAS_MATPLOTLIB or not HAS_DATA_LIBS:
        logger.error("Required libraries not available")
        return None, "libraries_unavailable", b""

    col, ylabel = select_delta_col(metric, delta_type, df)
    if col not in df.columns:
        df = compute_missing_deltas(df, metric, delta_type,
                                    getattr(config.logic, 'granularity_family', 'quarterly'))
    plot_df = df[[period_col, col]].dropna(subset=[col]).copy()
    if pd.api.types.is_period_dtype(plot_df[period_col]):
        plot_df[period_col] = plot_df[period_col].astype(str)
    if plot_df.empty:
        return None, f"no_delta_{metric}", b""

    cache_params = {
        'chart_type': 'A4_delta',
        'metric': metric,
        'delta_type': delta_type
    }
    cache_key = get_cache_key(plot_df[[period_col, col]], cache_params)

    if cache_key in FIGURE_CACHE:
        logger.info(f"Using cached A4 delta for {metric}")
        return None, cache_key, get_cached_figure(cache_key)

    fig, ax = plt.subplots(figsize=config.render.figure_size, dpi=config.render.figure_dpi)

    values = pd.to_numeric(plot_df[col], errors='coerce').values
    unit = 'pp' if is_rate_metric(metric) else '%'
    if is_rate_metric(metric) and np.nanmax(np.abs(values)) < 1:
        values = values * 100
        unit = 'bps'

    colors = [STYLE_COLORS['PRIME'] if v >= 0 else STYLE_COLORS['SUBPRIME'] for v in values]
    ax.bar(plot_df[period_col], values, color=colors)
    ax.axhline(y=0, color=STYLE_COLORS['axes'], linewidth=0.5)
    ax.set_xlabel('')
    ax.set_ylabel(f"{delta_type.upper()} Change ({unit})")

    from matplotlib.ticker import FuncFormatter
    if unit == 'bps':
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}"))
    elif unit == 'pp':
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}%"))

    set_nice_ticks(ax, config.render.tick_steps_min, config.render.tick_steps_max)

    ax.grid(True, axis='y', linewidth=0.3, color=STYLE_COLORS.get('gridlines', '#E5E7EB'), alpha=0.2)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(STYLE_COLORS.get('axes', '#333333'))
    ax.spines['bottom'].set_color(STYLE_COLORS.get('axes', '#333333'))

    plt.tight_layout()
    png = _save_figure_to_cache(fig, cache_key, config.render.figure_dpi)

    logger.info(f"Generated visualization: A4 delta for {metric}")
    return fig, cache_key, png


def delinquency_small_multiples(df: pd.DataFrame,
                               delinq_cols: List[str],
                               period_col: str,
                               config: SynthesisConfig) -> Tuple[Figure, str, bytes]:
    """
    Create small multiples for delinquency metrics with MATCHED Y scales.
    
    Args:
        df: DataFrame with delinquency data
        delinq_cols: List of delinquency column names
        period_col: Period column name
        config: Synthesis configuration
    
    Returns:
        Tuple of (Figure, cache_key, png_bytes)
    """
    # Generate cache key
    cache_params = {
        'chart_type': 'delinquency',
        'metrics': delinq_cols
    }
    cache_key = get_cache_key(df[delinq_cols + [period_col]], cache_params)
    
    # Check cache
    if cache_key in FIGURE_CACHE:
        logger.info(f"Using cached delinquency figure")
        return None, cache_key, get_cached_figure(cache_key)
    
    # Determine grid layout
    n_charts = len(delinq_cols)
    n_cols = min(3, n_charts)
    n_rows = (n_charts + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(config.render.figure_size[0]*n_cols/2,
                                   config.render.figure_size[1]*n_rows/2),
                            dpi=config.render.figure_dpi)
    
    if n_charts == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Prepare data with Period conversion
    plot_df = df[[period_col] + delinq_cols].copy()
    if pd.api.types.is_period_dtype(plot_df[period_col]):
        plot_df[period_col] = plot_df[period_col].astype(str)
    
    # Find global Y scale for matching
    global_min = plot_df[delinq_cols].min().min()
    global_max = plot_df[delinq_cols].max().max()
    
    # Plot each delinquency metric
    for i, col in enumerate(delinq_cols):
        ax = axes[i]
        ax.plot(plot_df[period_col], plot_df[col], color=STYLE_COLORS['SUBPRIME'],
               linewidth=2, marker='o')
        
        ax.set_title(pretty_label(col), fontsize=10)
        ax.set_xlabel('')
        ax.set_ylim(global_min * 0.95, global_max * 1.05)  # Matched scales
        ax.grid(True, color=STYLE_COLORS['gridlines'], alpha=0.18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(STYLE_COLORS['axes'])
        ax.spines['bottom'].set_color(STYLE_COLORS['axes'])
        
        set_nice_ticks(ax, config.render.tick_steps_min, config.render.tick_steps_max)
    
    # Hide unused subplots
    for i in range(n_charts, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Cache figure and get PNG bytes
    png = _save_figure_to_cache(fig, cache_key, config.render.figure_dpi)
    
    return fig, cache_key, png


# Helper functions

def apply_smart_labels(ax, x_vals, y_vals, config: SynthesisConfig):
    """Apply smart labels to line chart."""
    # Guard against object dtype issues
    y_vals = pd.to_numeric(y_vals, errors="coerce")
    
    if len(x_vals) <= 5:
        # Label all points if few
        for x, y in zip(x_vals, y_vals):
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                       xytext=(0,5), ha='center', fontsize=8)
    else:
        # Label first, last, min, max
        indices = [0, len(y_vals)-1, np.argmin(y_vals), np.argmax(y_vals)]
        indices = list(set(indices))  # Remove duplicates
        
        for i in indices[:config.render.max_annotations_per_chart]:
            ax.annotate(f'{y_vals.iloc[i]:.1f}', 
                       (x_vals.iloc[i], y_vals.iloc[i]),
                       textcoords="offset points",
                       xytext=(0,5), ha='center', fontsize=8)


def add_a3_labels(ax, df: pd.DataFrame, tiers: List[str], config: SynthesisConfig):
    """
    Add labels INSIDE A3 stacked area segments.
    CRITICAL: Labels must ALWAYS be ON for A3 charts.
    """
    # Get midpoint of x-axis
    x_mid = len(df) // 2
    
    # Calculate cumulative percentages at midpoint
    cumulative = 0
    for tier in tiers:
        value = df.iloc[x_mid][tier]
        if value > config.render.smart_label_threshold * 100:  # Only if segment is large enough
            y_pos = cumulative + value / 2
            ax.text(df.index[x_mid], y_pos, f'{tier}\n{value:.1f}%',
                   ha='center', va='center', fontsize=9, fontweight='bold')
        cumulative += value


def order_legend_items(items: List[str], latest_values: Dict[str, float]) -> List[str]:
    """Order legend items by TIER_ORDER then latest value descending."""
    # First sort by TIER_ORDER
    tier_items = []
    other_items = []
    
    for item in items:
        if item.upper() in TIER_ORDER:
            tier_items.append(item)
        else:
            other_items.append(item)
    
    # Sort tier items by TIER_ORDER
    tier_items.sort(key=lambda x: TIER_ORDER.index(x.upper()) if x.upper() in TIER_ORDER else 999)
    
    # Sort other items by latest value
    other_items.sort(key=lambda x: latest_values.get(x, 0), reverse=True)
    
    return tier_items + other_items


def check_units_differ(metric1: str, metric2: str) -> bool:
    """Check if two metrics have different units."""
    # Simple heuristic: _bal vs _vol, or different base metrics
    if '_bal' in metric1 and '_vol' in metric2:
        return True
    if '_vol' in metric1 and '_bal' in metric2:
        return True
    if 'rate' in metric1.lower() and 'rate' not in metric2.lower():
        return True
    if 'rate' in metric2.lower() and 'rate' not in metric1.lower():
        return True
    
    return False


def set_nice_ticks(ax, min_ticks: int = 5, max_ticks: int = 7):
    """Set nice tick counts within specified range."""
    # Get current ticks
    y_ticks = ax.get_yticks()
    
    # Adjust if needed
    if len(y_ticks) < min_ticks:
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=min_ticks))
    elif len(y_ticks) > max_ticks:
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks))


def spaghetti_cap_and_fade(df: pd.DataFrame,
                          series_cols: List[str],
                          max_series: int = 6,
                          fade_alpha: float = 0.45) -> Tuple[List[str], List[str]]:
    """
    Cap spaghetti chart at max_series and fade others.
    CRITICAL: Cap is 6 (not 5).
    
    Args:
        df: DataFrame with series data
        series_cols: List of series column names
        max_series: Maximum series to show fully (6)
        fade_alpha: Alpha for faded series
    
    Returns:
        Tuple of (primary_series, faded_series)
    """
    if len(series_cols) <= max_series:
        return series_cols, []
    
    # Sort by latest value
    latest_values = df[series_cols].iloc[-1].sort_values(ascending=False)
    
    primary = latest_values.head(max_series).index.tolist()
    faded = latest_values.iloc[max_series:].index.tolist()
    
    logger.info(f"Spaghetti chart: {len(primary)} primary, {len(faded)} faded series")
    
    return primary, faded


def utilization_overlay(ax, df: pd.DataFrame, period_col: str, config: SynthesisConfig):
    """Add utilization overlay if credit lines > 0."""
    if 'tot_cr_lines' in df.columns and 'tot_open_to_buy' in df.columns:
        # Calculate utilization where CL > 0
        mask = df['tot_cr_lines'] > 0
        if mask.any():
            utilization = 1 - (df.loc[mask, 'tot_open_to_buy'] / df.loc[mask, 'tot_cr_lines'])
            
            # Convert Period objects to string for plotting
            plot_periods = df.loc[mask, period_col]
            if pd.api.types.is_period_dtype(plot_periods):
                plot_periods = plot_periods.astype(str)
            
            # Add as secondary axis
            ax2 = ax.twinx()
            ax2.plot(plot_periods, utilization * 100,
                    color=STYLE_COLORS['annotation'], linestyle='--',
                    alpha=0.7, label='Utilization %')
            ax2.set_ylabel('Utilization (%)', color=STYLE_COLORS['annotation'])
            ax2.tick_params(axis='y', labelcolor=STYLE_COLORS['annotation'])


def compute_nice_ticks(data_min: float, data_max: float, 
                       target_count: int = 6) -> List[float]:
    """Compute nice tick values for axis."""
    data_range = data_max - data_min
    
    if data_range == 0:
        return [data_min]
    
    # Find nice step size
    rough_step = data_range / (target_count - 1)
    magnitude = 10 ** np.floor(np.log10(rough_step))
    
    nice_steps = [1, 2, 2.5, 5, 10]
    step = min(nice_steps, key=lambda x: abs(x * magnitude - rough_step))
    step = step * magnitude
    
    # Generate ticks
    start = np.floor(data_min / step) * step
    stop = np.ceil(data_max / step) * step
    ticks = np.arange(start, stop + step/2, step)
    
    return ticks.tolist()


def get_cached_figure(cache_key: str) -> Optional[bytes]:
    """Retrieve cached figure by key."""
    return FIGURE_CACHE.get(cache_key)


def clear_figure_cache():
    """Clear the figure cache."""
    global FIGURE_CACHE
    FIGURE_CACHE = {}
    logger.info("Figure cache cleared")