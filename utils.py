"""
Utility functions for Agent 3 synthesis pipeline.
Handles formatting, rounding, sorting, caching, and logging.
"""

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import pandas as pd
import numpy as np

# Period helpers
import re as _re


def is_quarter_code(s: str) -> bool:
    s = str(s).upper()
    return bool(_re.fullmatch(r"\d{4}Q[1-4]", s)) or bool(
        _re.fullmatch(r"\d{4}-Q[1-4]", s)
    )


def is_month_code(s: str) -> bool:
    return bool(_re.fullmatch(r"\d{4}-\d{2}", str(s)))


def _to_canonical_quarter(s: str) -> str:
    return str(s).upper().replace("-", "")


def prev_quarter(q: str) -> str:
    y, qn = map(int, _to_canonical_quarter(q).split("Q"))
    return f"{y-1}Q4" if qn == 1 else f"{y}Q{qn-1}"


def prev_year_same_quarter(q: str) -> str:
    y, qn = map(int, _to_canonical_quarter(q).split("Q"))
    return f"{y-1}Q{qn}"


def prev_month(m: str) -> str:
    y, mn = map(int, str(m).split("-"))
    if mn > 1:
        return f"{y}-{mn-1:02d}"
    return f"{y-1}-12"

def ensure_triplet(ret):
    """
    Normalize chart function returns to (fig, cache_key, extra).
    Accepts (fig, key) or (fig, key, extra) and returns a 3-tuple.
    'extra' is usually PNG bytes; we default to None.
    """
    if isinstance(ret, tuple):
        if len(ret) == 3:
            return ret
        if len(ret) == 2:
            fig, key = ret
            return fig, key, None
    # very legacy: just a Figure
    return ret, f"fig:{uuid4().hex}", None


def round_half_up(value: float, decimals: int = 0) -> float:
    """Round using ROUND_HALF_UP (banker's rounding)."""
    d = Decimal(str(value))
    exp = Decimal(10) ** -decimals
    return float(d.quantize(exp, rounding=ROUND_HALF_UP))


def fmt_currency(value: float, sig_figs: int = 2, use_thousands_sep: bool = True) -> str:
    """Format currency with K/M/B/T scaling."""
    if pd.isna(value):
        return "N/A"
    
    abs_val = abs(value)
    
    if abs_val >= 1e12:
        scaled = value / 1e12
        suffix = "T"
    elif abs_val >= 1e9:
        scaled = value / 1e9
        suffix = "B"
    elif abs_val >= 1e6:
        scaled = value / 1e6
        suffix = "M"
    elif abs_val >= 1e3:
        scaled = value / 1e3
        suffix = "K"
    else:
        scaled = value
        suffix = ""
    
    rounded = round_half_up(scaled, sig_figs)
    
    if suffix:
        formatted = f"{rounded:.{sig_figs}f}{suffix}"
    else:
        if use_thousands_sep:
            formatted = f"{rounded:,.{sig_figs}f}"
        else:
            formatted = f"{rounded:.{sig_figs}f}"
    
    # Use Unicode minus sign (U+2212)
    formatted = formatted.replace("-", "\u2212")
    return formatted


def fmt_percent(value: float, decimals: int = 1, use_bps: bool = False) -> str:
    """Format percentage values, using bps for <1pp changes."""
    if pd.isna(value):
        return "N/A"
    
    # Check if value is already in percentage form (0-100) or decimal (0-1)
    if abs(value) <= 1.5:  # Likely decimal form
        value = value * 100
    
    if use_bps and abs(value) < 1.0:
        bps_value = round_half_up(value * 100, 0)
        formatted = f"{int(bps_value)}bps"
    else:
        rounded = round_half_up(value, decimals)
        formatted = f"{rounded:.{decimals}f}%"
    
    # Use Unicode minus sign
    formatted = formatted.replace("-", "\u2212")
    return formatted


def fmt_delta(value: float, decimals: int = 1) -> str:
    """Format delta values with bps for <1pp and Unicode minus."""
    if pd.isna(value):
        return "N/A"
    
    # Auto-detect if value needs conversion to percentage
    if abs(value) <= 1.5:
        value = value * 100
    
    # Use bps for small changes
    if abs(value) < 1.0:
        bps_value = round_half_up(value * 100, 0)
        if bps_value > 0:
            return f"+{int(bps_value)}bps"
        else:
            formatted = f"{int(bps_value)}bps"
            return formatted.replace("-", "\u2212")
    else:
        rounded = round_half_up(value, decimals)
        if rounded > 0:
            formatted = f"+{rounded:.{decimals}f}pp"
        else:
            formatted = f"{rounded:.{decimals}f}pp"
        return formatted.replace("-", "\u2212")


def use_en_dash(text: str) -> str:
    """Replace hyphens with en-dashes for ranges."""
    # Replace patterns like "2022-2024" with en-dash
    pattern = r'(\d{4})-(\d{4})'
    return re.sub(pattern, r'\1–\2', text)


def add_nbsp_units(text: str) -> str:
    """Add non-breaking spaces between numbers and units."""
    # Add NBSP before common units
    units = ['K', 'M', 'B', 'T', '%', 'pp', 'bps']
    for unit in units:
        pattern = rf'(\d)({unit})'
        text = re.sub(pattern, r'\1\u00A0\2', text)
    return text


def enforce_consistent_decimals(value: float, decimals: int = 2) -> str:
    """Enforce consistent decimal places with Unicode minus."""
    if pd.isna(value):
        return "N/A"
    
    formatted = f"{value:.{decimals}f}"
    # Use Unicode minus sign
    return formatted.replace("-", "\u2212")


def slugify_client(name: str) -> str:
    """Convert client name to filesystem-safe slug."""
    # Remove special characters and replace spaces with underscores
    slug = re.sub(r'[^\w\s-]', '', name.lower())
    slug = re.sub(r'[-\s]+', '_', slug)
    return slug.strip('_')


def canonical_sort_periods(periods: List[str]) -> List[str]:
    """
    Return the original period strings in chronological order,
    supporting mixed monthly ('YYYY-MM') and quarterly ('YYYYQn')
    without collapsing months into quarters. Output is unique and ordered.
    """
    if not periods:
        return []

    # Normalize to strings and drop Nones
    raw = [str(p) for p in periods if p is not None]

    def sort_key(s: str):
        s_up = s.upper().replace("-", "")
        try:
            if 'Q' in s_up:
                # Quarterly like '2024Q4'
                per = pd.Period(s_up, freq='Q')
            else:
                # Assume monthly like '2024-10'
                per = pd.Period(pd.to_datetime(s), freq='M')
            return per.start_time
        except Exception:
            # Fallback: string key to keep function total
            return s

    # Sort by temporal start, but return the ORIGINAL strings
    pairs = [(s, sort_key(s)) for s in raw]
    pairs.sort(key=lambda x: x[1])

    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for s, _ in pairs:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    return ordered


def get_cache_key(data: Any, params: Optional[Dict] = None) -> str:
    """Generate deterministic cache key for figure artifacts."""
    cache_dict = {
        'data_hash': None,
        'params': params or {}
    }
    
    if isinstance(data, pd.DataFrame):
        # Create a copy to avoid modifying original
        data_copy = data.copy()
        
        # Convert Period and datetime columns to string for serialization
        for col in data_copy.columns:
            if pd.api.types.is_period_dtype(data_copy[col]) or \
               pd.api.types.is_datetime64_any_dtype(data_copy[col]):
                data_copy[col] = data_copy[col].astype(str)
        
        # Hash DataFrame contents
        data_str = data_copy.to_json(orient='split', date_format='iso')
        cache_dict['data_hash'] = hashlib.md5(data_str.encode()).hexdigest()
    elif isinstance(data, (list, dict)):
        data_str = json.dumps(data, sort_keys=True)
        cache_dict['data_hash'] = hashlib.md5(data_str.encode()).hexdigest()
    else:
        data_str = str(data)
        cache_dict['data_hash'] = hashlib.md5(data_str.encode()).hexdigest()
    
    # Create final cache key
    cache_str = json.dumps(cache_dict, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()


def setup_logging(name: str = "synthesis_agent", level: int = logging.INFO) -> logging.Logger:
    """Set up structured logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def detect_period_gaps(periods: List[str], freq: str = None) -> List[tuple]:
    """Detect gaps in period sequence for dotted line rendering."""
    if not periods or len(periods) < 2:
        return []
    
    # Auto-detect frequency if not provided
    if freq is None:
        sample = str(periods[0])
        freq = 'M' if ('-' in sample or ('Q' not in sample and len(sample) >= 7)) else 'Q'
    
    try:
        period_objs = [pd.Period(p, freq=freq) for p in periods]
        period_objs = sorted(period_objs)
        
        gaps = []
        for i in range(1, len(period_objs)):
            expected_next = period_objs[i-1] + 1
            if period_objs[i] != expected_next:
                gaps.append((str(period_objs[i-1]), str(period_objs[i])))
        
        return gaps
    except:
        return []


def format_number_display(value: float, number_type: str = "general") -> str:
    """Format numbers for display with appropriate precision."""
    if pd.isna(value):
        return "N/A"
    
    if number_type == "currency":
        return fmt_currency(value)
    elif number_type == "percent":
        return fmt_percent(value)
    elif number_type == "delta":
        return fmt_delta(value)
    elif number_type == "count":
        if value >= 1e6:
            return fmt_currency(value, sig_figs=1)
        else:
            formatted = f"{int(value):,}"
            return formatted.replace("-", "\u2212")
    else:
        # General number formatting
        if abs(value) >= 1000:
            return fmt_currency(value)
        else:
            return enforce_consistent_decimals(value)


def extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from formatted text."""
    if not text or text == "N/A":
        return None
    
    # Remove Unicode minus and replace with regular minus
    text = text.replace("\u2212", "-")
    
    # Remove units and suffixes
    text = re.sub(r'[KMBTpp%bps\s\u00A0]', '', text)
    
    try:
        return float(text)
    except:
        return None


def coalesce_suffix_columns(df, suffixes=None):
    """
    Collapse suffixed duplicates like 'col__QOQ', 'col__MOM' back into 'col'
    by taking the first non-null across variants, preferring the base if present.
    """
    import re
    
    if suffixes is None:
        suffixes = ['__TRENDS', '__SNAPSHOT', '__QOQ', '__MOM', '__YOY', '__PERIODIC_TREND']

    cols = list(df.columns)
    suffix_pattern = f"({'|'.join(map(re.escape, suffixes))})$"

    # Group suffixed variants by base name
    groups = {}
    for c in cols:
        m = re.search(suffix_pattern, str(c))
        if m:
            base = c[: -len(m.group(1))]
            groups.setdefault(base, []).append(c)

    # For each base, coalesce variants into the base col
    for base, variants in groups.items():
        series = None
        if base in df.columns:
            series = df[base]
            for v in variants:
                series = series.combine_first(df[v])
        else:
            # take first non-null across all variants
            series = df[variants].bfill(axis=1).iloc[:, 0]
        df[base] = series

    # Drop the suffixed variants
    to_drop = [v for vs in groups.values() for v in vs]
    df.drop(columns=to_drop, inplace=True, errors='ignore')
    return df


# Display label mappings for human-readable chart titles
DISPLAY_LABELS = {
    # Delinquency
    'cnsmr_cnts_w_deliq_bal_30_rate': '30 DPD rate (consumers)',
    'cnsmr_cnts_w_deliq_bal_60_rate': '60 DPD rate (consumers)',
    'cnsmr_cnts_w_deliq_bal_90_rate': '90 DPD rate (consumers)',
    'acct_cnts_w_deliq_bal_30_rate': '30 DPD rate (accounts)',
    'acct_cnts_w_deliq_bal_60_rate': '60 DPD rate (accounts)',
    'acct_cnts_w_deliq_bal_90_rate': '90 DPD rate (accounts)',
    # Balances/Volumes
    'tot_acct_bal': 'Total account balance',
    'tot_acct_vol': 'Total accounts',
    'avg_acct_bal': 'Average account balance',
    'avg_acct_per_cnsmr': 'Avg. accounts per consumer',
    'avg_tot_bal_per_cnsmr': 'Avg. balance per consumer',
    # Supply
    'tot_cr_lines': 'Total credit lines',
    'tot_open_to_buy': 'Open-to-buy',
    'avg_cr_lines': 'Average credit lines',
    'avg_open_to_buy': 'Average open-to-buy',
    'avg_open_to_buy_per_cnsmr': 'Avg. OTB per consumer',
    # New accounts
    'tot_new_acct_bal': 'New-account balances',
    'tot_new_acct': 'New accounts',
    # Consumer counts
    'tot_cnsmr_cnts': 'Total consumers',
    'tot_cnsmr_cnts_w_bal': 'Consumers with balance',
    # Utilization
    'utilization_rate': 'Utilization rate',
    'utilization_rate_display': 'Utilization rate'
}


def pretty_label(col: str) -> str:
    """Convert column name to human-readable label."""
    c = col.lower()
    if c in DISPLAY_LABELS:
        return DISPLAY_LABELS[c]
    # Heuristics for unknown columns
    if c.endswith('_rate'):
        base = c[:-5].replace('_', ' ')
        base = base.replace('cnsmr', 'consumer').replace('acct', 'account')
        return base.title() + ' rate'
    if c.endswith('_pct'):
        base = c[:-4].replace('_', ' ')
        return base.title() + ' (%)'
    # Generic cleanup
    cleaned = col.replace('_', ' ')
    cleaned = cleaned.replace('cnsmr', 'consumer').replace('acct', 'account')
    cleaned = cleaned.replace('tot', 'total').replace('avg', 'average')
    return cleaned.title()


def fmt_value(val, metric: str) -> str:
    """Format value based on metric type."""
    if pd.isna(val):
        return "N/A"
    
    m = metric.lower()
    # Currency metrics
    if any(x in m for x in ['bal', 'lines', 'buy', 'credit']):
        return fmt_currency(val)
    # Rate/percentage metrics
    if 'rate' in m or 'pct' in m:
        if abs(val) <= 1.5:  # Decimal form
            return f"{val*100:.1f}%"
        else:  # Already percentage
            return f"{val:.1f}%"
    # Count metrics
    if any(x in m for x in ['count', 'cnts', 'accounts', 'consumers']):
        return f"{val:,.0f}"
    # Default numeric
    return f"{val:,.0f}"


# Export DELTA_CLUSTERS for use in other modules
DELTA_CLUSTERS = {
    "Balances": ["tot_acct_bal", "tot_acct_vol"],  # Only 2
    "Originations": ["tot_new_acct_bal", "tot_new_acct"],  # Only 2
    "Supply totals": ["tot_cr_lines", "tot_open_to_buy"],
    "Supply averages": ["avg_cr_lines", "avg_open_to_buy", "avg_open_to_buy_per_cnsmr"],
    "Average set": ["avg_acct_bal", "avg_tot_bal_per_cnsmr", "avg_acct_per_cnsmr"]
}

# --- Added: tier-aware split detection for narratives ---
TIER_LABELS = {"SUBPRIME","NEAR_PRIME","PRIME","PRIME_PLUS","SUPER_PRIME","UNSCORED","OTHER","TOTAL"}

def infer_split_dimension(chart_meta: dict = None, df=None) -> str:
    """
    Infer the split dimension for a chart to guide tier-aware narratives.
    Returns one of: 'credit_tier', 'score_bin', 'none'.

    Priority:
    1) chart_meta['split_dim'] if present and recognized
    2) Legend/series labels in chart_meta (looks like tier names)
    3) DataFrame column names: credit_tier/score_bin/tier/score_bins
    """
    chart_meta = chart_meta or {}

    split_dim = chart_meta.get("split_dim")
    if split_dim in {"credit_tier", "score_bin"}:
        return split_dim

    # Check series labels from metadata
    labels = set([str(s).upper() for s in chart_meta.get("series_labels", [])])
    if labels & TIER_LABELS and len(labels & TIER_LABELS) >= 2:
        return "credit_tier"

    # Check dataframe columns
    if df is not None:
        try:
            cols = {c.lower() for c in getattr(df, "columns", [])}
            if {"score_bin", "score_bins", "tier", "credit_tier"} & cols:
                if "score_bin" in cols or "score_bins" in cols:
                    return "score_bin"
                return "credit_tier"
        except Exception:
            pass

    return "none"
