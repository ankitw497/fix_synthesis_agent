"""
Data loading and normalization for Agent 3 synthesis pipeline.
Handles CSV loading, period alignment, composition base selection, and display derivations.
CRITICAL: Never computes delta columns (*_yoy, *_qoq, *_mom).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from synthesis_agent.config import (
    COLUMN_ALIASES, TIER_ORDER, TOPIC_TO_LOB, SynthesisConfig
)
from synthesis_agent.utils import setup_logging

logger = setup_logging("io_normalize")


# --- Delta helpers -------------------------------------------------------
EPS = 1e-8


def is_rate_metric(metric: str) -> bool:
    """Return True if metric name implies a rate/percentage series."""
    m = (metric or "").lower()
    return (
        "rate" in m
        or "utilization" in m
        or m.endswith("_pct_rate")
    )


def infer_percent_scale(series: pd.Series) -> str:
    """Infer whether a rate series is stored as decimal (0-1) or percent (0-100)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "unknown"
    q = s.abs().quantile(0.9)
    if q <= 1.5:
        return "decimal"
    if q <= 100 + EPS:
        return "percent"
    return "unknown"


def qa_assert_delta_sanity(df: pd.DataFrame, metric: str, delta_type: str) -> None:
    """Log a warning if percent-change deltas for rate metrics look unreasonable."""
    if not is_rate_metric(metric):
        return
    diag_col = f"{metric}_{delta_type}_pct_diag"
    if diag_col in df.columns:
        try:
            max_val = pd.to_numeric(df[diag_col], errors="coerce").abs().max()
            if pd.notna(max_val) and max_val > 500:
                logger.warning(
                    f"[QA] Suppressing {diag_col} for {metric}: base near zero; use _pp instead."
                )
        except Exception:
            pass


def load_frame(filepath: Union[str, Path], 
               missing_as_na: bool = True,
               parse_dates: bool = False) -> pd.DataFrame:
    """
    Load DataFrame from CSV with proper NA handling.
    
    Args:
        filepath: Path to CSV file
        missing_as_na: If True, treat missing values as NA (per spec)
        parse_dates: Whether to parse date columns
    
    Returns:
        Loaded DataFrame with normalized column names
    """
    logger.info(f"Loading frame from {filepath}")
    
    # Load with missing_as_na handling
    if missing_as_na:
        df = pd.read_csv(filepath, na_values=['', 'NA', 'N/A', 'null', 'NULL', None])
    else:
        df = pd.read_csv(filepath)
    
    # Normalize column names (lowercase, strip spaces)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Parse dates if requested
    if parse_dates and 'period' in df.columns:
        df['period'] = pd.to_datetime(df['period'])
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    if "period" in df.columns:
        df["period"] = df["period"].astype(str).str.upper().str.replace('-Q', 'Q')
    if "score_curr_tier" in df.columns:
        df["score_curr_tier"] = df["score_curr_tier"].astype(str).str.upper()
    return df


def attach_lob_context(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    """
    Attach Line of Business (LOB) context to DataFrame.
    CRITICAL: Required for product gating validation.
    
    Args:
        df: Input DataFrame
        topic: Topic name (e.g., 'bankcard', 'auto')
    
    Returns:
        DataFrame with LOB attached in attrs
    """
    # Try multiple key variants for robustness (spaces vs underscores, hyphens)
    variants = [
        topic.lower(),
        topic.lower().replace(' ', '_'),
        topic.lower().replace('-', '_'),
        topic.replace(' ', '_').lower(),
    ]
    lob = "unknown"
    for k in variants:
        if k in TOPIC_TO_LOB:
            lob = TOPIC_TO_LOB[k]
            break
    
    df.attrs['lob'] = lob
    df.attrs['topic'] = topic
    logger.info(f"Attached LOB context: topic={topic}, lob={lob}")
    return df


def select_composition_base(df: pd.DataFrame, 
                           metric: str,
                           period_col: str = "period") -> Optional[str]:
    """
    Select and log composition base for A3 charts.
    CRITICAL: Must log the selected base.
    
    Args:
        df: DataFrame with composition data
        metric: Metric to analyze
        period_col: Period column name
    
    Returns:
        Selected base period or None
    """
    if period_col not in df.columns:
        logger.warning(f"Period column '{period_col}' not found")
        return None
    
    # Get unique periods sorted
    periods = df[period_col].dropna().unique()
    if len(periods) == 0:
        logger.warning("No valid periods found")
        return None
    
    # Instead of sorting by pd.to_datetime(...), use canonical period ordering
    from synthesis_agent.utils import canonical_sort_periods
    periods = df[period_col].astype(str).tolist()
    sorted_periods = canonical_sort_periods(periods)
    
    # Pick latest canonical period string
    base_period = sorted_periods[-1] if sorted_periods else None
    logger.info(f"Selected composition base for {metric}: {base_period}")
    
    return base_period


def align_periods(df: pd.DataFrame,
                 period_col: str = "period",
                 freq: Optional[str] = None) -> pd.DataFrame:
    """
    Align periods using pandas.Period for continuous time axis.
    
    Args:
        df: Input DataFrame
        period_col: Period column name  
        freq: Frequency ('Q' for quarterly, 'M' for monthly)
    
    Returns:
        DataFrame with aligned periods
    """
    if period_col not in df.columns:
        return df
    
    # Detect frequency if not provided
    if freq is None:
        sample = str(df[period_col].iloc[0])
        if 'Q' in sample:
            freq = 'Q'
        else:
            freq = 'M'
    
    try:
        # Convert to Period objects
        df[period_col] = df[period_col].apply(lambda x: pd.Period(x, freq=freq))
        
        # Sort by period
        df = df.sort_values(period_col)
        
        # Convert back to string for compatibility
        df[period_col] = df[period_col].astype(str)
        
        logger.info(f"Aligned periods with frequency={freq}")
    except Exception as e:
        logger.warning(f"Could not align periods: {e}")
    
    return df


def route_analysis_to_granularity(analysis_type: str) -> str:
    """
    Route analysis type to appropriate granularity.
    MoM → monthly; QoQ/YoY → quarterly
    
    Args:
        analysis_type: Type of analysis (mom, qoq, yoy, etc.)
    
    Returns:
        Granularity ('monthly' or 'quarterly')
    """
    analysis_lower = analysis_type.lower()
    
    if 'mom' in analysis_lower or 'month' in analysis_lower:
        return 'monthly'
    else:
        return 'quarterly'


def compute_display_derivations(df: pd.DataFrame,
                               compute_utilization: bool = True,
                               compute_shares: bool = False,
                               compute_index: bool = False) -> pd.DataFrame:
    """
    Compute display-only derivations.
    CRITICAL: These are display-only and must NOT be persisted.
    CRITICAL: Never compute *_yoy, *_qoq, *_mom columns.
    
    Args:
        df: Input DataFrame
        compute_utilization: Whether to compute utilization rate
        compute_shares: Whether to compute percentage shares
        compute_index: Whether to compute indexed values
    
    Returns:
        DataFrame with display columns (not persisted)
    """
    df_display = df.copy()
    
    # Compute utilization if requested
    if compute_utilization:
        if 'tot_open_to_buy' in df.columns and 'tot_cr_lines' in df.columns:
            # Utilization = 1 - OTB/CL (guard against division by zero)
            mask = df['tot_cr_lines'] > 0
            df_display.loc[mask, 'utilization_rate_display'] = (
                1 - df.loc[mask, 'tot_open_to_buy'] / df.loc[mask, 'tot_cr_lines']
            )
            logger.info("Computed utilization rate for display")
    
    # Compute shares if requested
    if compute_shares:
        # Find tier columns
        tier_cols = extract_tier_columns(df)
        if tier_cols:
            # Compute row sums
            row_sums = df[tier_cols].sum(axis=1)
            mask = row_sums > 0
            
            for col in tier_cols:
                share_col = f"{col}_share_display"
                df_display.loc[mask, share_col] = df.loc[mask, col] / row_sums[mask] * 100
            
            logger.info(f"Computed shares for {len(tier_cols)} tier columns")
    
    # Compute index if requested
    if compute_index:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        period_col = next((c for c in df.columns if 'period' in c.lower()), None)
        
        if period_col and len(df) > 0:
            # Index to first period
            first_period = df[period_col].iloc[0]
            base_row = df[df[period_col] == first_period]
            
            for col in numeric_cols:
                if col != period_col and len(base_row) > 0:
                    base_val = base_row[col].iloc[0]
                    if base_val != 0:
                        index_col = f"{col}_index_display"
                        df_display[index_col] = (df[col] / base_val) * 100
            
            logger.info(f"Computed index values base={first_period}")
    
    return df_display


def extract_tier_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract tier columns from DataFrame.
    Supports underscore/hyphen variants and maps to canonical TIER_ORDER.
    
    Args:
        df: Input DataFrame
    
    Returns:
        List of tier column names found
    """
    tier_cols = []
    
    for tier in TIER_ORDER:
        # Check exact canonical name first
        if tier in df.columns:
            tier_cols.append(tier)
            continue
        
        # Check lowercase canonical
        if tier.lower() in df.columns:
            tier_cols.append(tier.lower())
            continue
        
        # Check aliases
        if tier in COLUMN_ALIASES:
            aliases = COLUMN_ALIASES[tier]
            for alias in aliases:
                # Check exact match
                if alias in df.columns:
                    tier_cols.append(alias)
                    break
                # Check lowercase match
                if alias.lower() in df.columns:
                    tier_cols.append(alias.lower())
                    break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tier_cols = []
    for col in tier_cols:
        if col not in seen:
            seen.add(col)
            unique_tier_cols.append(col)
    
    logger.info(f"Found {len(unique_tier_cols)} tier columns: {unique_tier_cols}")
    return unique_tier_cols


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names using aliases.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with normalized column names
    """
    # Create reverse mapping from aliases to canonical names
    alias_map = {}
    
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_map[alias.lower()] = canonical.lower()
    
    # Rename columns
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in alias_map:
            rename_map[col] = alias_map[col_lower]
    
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Normalized {len(rename_map)} column names")
    
    return df


def detect_rate_format(series: pd.Series) -> str:
    """
    Detect if rate values are in 0..1 or 0..100 format.
    
    Args:
        series: Pandas series with rate values
    
    Returns:
        'decimal' for 0..1, 'percentage' for 0..100, 'unknown' otherwise
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'unknown'
    
    # Check if values are strings with % sign
    if non_null.dtype == object:
        pct_count = non_null.astype(str).str.contains('%').sum()
        if pct_count > len(non_null) / 2:
            return 'percentage_string'
    
    # Check numeric ranges
    numeric_vals = pd.to_numeric(non_null, errors='coerce').dropna()
    if len(numeric_vals) == 0:
        return 'unknown'
    
    max_val = numeric_vals.max()
    min_val = numeric_vals.min()
    
    if max_val <= 1.5 and min_val >= -1.5:
        return 'decimal'
    elif max_val <= 150 and min_val >= -150:
        return 'percentage'
    else:
        return 'unknown'


def normalize_rates(df: pd.DataFrame, rate_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize rate columns to consistent format.
    
    Args:
        df: Input DataFrame
        rate_columns: List of rate column names (auto-detect if None)
    
    Returns:
        DataFrame with normalized rates
    """
    if rate_columns is None:
        # Auto-detect rate columns
        rate_columns = [col for col in df.columns 
                       if any(x in col.lower() for x in ['rate', 'pct', 'percent', 'ratio'])]
    
    for col in rate_columns:
        if col not in df.columns:
            continue
        
        format_type = detect_rate_format(df[col])
        
        if format_type == 'percentage_string':
            # Remove % sign and convert to decimal
            df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100
            logger.info(f"Normalized {col} from percentage strings to decimals")
        elif format_type == 'percentage':
            # Convert from 0..100 to 0..1
            df[col] = df[col] / 100
            logger.info(f"Normalized {col} from percentages to decimals")
        elif format_type == 'decimal':
            logger.info(f"{col} already in decimal format")
    
    return df


def load_manifest(manifest_path: Union[str, Path]) -> Dict:
    """
    Load and normalize manifest.json file.
    
    Args:
        manifest_path: Path to manifest.json
    
    Returns:
        Normalized manifest dictionary
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Normalize manifest entries
    for f in manifest.get('files', []):
        # Normalize paths
        if not f.get('path') and f.get('filepath'):
            f['path'] = f['filepath']
        
        # Lowercase analysis types
        if f.get('analysis_type'):
            f['analysis_type'] = str(f['analysis_type']).lower()
        
        # Lowercase granularity
        if f.get('granularity'):
            f['granularity'] = str(f['granularity']).lower()
        
        # Strip topic names
        if f.get('topic'):
            f['topic'] = str(f['topic']).strip()
    
    logger.info(f"Normalized manifest with {len(manifest.get('files', []))} files")
    return manifest


def validate_data_integrity(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate data integrity and return issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for required columns
    if 'period' not in df.columns:
        issues.append("Missing required 'period' column")
    
    # Check for at least one numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        issues.append("No numeric columns found")
    
    # Check for empty DataFrame
    if len(df) == 0:
        issues.append("DataFrame is empty")
    
    # Check for all-NA columns
    for col in df.columns:
        if df[col].isna().all():
            issues.append(f"Column '{col}' contains only NA values")
    
    # CRITICAL: Check for delta column computation (should not exist)
    delta_patterns = ['_yoy', '_qoq', '_mom']
    for col in df.columns:
        for pattern in delta_patterns:
            if pattern in col and not col.endswith('_pct'):
                # This would indicate computed delta columns (forbidden)
                issues.append(f"CRITICAL: Found computed delta column '{col}' - deltas must not be computed locally")
    
    is_valid = len(issues) == 0
    if not is_valid:
        logger.warning(f"Data validation found {len(issues)} issues")
    
    return is_valid, issues


def normalize_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create canonical 'period' column from month/quarter.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with normalized period column
    """
    if "month" in df.columns:
        # Monthly data
        df["period"] = pd.PeriodIndex(pd.to_datetime(df["month"]), freq="M")
        logger.info("Created period column from month (monthly frequency)")
    elif "quarter" in df.columns:
        # Quarterly data
        df["period"] = pd.PeriodIndex(df["quarter"], freq="Q")
        logger.info("Created period column from quarter (quarterly frequency)")
    elif "period" not in df.columns:
        raise ValueError("No time column (month/quarter/period) found in DataFrame")
    
    # --- NEW: canonicalize any pre-existing period field ---
    if 'period' in df.columns:
        s = df['period'].astype(str).str.strip().str.upper()

        def _canon_one(x: str) -> str:
            # Quarterly like 2024Q4
            if 'Q' in x:
                try:
                    return str(pd.Period(x, freq='Q'))
                except Exception:
                    pass
            # Try date-ish: 2024-10, 2024-10-01, etc. -> monthly YYYY-MM
            try:
                dt = pd.to_datetime(x, errors='raise')
                return str(pd.Period(dt, freq='M'))
            except Exception:
                # Last resort: return as-is
                return x

        df['period'] = s.map(_canon_one)
    
    # Sort by period
    df = df.sort_values("period")
    
    # Deduplicate
    dedupe_cols = ["period"]
    if "lob" in df.columns:
        dedupe_cols.append("lob")
    if "score_curr_tier" in df.columns:
        dedupe_cols.append("score_curr_tier")
    
    original_len = len(df)
    df = df.drop_duplicates(dedupe_cols)
    if len(df) < original_len:
        logger.info(f"Dropped {original_len - len(df)} duplicate rows")
    
    # Convert period to string to avoid Period frequency issues when merging
    df['period'] = df['period'].astype(str)
    
    return df


def normalize_score_tiers(df: pd.DataFrame, allowed_tiers: Optional[List[str]] = None) -> pd.DataFrame:
    """Map raw tier labels to canonical names and filter to allowed tiers."""
    if df is None or len(df) == 0:
        return df

    tier_col = next((c for c in ["score_curr_tier", "score_tier", "tier", "risk_tier"] if c in df.columns), None)
    if not tier_col:
        logger.debug("No tier column found for normalization")
        return df

    CANON = {
        "subprime": "SUBPRIME",
        "near_prime": "NEAR_PRIME", "near-prime": "NEAR_PRIME", "near prime": "NEAR_PRIME",
        "prime": "PRIME",
        "prime_plus": "PRIME_PLUS", "prime-plus": "PRIME_PLUS", "prime plus": "PRIME_PLUS",
        "super_prime": "SUPER_PRIME", "super-prime": "SUPER_PRIME", "super prime": "SUPER_PRIME",
        "uns cored": "UNSCORED", "unscored": "UNSCORED",
        "other": "OTHER", "total": "TOTAL"
    }

    def canon(x):
        if pd.isna(x):
            return x
        k = str(x).strip().lower().replace("/", " ").replace("-", " ").replace("__", "_")
        k = " ".join(k.split())
        return CANON.get(k, str(x).strip().upper())

    df[tier_col] = df[tier_col].map(canon)

    if allowed_tiers:
        keep = set([t.upper() for t in allowed_tiers] + ["TOTAL"])
        df = df[df[tier_col].isin(keep)].copy()

    df[tier_col] = pd.Categorical(df[tier_col], categories=TIER_ORDER, ordered=True)

    logger.info(f"Normalized score tiers: {df[tier_col].value_counts().to_dict()}")
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean inf values and coerce to numeric.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    inf_count = 0
    
    for col in df.columns:
        if df[col].dtype == object:
            # Count inf strings
            inf_mask = df[col].astype(str).str.lower().isin(['inf', '-inf', 'infinity', '-infinity'])
            col_inf_count = inf_mask.sum()
            if col_inf_count > 0:
                inf_count += col_inf_count
                
            # Replace inf strings with None
            df[col] = df[col].replace({
                "inf": None, "Inf": None, "INF": None,
                "-inf": None, "-Inf": None, "-INF": None,
                "infinity": None, "Infinity": None, "INFINITY": None,
                "-infinity": None, "-Infinity": None, "-INFINITY": None
            })
            
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors="ignore")
    
    # Replace numeric inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    if inf_count > 0:
        logger.info(f"Cleaned numeric columns: replaced {inf_count} inf values with NaN")
    
    return df


def find_delta_column(df: pd.DataFrame, metric: str, delta_type: str, as_pct: bool = True) -> Optional[str]:
    """
    Find existing delta column under various names.
    
    Args:
        df: DataFrame to search
        metric: Base metric name
        delta_type: Delta type (qoq, mom, yoy)
        as_pct: Whether looking for percentage columns
    
    Returns:
        Column name if found, else None
    """
    from synthesis_agent.semantics import find_delta_pattern
    
    patterns = find_delta_pattern(metric, delta_type, as_pct)
    
    for pattern in patterns:
        if pattern in df.columns:
            return pattern
        # Try lowercase
        if pattern.lower() in df.columns:
            return pattern.lower()
    
    return None


def compute_missing_deltas(df: pd.DataFrame, metric: str, delta_type: str,
                          granularity: str, min_periods: int = 1) -> pd.DataFrame:
    """Compute delta columns, producing percentage-point (pp) for rate metrics and
    percent change (%) for others."""

    if metric not in df.columns or 'period' not in df.columns:
        return df

    is_rate = is_rate_metric(metric)

    # Skip if target delta already exists
    target_col = f"{metric}_{delta_type}_pp" if is_rate else f"{metric}_{delta_type}_pct"
    if target_col in df.columns:
        return df

    # Determine lag by granularity
    delta_lower = delta_type.lower()
    if delta_lower == 'yoy':
        lag = 4 if granularity == 'quarterly' else 12
    else:
        lag = 1

    s = pd.to_numeric(df[metric], errors='coerce')
    tier_col = 'score_curr_tier' if 'score_curr_tier' in df.columns else None
    base = s.groupby(df[tier_col]).shift(lag) if tier_col else s.shift(lag)
    safe_base = base.where(base.abs() > EPS)

    try:
        if is_rate:
            scale = infer_percent_scale(s)
            pp = (s - base) * (100.0 if scale == 'decimal' else 1.0)
            df[f"{metric}_{delta_type}_pp"] = pp
            pct = ((s - base) / safe_base) * 100.0
            df[f"{metric}_{delta_type}_pct_diag"] = pct
        else:
            pct = ((s - base) / safe_base) * 100.0
            df[f"{metric}_{delta_type}_pct"] = pct

    except Exception as e:
        logger.warning(f"Could not compute delta for {metric}: {e}")

    return df


def compute_calculated_metrics(df: pd.DataFrame, group_specs: List[Dict]) -> pd.DataFrame:
    """
    Compute calculated metrics from formulas.
    
    Args:
        df: Input DataFrame
        group_specs: List of calculated metric specifications
    
    Returns:
        DataFrame with calculated metrics added
    """
    for spec in group_specs:
        name = spec.get("name")
        formula = spec.get("formula")
        
        if not name or not formula:
            continue
        
        try:
            # Create safe evaluation environment
            safe_dict = {"__builtins__": {}}
            
            # Add DataFrame columns to environment
            for col in df.columns:
                if col in formula:
                    safe_dict[col] = df[col]
            
            # Evaluate formula
            df[name] = eval(formula, safe_dict)
            
            logger.info(f"Computed calculated metric: {name}")
            
        except Exception as e:
            logger.warning(f"Could not compute {name} with formula '{formula}': {e}")
    
    return df


def hydrate_topic_groups_with_columns(topic_spec: Dict, topic_df: pd.DataFrame) -> Dict:
    """
    Expand insight groups to concrete metrics based on LOB.
    
    Args:
        topic_spec: Topic specification dictionary
        topic_df: DataFrame with topic data
    
    Returns:
        Updated topic spec with expanded metrics
    """
    from synthesis_agent.semantics import (
        is_revolving_lob, expand_group_to_columns, calculated_specs
    )
    
    # Detect LOB from data
    lob = "unknown"
    if "lob" in topic_df.columns:
        lob_values = topic_df["lob"].dropna().astype(str).str.strip().unique()
        if len(lob_values) > 0:
            lob = lob_values[0]
    
    revolving = is_revolving_lob(lob)
    
    for grp in topic_spec.get("insight_groups", []):
        # Expand to concrete columns
        cols = expand_group_to_columns(grp["name"], for_revolving=revolving)
        grp["metrics"] = sorted(set(cols))
        grp["calculated"] = calculated_specs(grp["name"])
        
        req_count = len(grp["metrics"])
        calc_count = len(grp["calculated"])
        
        logger.info(f"Hydrated metrics: {grp['name']} -> {req_count} req / {calc_count} calc")
    
    return topic_spec


def normalize_engagement(eng: Dict) -> Dict:
    """
    Normalize engagement spec with semantic understanding.
    
    Args:
        eng: Raw engagement dictionary
    
    Returns:
        Normalized engagement dictionary
    """
    from synthesis_agent.semantics import _canonicalize_group, allowed_groups_for_topic
    
    topics = eng.get("topics") or eng.get("specification", {}).get("topics") or []
    norm_topics = []
    
    for t in topics:
        topic_name = t["name"] if isinstance(t, dict) else str(t)
        requested_groups = (t.get("insight_groups") if isinstance(t, dict) else []) or []
        
        # Normalize and validate groups
        igs = []
        allowed = allowed_groups_for_topic(topic_name)
        
        for g in requested_groups:
            gname = _canonicalize_group(g if isinstance(g, str) else g.get("name", ""))
            
            # Validate against whitelist
            if allowed and gname not in allowed:
                logger.warning(f"Group '{gname}' not allowed for topic '{topic_name}', skipping")
                continue
            
            igs.append({"name": gname, "metrics": []})  # Metrics filled after LOB known
        
        norm_topics.append({"name": topic_name, "insight_groups": igs})
        
        group_names = [g['name'] for g in igs]
        logger.info(f"Normalized engagement: topic={topic_name}, groups={group_names}")
    
    # Lowercase analysis types
    ats = eng.get("analysis_types") or eng.get("specification", {}).get("analysis_types") or []
    eng["analysis_types"] = [str(x).lower() for x in ats]
    eng["topics"] = norm_topics
    
    return eng