"""
Configuration for Agent 3 synthesis pipeline.
Contains all constants, flags, colors, and settings.
Supports loading from YAML configuration files.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields, MISSING, Field
from typing import Dict, List, Optional, Tuple, Any

STRICT_CONFIG = False
_UNKNOWN_KEYS = 0

@dataclass
class RuntimeConfig:
    """Runtime configuration with GCP settings."""
    # GCP Configuration (hardcoded per spec)
    project_id: str = "426360366917"
    location: str = "us-central1"
    generative_model_name: str = "gemini-2.5-flash"
    max_output_tokens: int = 6000

    # LLM sampling
    temperature: float = 0.6

    # Runtime settings
    log_level: str = "info"
    cache_enabled: bool = True
    strict_config: bool = False
    random_seed: int = 42
    cache_dir: str = ".cache/figures"
    output_dir: str = "synthesis_output"
    template_dir: str = "templates"
    
    # Execution settings
    max_retries: int = 3
    timeout_seconds: int = 300
    parallel_workers: int = 4


@dataclass
class RenderConfig:
    """Rendering and visualization settings."""
    # Figure settings (exact per spec)
    figure_dpi: int = 220
    figure_size: Tuple[float, float] = (10, 6)
    
    # Formatting settings
    label_sig_figs: int = 2
    percent_decimals: int = 1
    currency_sig_figs: int = 2
    add_thousands_separators: bool = True
    currency_scale: str = "auto"

    # Visual thresholds
    overlay_legibility_threshold: float = 0.35
    spaghetti_fade_alpha: float = 0.45
    spaghetti_fade_others: bool = True
    tick_steps_target: int = 6
    tick_steps_min: int = 5
    tick_steps_max: int = 7

    # Chart limits
    max_spaghetti_series: int = 6  # Cap at 6 (not 5)
    max_annotations_per_chart: int = 2

    # Legend placement
    legend_bottom: bool = True
    legend_bottom_pad: float = 0.22
    legend_position: str = "TOP"
    legend_font_pt: int = 11
    category_font_pt: int = 9
    value_font_pt: int = 9

    # Single-chart layout sizing (inches)
    single_chart_left_in: float = 0.35
    single_chart_right_in: float = 0.35
    single_chart_top_in: float = 0.95
    single_chart_bottom_in: float = 0.6
    single_chart_width_in: float = 9.3
    single_chart_height_in: float = 5.9
    single_chart_min_height_in: float = 3.5
    single_chart_min_width_in: float = 6.0
    single_chart_bottom_margin_in: float = 0.6
    subtitle_gap_in: float = 0.10
    subtitle_height_in: float = 0.45
    chart_gap_in: float = 0.05
    two_chart_gap_in: float = 0.25
    single_chart_fill_height: bool = False

    # Label settings
    a3_labels_always_on: bool = True
    smart_label_threshold: float = 0.05

    # Rendering policies
    zero_cut_policy: str = "auto"
    gridline_color: str = "#E5E7EB"
    axis_color: str = "#424242"
    a3_chart_kind: str = "stacked_area_100"


@dataclass
class LogicConfig:
    """Business logic and validation settings."""
    # Data thresholds
    sparsity_threshold: float = 0.5
    min_data_points: int = 3
    min_periods_for_trend: int = 3
    min_periods_for_snapshot: int = 1
    
    # Per-consumer materiality thresholds (separate fields per spec)
    per_consumer_correlation_threshold: float = 0.92
    per_consumer_difference_threshold: float = 0.15
    
    # Processing flags
    missing_as_na: bool = True
    guard_product_restrictions: bool = True
    require_spec_alignment: bool = True
    allow_sparse_legend: bool = False
    allow_indexing: bool = True
    allow_utilization: bool = True
    allow_shares: bool = True
    
    # Semantic processing flags (new)
    normalize_engagement: bool = True
    normalize_manifest: bool = True
    resolve_missing_paths: bool = True
    compute_missing_deltas: bool = True
    compute_calculated_metrics: bool = True
    
    # Granularity settings
    granularity_family: str = "both"  # auto|quarterly|monthly|both
    use_yoy_default: bool = True
    prefer_granularity_from_engagement: bool = True
    enforce_single_granularity_family: bool = True
    delta_short_family: str = "auto"
    
    # Validation
    enforce_coverage: bool = True
    allow_a2_fallback: bool = True
    legibility_threshold: float = 0.15
    max_callouts: int = 2
    max_annotations: int = 2
    
    # Chart limits
    max_series_per_chart: int = 4  # Cap series to avoid clutter
    title_style: str = "insight-first"
    aggregate_tail_as_others: bool = False
    label_policy: str = "smart"
    include_per_consumer_counterparts: str = "auto"

    # Periodic trend defaults
    default_focus_quarter: str = "Q4"


@dataclass
class BrandConfig:
    """Branding and template configuration."""
    use_brand_template: bool = True
    preferred_template: Optional[str] = None
    fallback_template: Optional[str] = None
    client_name: str = "Client"
    report_title: str = "Credit Portfolio Analysis"
    primary_color: str = "#003B70"
    secondary_color: str = "#00BDF2"
    font_family: str = "Arial"
    logo_path: Optional[str] = None
    objectives: List[str] = field(default_factory=list)
    
    # Slide settings
    include_cover: bool = True
    include_agenda: bool = True
    include_appendix: bool = True
    include_topic_summaries: bool = True


@dataclass
class FeatureFlags:
    """Feature flags for controlling behavior."""
    chart_engine: str = "pptx"
    single_content_only: bool = True
    combo_strategy: str = "two_charts"  # two_charts | template_combo
    tie_strapline_to_objectives: bool = True
    force_single_content: bool = True
    only_periodic_line_trends: bool = False
    show_subtitle: bool = False
    descriptive_titles_only: bool = False

    enable_pdf_export: bool = False
    enable_cross_topic_rollup: bool = False
    enable_speaker_notes: bool = True
    narrative_engine: str = "vertex"  # Options: "vertex" or "deterministic"
    agenda_sections_only: bool = True
    add_dpd_change_table: bool = True
    chart_title_inside: bool = True

    composition_base: str = "auto"
    auto_light_up_new_fields: bool = True
    payments_on_bankcard_privatelabel: bool = False
    auto_add_overview_topics: bool = False
    long_horizon_indexed_slide: bool = False

    verbose_logging: bool = True
    save_intermediate_artifacts: bool = True
@dataclass
class SynthesisConfig:
    """Main configuration container."""
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    brand: BrandConfig = field(default_factory=BrandConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    viz_defaults: Dict[str, Any] = field(default_factory=dict)
    fallback_compare_order: List[str] = field(default_factory=lambda: ["QOQ", "YOY"])

    def __post_init__(self) -> None:
        # Convert any stray dataclasses.Field objects to real instances
        for name, f in ((f.name, f) for f in fields(self)):
            val = getattr(self, name)
            if isinstance(val, Field):
                if f.default_factory is not MISSING:
                    setattr(self, name, f.default_factory())
                else:
                    setattr(self, name, f.default if f.default is not MISSING else None)
        # Normalize figure_size if provided as list via YAML
        if isinstance(self.render.figure_size, list):
            self.render.figure_size = tuple(self.render.figure_size)


# Color Constants (CRITICAL: SUBPRIME must be #E53935)
STYLE_COLORS = {
    "SUPER_PRIME": "#60A5FA",   # Bright sky blue
    "PRIME_PLUS": "#34D399",    # Fresh green
    "PRIME": "#FBBF24",         # Warm amber
    "NEAR_PRIME": "#F59E0B",    # Rich orange
    "SUBPRIME": "#E53935",      # CRITICAL: Must be exactly this value
    "UNSCORED": "#CBD5E1",      # Soft gray-blue
    "OTHER": "#94A3B8",         # Muted slate
    "TOTAL": "#475569",         # Charcoal slate (matches axes per spec)
    
    # Chart colors
    "axes": "#424242",
    "gridlines": "#E5E7EB",
    "annotation": "#FF6B6B",
    "highlight": "#FFD93D",
    "CURRENT_PERIOD": "#FFC000",
    "PRIOR_PERIOD": "#00B0F0",
    "DPD_30": "#00B0F0",
    "DPD_60": "#FFC000",
    "DPD_90": "#FF0000"
}

# Assert SUBPRIME color is correct (will fail if changed)
assert STYLE_COLORS["SUBPRIME"] == "#E53935", "SUBPRIME color must be #E53935"

# Tier ordering (canonical - per spec, with TOTAL at end)
TIER_ORDER = ["SUPER_PRIME", "PRIME_PLUS", "PRIME", "NEAR_PRIME", "SUBPRIME", "UNSCORED", "OTHER", "TOTAL"]

# Column aliases for normalization
COLUMN_ALIASES = {
    # Tier variants
    "SUPER_PRIME": ["SUPER_PRIME", "Super_Prime", "super_prime", "SUPER-PRIME", "Super-Prime"],
    "PRIME_PLUS": ["PRIME_PLUS", "Prime_Plus", "prime_plus", "PRIME-PLUS", "Prime-Plus", "Prime+"],
    "PRIME": ["PRIME", "Prime", "prime"],
    "NEAR_PRIME": ["NEAR_PRIME", "Near_Prime", "near_prime", "NEAR-PRIME", "Near-Prime"],
    "SUBPRIME": ["SUBPRIME", "Subprime", "subprime", "SUB_PRIME", "Sub_Prime"],
    "UNSCORED": ["UNSCORED", "Unscored", "unscored", "NO_SCORE", "No_Score"],
    
    # Metric aliases
    "tot_acct_bal": ["tot_acct_bal", "total_account_balance", "balance"],
    "tot_acct_vol": ["tot_acct_vol", "total_account_volume", "volume"],
    "tot_new_acct_bal": ["tot_new_acct_bal", "new_account_balance", "new_balance"],
    "tot_new_acct": ["tot_new_acct", "new_accounts", "new_acct"],
    "tot_cr_lines": ["tot_cr_lines", "credit_lines", "cr_lines"],
    "tot_open_to_buy": ["tot_open_to_buy", "open_to_buy", "otb"],
    "avg_cr_lines": ["avg_cr_lines", "average_credit_lines", "avg_credit"],
    "avg_open_to_buy": ["avg_open_to_buy", "average_open_to_buy", "avg_otb"],
    "avg_open_to_buy_per_cnsmr": ["avg_open_to_buy_per_cnsmr", "otb_per_consumer", "avg_otb_per_consumer"],
    "tot_cnsmr_cnts": ["tot_cnsmr_cnts", "total_consumers", "total_consumer_count", "consumers_total"],
    "tot_cnsmr_cnts_w_bal": ["tot_cnsmr_cnts_w_bal", "consumers_with_balance", "total_consumers_with_balance"],
    "avg_acct_per_cnsmr": ["avg_acct_per_cnsmr", "avg_accounts_per_consumer", "avg_accts_per_cnsmr"]
}

# Delta clusters (EXACT per spec - 2-3 metrics max)
DELTA_CLUSTERS = {
    "Balances": ["tot_acct_bal", "tot_acct_vol"],  # Only 2
    "Originations": ["tot_new_acct_bal", "tot_new_acct"],  # Only 2
    "Supply totals": ["tot_cr_lines", "tot_open_to_buy"],
    "Supply averages": ["avg_cr_lines", "avg_open_to_buy", "avg_open_to_buy_per_cnsmr"],
    "Average set": ["avg_acct_bal", "avg_tot_bal_per_cnsmr", "avg_acct_per_cnsmr"]
}

# Heatmap control (CRITICAL: must be False)
ALLOW_HEATMAPS = False

# Product restrictions
REVOLVING_ONLY_METRICS = [
    "tot_open_to_buy",
    "avg_open_to_buy",
    "avg_open_to_buy_per_cnsmr",
    "utilization_rate"
]

# Topic to LOB mapping for product gating
TOPIC_TO_LOB = {
    "bankcard": "revolving",
    "credit_card": "revolving",
    "heloc": "revolving",
    "auto": "installment",
    "mortgage": "installment",
    "student": "installment",
    "personal": "installment"
}

# Deck footer constants (EXACT verbatim text)
DECK_FOOTERS = {
    "default": "Source: Client CSV manifest (Agent 2 output). Totals may not equal 100 due to rounding.",
    "utilization": "Utilization = 1 − OTB/CL; not shown where CL = 0 or inputs missing.",
    "sparse": "Some series suppressed due to limited data availability.",
    "missing_base": "YoY/QoQ not shown where a valid base period is unavailable.",
    "snapshot": "Some tiers suppressed due to missing values in the latest period."
}

# Agenda slide caveats (EXACT verbatim text)
AGENDA_CAVEATS = [
    "PoC: local files only",
    "No recompute of deltas",
    "Single granularity & delta family (lean)",
    "Single-content slides",
    "No macroeconomic module (PoC scope)"
]

# Chart type mapping
CHART_TYPES = {
    "trend": "A2",
    "composition": "A3",
    "delta": "A4",
    "dual_axis": "A5",
    "delinquency": "small_multiples"
}

# Narrative limits
NARRATIVE_LIMITS = {
    "title_max_chars": 85,
    "bullet_min": 2,
    "bullet_max": 3,
    "strapline_max_chars": 140,
    "speaker_notes_bullets": 3
}

# Default configuration instance
DEFAULT_CONFIG = SynthesisConfig()


def _apply_section(obj, data, section_name: str) -> None:
    global _UNKNOWN_KEYS
    for k, v in data.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
        else:
            _UNKNOWN_KEYS += 1
            msg = f"Ignoring unknown {section_name} key: {k}"
            if STRICT_CONFIG:
                raise KeyError(msg)
            logging.getLogger("config").warning(msg)


def _validate_config_shape(cfg: SynthesisConfig) -> None:
    assert isinstance(cfg.runtime, RuntimeConfig), "cfg.runtime must be RuntimeConfig"
    assert isinstance(cfg.render, RenderConfig), "cfg.render must be RenderConfig"
    assert isinstance(cfg.logic, LogicConfig), "cfg.logic must be LogicConfig"
    assert isinstance(cfg.brand, BrandConfig), "cfg.brand must be BrandConfig"
    assert isinstance(cfg.features, FeatureFlags), "cfg.features must be FeatureFlags"
    fs = cfg.render.figure_size
    assert isinstance(fs, (tuple, list)) and len(fs) == 2, "figure_size must be (w,h)"
    if isinstance(fs, list):
        cfg.render.figure_size = (float(fs[0]), float(fs[1]))


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    return yaml_data


def apply_yaml_to_config(config: SynthesisConfig, yaml_data: Dict[str, Any]) -> SynthesisConfig:
    """Apply YAML configuration to a SynthesisConfig instance."""
    # Compatibility remapping for legacy keys
    r = yaml_data.get('render', {})
    l = yaml_data.get('logic', {})
    if 'export_pdf' in r:
        config.features.enable_pdf_export = bool(r.pop('export_pdf'))
    if 'tie_strapline_to_objectives' in l:
        config.features.tie_strapline_to_objectives = bool(l.pop('tie_strapline_to_objectives'))
    if 'legibility_threshold' in l and 'overlay_legibility_threshold' not in r:
        config.render.overlay_legibility_threshold = float(l.pop('legibility_threshold'))

    # Apply runtime settings
    if 'runtime' in yaml_data:
        _apply_section(config.runtime, yaml_data['runtime'], 'runtime')
        global STRICT_CONFIG
        STRICT_CONFIG = getattr(config.runtime, 'strict_config', False)

    # Apply render settings
    if 'render' in yaml_data:
        r = yaml_data['render']
        if ('figure_width_inches' in r) or ('figure_height_inches' in r):
            w = float(r.get('figure_width_inches', config.render.figure_size[0]))
            h = float(r.get('figure_height_inches', config.render.figure_size[1]))
            config.render.figure_size = (w, h)
            r.pop('figure_width_inches', None)
            r.pop('figure_height_inches', None)
        _apply_section(config.render, r, 'render')

    # Apply logic settings
    if 'logic' in yaml_data:
        _apply_section(config.logic, yaml_data['logic'], 'logic')

    # Apply brand settings
    if 'brand' in yaml_data:
        _apply_section(config.brand, yaml_data['brand'], 'brand')

    # Apply feature flags
    if 'features' in yaml_data:
        _apply_section(config.features, yaml_data['features'], 'features')

    # Extra sections
    if 'viz_defaults' in yaml_data:
        config.viz_defaults = yaml_data['viz_defaults']
    if 'fallback_compare_order' in yaml_data:
        config.fallback_compare_order = yaml_data['fallback_compare_order']

# Update global constants from YAML if present
    if 'style_colors' in yaml_data:
        global STYLE_COLORS
        STYLE_COLORS.update(yaml_data['style_colors'])
    
    if 'tier_order' in yaml_data:
        global TIER_ORDER
        TIER_ORDER = yaml_data['tier_order']
    
    if 'delta_clusters' in yaml_data:
        global DELTA_CLUSTERS
        DELTA_CLUSTERS = yaml_data['delta_clusters']
    
    if 'column_aliases' in yaml_data:
        global COLUMN_ALIASES
        COLUMN_ALIASES.update(yaml_data['column_aliases'])
    
    if 'topic_to_lob' in yaml_data:
        global TOPIC_TO_LOB
        TOPIC_TO_LOB.update(yaml_data['topic_to_lob'])
    
    if 'narrative_limits' in yaml_data:
        global NARRATIVE_LIMITS
        NARRATIVE_LIMITS.update(yaml_data['narrative_limits'])
    
    if 'footers' in yaml_data:
        global DECK_FOOTERS
        DECK_FOOTERS.update(yaml_data['footers'])
    
    if 'agenda_caveats' in yaml_data:
        global AGENDA_CAVEATS
        AGENDA_CAVEATS = yaml_data['agenda_caveats']
    
    # Handle validation settings
    if 'validation' in yaml_data:
        if 'allow_heatmaps' in yaml_data['validation']:
            global ALLOW_HEATMAPS
            ALLOW_HEATMAPS = yaml_data['validation']['allow_heatmaps']
    
    return config


def load_config(overrides: Optional[Dict] = None, config_path: Optional[str] = None) -> SynthesisConfig:
    """
    Load configuration with optional YAML file and overrides.
    
    Args:
        overrides: Dictionary of configuration overrides
        config_path: Path to YAML configuration file
    
    Returns:
        SynthesisConfig instance with applied settings
    """
    global _UNKNOWN_KEYS, STRICT_CONFIG
    _UNKNOWN_KEYS = 0
    STRICT_CONFIG = False
    config = SynthesisConfig()
    
    # First, try to load default YAML config if it exists
    default_yaml_path = Path("synthesis_agent/config.yaml")
    if default_yaml_path.exists() and not config_path:
        yaml_data = load_yaml_config(str(default_yaml_path))
        config = apply_yaml_to_config(config, yaml_data)
    
    # Then, load specified YAML config if provided
    if config_path:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml_data = load_yaml_config(config_path)
            config = apply_yaml_to_config(config, yaml_data)
        elif config_path.endswith('.json'):
            # Support JSON config for backward compatibility
            with open(config_path, 'r') as f:
                json_data = json.load(f)
            if json_data:
                overrides = json_data
    
    # Finally, apply any direct overrides
    if overrides:
        # Apply overrides to appropriate sections
        for key, value in overrides.items():
            if hasattr(config.runtime, key):
                setattr(config.runtime, key, value)
            elif hasattr(config.render, key):
                setattr(config.render, key, value)
            elif hasattr(config.logic, key):
                setattr(config.logic, key, value)
            elif hasattr(config.brand, key):
                setattr(config.brand, key, value)
            elif hasattr(config.features, key):
                setattr(config.features, key, value)

    _validate_config_shape(config)

    logger = logging.getLogger("config")
    if _UNKNOWN_KEYS:
        logger.warning(f"Ignored {_UNKNOWN_KEYS} unknown config key(s)")
    else:
        logger.info("No unknown config keys detected")

    # Validate critical invariants
    assert STYLE_COLORS["SUBPRIME"] == "#E53935", "SUBPRIME color must be #E53935"
    assert ALLOW_HEATMAPS == False, "ALLOW_HEATMAPS must be False"
    assert config.render.max_spaghetti_series == 6, "Spaghetti cap must be 6"
    assert config.render.figure_dpi == 220, "Figure DPI must be 220"
    assert config.render.a3_labels_always_on == True, "A3 labels must always be ON"

    return config
