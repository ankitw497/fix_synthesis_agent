"""
Semantic mappings for Agent 3 synthesis pipeline.
Maps topics to allowed insight groups, groups to metrics, and provides product classification.
"""

from typing import List, Dict, Any, Optional

def resolve_analysis_modes(engagement: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Determine which analysis modes are allowed for this engagement.

    In addition to the existing trend/comparison detection this helper now
    understands ``PERIODIC_TREND`` requests.  The engagement may also specify
    a ``trend_focus_quarter`` (e.g. ``"Q2"``) which we propagate so callers can
    build a quarterly slice across years.
    """

    spec = engagement.get("specification", {})
    client_tf = spec.get("client", {}).get("time_frame", {})
    gran = client_tf.get("interpretation", {}).get("granularity", "").lower()
    compat = client_tf.get("analysis_compatibility", {})

    requested = [r.upper() for r in spec.get("analysis_types", [])]

    use_trend = ("TRENDS" in requested) and bool(compat.get("TRENDS", False))

    periodic_trend = (
        ("PERIODIC_TREND" in requested and bool(compat.get("PERIODIC_TREND", False)))
        or bool(spec.get("trend_focus_quarter"))
    )
    focus_q = spec.get("trend_focus_quarter")

    comp = None
    for m in (r for r in requested if r in ("QOQ", "YOY", "MOM")):
        if not compat.get(m, False):
            continue
        if m == "MOM" and gran != "monthly":
            continue
        comp = m
        break

    return {
        "use_trend": use_trend,
        "periodic_trend": periodic_trend,
        "focus_quarter": focus_q,
        "comparison": comp,
    }

# Topic to Insight Group Mapping - defines which insight groups are valid for each topic
TOPIC_INSIGHT_GROUPS = {
    "Bankcard": ["Balances", "Credit lines", "Available credit", "Delinquencies", "Originations", "Score distributions"],
    "Private Label": ["Balances", "Credit lines", "Available credit", "Delinquencies", "Originations", "Score distributions"],
    "Auto": ["Balances", "Term Distribution", "Delinquencies", "Originations", "Score distributions", "Payments"],
    "Mortgage": ["Balances", "Term Distribution", "Delinquencies", "Originations", "Score distributions", "Payments"],
    "Home Equity Loan": ["Balances", "Delinquencies", "Originations", "Score distributions", "Payments"],
    "HELOC": ["Balances", "Credit lines", "Available credit lines", "Delinquencies", "Originations", "Score distributions", "Payments"],
    "Unsecured Personal Loan": ["Balances", "Term Distribution", "Delinquencies", "Originations", "Score distributions", "Payments"],
    "Secured Personal Loan": ["Balances", "Delinquencies", "Originations", "Score distributions", "Payments"],
    "All Revolving": ["Balances", "Credit lines", "Available credit lines", "Delinquencies", "Originations", "Score distributions"],
    "All Non-Revolving": ["Balances", "Delinquencies", "Originations", "Score distributions"],
    "Collections Accounts": ["Balances"],
    "All Loans & Lines": ["Balances", "Credit lines", "Score distributions"],
    "Student Loan": ["Balances", "Term Distribution", "Delinquencies", "Originations", "Score distributions", "Payments"],
    "Other Consumer Loan": ["Balances", "Delinquencies", "Originations", "Score distributions", "Payments"],
    "Credit Builder Card": ["Balances", "Credit lines", "Available credit", "Delinquencies", "Originations", "Score distributions"],
    "Manufactured Housing": ["Balances", "Term Distribution", "Delinquencies", "Originations", "Score distributions", "Payments"],
    "Recreational Merchandise": ["Balances", "Term Distribution", "Delinquencies", "Originations", "Score distributions", "Payments"]
}

# Insight Group to Metric Mapping - defines concrete metrics for each insight group
INSIGHT_GROUP_TO_METRICS = {
    "Balances": {
        "description": "Account balance metrics",
        "required_metrics": [
            {"name": "Total Account Volume", "column": "tot_acct_vol"},
            {"name": "Total Account Balance", "column": "tot_acct_bal"},
            {"name": "Average Account Balance", "column": "avg_acct_bal"},
            {"name": "Average Total Balance Per Consumer", "column": "avg_tot_bal_per_cnsmr"}
        ],
        "calculated_metrics": [],
        "unavailable": ["Median Account Balance"]
    },
    "Credit lines": {
        "description": "Credit line metrics for revolving products",
        "required_metrics": [
            {"name": "Total Credit Lines", "column": "tot_cr_lines"},
            {"name": "Average Credit Lines", "column": "avg_cr_lines"},
            {"name": "Average Credit Line Per Consumer", "column": "avg_cr_line_per_cnsmr"}
        ],
        "calculated_metrics": [],
        "unavailable": [],
        "product_restriction": "revolving_only"
    },
    "Available credit": {
        "description": "Available credit (open-to-buy) metrics",
        "required_metrics": [
            {"name": "Total Open-to-Buy", "column": "tot_open_to_buy"},
            {"name": "Average Open-to-Buy", "column": "avg_open_to_buy"},
            {"name": "Total Origination Open-to-Buy", "column": "tot_new_acct_open_to_buy"},
            {"name": "Average Origination Open-to-Buy", "column": "avg_new_acct_open_to_buy"},
            {"name": "Average Open-to-Buy Per Consumer", "column": "avg_open_to_buy_per_cnsmr"}
        ],
        "calculated_metrics": [],
        "unavailable": [],
        "product_restriction": "revolving_only"
    },
    "Available credit lines": {  # HELOC specific
        "description": "Available credit lines with utilization calculations",
        "required_metrics": [
            {"name": "Total Open-to-Buy", "column": "tot_open_to_buy"},
            {"name": "Average Open-to-Buy", "column": "avg_open_to_buy"},
            {"name": "Total Origination Open-to-Buy", "column": "tot_new_acct_open_to_buy"},
            {"name": "Average Origination Open-to-Buy", "column": "avg_new_acct_open_to_buy"},
            {"name": "Average Open-to-Buy Per Consumer", "column": "avg_open_to_buy_per_cnsmr"},
            {"name": "Total Credit Lines", "column": "tot_cr_lines"},  # For utilization calc
            {"name": "Total New Account Credit Lines", "column": "tot_new_acct_cr_lines"}  # For origination utilization
        ],
        "calculated_metrics": [
            {"name": "utilization", "formula": "1 - (tot_open_to_buy / tot_cr_lines)"},
            {"name": "origination_utilization", "formula": "1 - (tot_new_acct_open_to_buy / tot_new_acct_cr_lines)"}
        ],
        "unavailable": ["Total End of Draw Account Volume"],
        "product_restriction": "revolving_only"
    },
    "Delinquencies": {
        "description": "Delinquency rate metrics",
        "required_metrics": [
            # Account delinquency rates
            {"name": "30-Day Account Delinquency Rate", "column": "deliq_30_acct_rate"},
            {"name": "60-Day Account Delinquency Rate", "column": "deliq_60_acct_rate"},
            {"name": "90-Day Account Delinquency Rate", "column": "deliq_90_acct_rate"},
            # Balance delinquency rates
            {"name": "30-Day Balance Delinquency Rate", "column": "deliq_30_acct_bal_rate"},
            {"name": "60-Day Balance Delinquency Rate", "column": "deliq_60_acct_bal_rate"},
            {"name": "90-Day Balance Delinquency Rate", "column": "deliq_90_acct_bal_rate"},
            # Consumer delinquency rates
            {"name": "Consumer 30-Day Delinquency Rate", "column": "cnsmr_cnts_w_deliq_bal_30_rate"},
            {"name": "Consumer 60-Day Delinquency Rate", "column": "cnsmr_cnts_w_deliq_bal_60_rate"},
            {"name": "Consumer 90-Day Delinquency Rate", "column": "cnsmr_cnts_w_deliq_bal_90_rate"}
        ],
        "calculated_metrics": [],
        "unavailable": [
            "Total Volume and Percentage of Loans Entering Charge-off",
            "Total Balances and Percentage of Loans Entering Charge-off",
            "Total Volume and Percentage of Loans Entering Repossession",
            "Total Volume and Percentage of Loans Entering Foreclosure"
        ]
    },
    "Originations": {
        "description": "New account origination metrics",
        "required_metrics": [
            {"name": "Total Origination Volume", "column": "tot_new_acct"},
            {"name": "Total Origination Balance", "column": "tot_new_acct_bal"},
            {"name": "Average Origination Balance", "column": "avg_new_acct_bal"}
        ],
        "required_metrics_revolving": [
            {"name": "Total Origination Credit Lines", "column": "tot_new_acct_cr_lines"},
            {"name": "Average Origination Credit Line", "column": "avg_new_acct_cr_lines"},
            {"name": "Total Origination Open-to-Buy", "column": "tot_new_acct_open_to_buy"},
            {"name": "Average Origination Open-to-Buy", "column": "avg_new_acct_open_to_buy"}
        ],
        "calculated_metrics": [],
        "unavailable": ["Median Origination Balance"]
    },
    "Score distributions": {
        "description": "Consumer credit score distribution metrics",
        "required_metrics": [
            {"name": "Total Number of Consumers", "column": "tot_cnsmr_cnts"},
            {"name": "Total Consumers with Balance", "column": "tot_cnsmr_cnts_w_bal"},
            {"name": "Average Accounts Per Consumer", "column": "avg_acct_per_cnsmr"}
        ],
        "calculated_metrics": [],
        "unavailable": [
            "Consumer VantageScore® 4.0 Distribution",
            "Consumer VantageScore® 4.0 Transition Matrix"
        ]
    },
    "Payments": {
        "description": "Payment metrics",
        "required_metrics": [
            {"name": "Average Minimum Payment", "column": "avg_min_payment"}
        ],
        "calculated_metrics": [],
        "unavailable": []
    },
    "Term Distribution": {
        "description": "Loan term distribution",
        "required_metrics": [],
        "calculated_metrics": [],
        "unavailable": ["Term Distribution"]
    }
}

# Product Type Classifications with aliases
REVOLVING_LOBS = [
    "Card", "Bankcard", "Private Label", "HELOC", "All Revolving", 
    "Credit Builder Card", "Private_Label"
]

NON_REVOLVING_LOBS = [
    "Auto", "Mortgage", "HELOAN", "Home Equity Loan", "Unsecured Loan", 
    "Unsecured Personal Loan", "Secured Loan", "Secured Personal Loan",
    "All Installment", "All Non-Revolving", "Student Loan", 
    "Other_Consumer_Loan", "Other Consumer Loan", "Manufactured Housing", 
    "Recreational Merchandise"
]

# Topic to LOB bridge for data filtering
TOPIC_TO_LOB_BRIDGE = {
    "Bankcard": "Card",
    "Private Label": "Private_Label",
    "Unsecured Personal Loan": "Unsecured Loan",
    "Home Equity Loan": "HELOAN",
    "Other Consumer Loan": "Other_Consumer_Loan",
    "Credit Builder Card": "Card",
    "All Revolving": "All_Revolving",
    "All Non-Revolving": "All_Non_Revolving",
    "All Loans & Lines": "All_Loans_Lines"
}

# Group synonyms for normalization
DEFAULT_GROUP_SYNONYMS = {
    # Credit variations
    "credit line": "Credit lines",
    "credit lines": "Credit lines",
    "available credit lines": "Available credit lines",
    "available credit": "Available credit",
    
    # Score variations
    "scores": "Score distributions",
    "score distribution": "Score distributions",
    "score distributions": "Score distributions",
    
    # Delinquency variations
    "delinquency": "Delinquencies",
    "delinquencies": "Delinquencies",
    
    # Origination variations
    "origination": "Originations",
    "originations": "Originations",
    "new accounts": "Originations",
    
    # Balance variations
    "balance": "Balances",
    "balances": "Balances",
    
    # Payment variations
    "payment": "Payments",
    "payments": "Payments",
    
    # Term variations
    "term": "Term Distribution",
    "terms": "Term Distribution",
    "term distribution": "Term Distribution"
}

# Score tier canonical order
TIER_ORDER_CANONICAL = ["Subprime", "Near Prime", "Prime", "Prime Plus", "Super Prime"]

# Delta column patterns for finding existing columns
DELTA_PATTERNS = {
    ("qoq", "pct"): ["{m}_qoq_pct", "{m}_qoq", "{m}_q_q_pct", "{m}_QoQ_%", "{m}_QoQ_pct"],
    ("mom", "pct"): ["{m}_mom_pct", "{m}_mom", "{m}_m_m_pct", "{m}_MoM_%", "{m}_MoM_pct"],
    ("yoy", "pct"): ["{m}_yoy_pct", "{m}_yoy", "{m}_y_y_pct", "{m}_YoY_%", "{m}_YoY_pct", 
                     "{m}_YearOverYear", "{m}_year_over_year"]
}


def topic_to_lob(topic: str) -> str:
    """
    Map topic name to LOB value for data filtering.
    
    Args:
        topic: Topic name from engagement
        
    Returns:
        LOB value for filtering dataframes
    """
    return TOPIC_TO_LOB_BRIDGE.get(topic, topic)


def is_revolving_lob(lob: str) -> bool:
    """
    Check if a LOB is revolving type.
    
    Args:
        lob: Line of business string
        
    Returns:
        True if revolving product type
    """
    s = (lob or "").strip()
    return any(s.lower() == x.lower() for x in REVOLVING_LOBS)


def allowed_groups_for_topic(topic: str) -> List[str]:
    """
    Get allowed insight groups for a topic.
    
    Args:
        topic: Topic name
        
    Returns:
        List of allowed insight group names
    """
    return TOPIC_INSIGHT_GROUPS.get(topic, [])


def expand_group_to_columns(group: str, *, for_revolving: bool) -> List[str]:
    """
    Expand an insight group to its concrete metric columns.
    
    Args:
        group: Insight group name
        for_revolving: Whether this is for a revolving product
        
    Returns:
        List of metric column names
    """
    g = INSIGHT_GROUP_TO_METRICS.get(group, {})
    cols = [m["column"] for m in g.get("required_metrics", [])]
    
    # Add revolving-specific metrics if applicable
    if for_revolving and "required_metrics_revolving" in g:
        cols += [m["column"] for m in g["required_metrics_revolving"]]
    
    return cols


def calculated_specs(group: str) -> List[Dict[str, str]]:
    """
    Get calculated metric specifications for a group.
    
    Args:
        group: Insight group name
        
    Returns:
        List of calculated metric specs with name and formula
    """
    return INSIGHT_GROUP_TO_METRICS.get(group, {}).get("calculated_metrics", [])


def _canonicalize_group(name: str) -> str:
    """
    Canonicalize an insight group name using synonyms.
    
    Args:
        name: Raw group name from engagement
        
    Returns:
        Canonical group name
    """
    n = (name or "").strip()
    return DEFAULT_GROUP_SYNONYMS.get(n.lower(), n)


def get_group_description(group: str) -> str:
    """
    Get description for an insight group.
    
    Args:
        group: Insight group name
        
    Returns:
        Description string
    """
    return INSIGHT_GROUP_TO_METRICS.get(group, {}).get("description", "")


def has_product_restriction(group: str) -> Optional[str]:
    """
    Check if a group has product restrictions.
    
    Args:
        group: Insight group name
        
    Returns:
        Restriction type ('revolving_only') or None
    """
    return INSIGHT_GROUP_TO_METRICS.get(group, {}).get("product_restriction")


def find_delta_pattern(metric: str, delta_type: str, as_pct: bool = True) -> List[str]:
    """
    Get patterns to search for existing delta columns.
    
    Args:
        metric: Base metric name
        delta_type: Delta type (qoq, mom, yoy)
        as_pct: Whether looking for percentage columns
        
    Returns:
        List of column name patterns to try
    """
    key = (delta_type.lower(), "pct" if as_pct else "raw")
    patterns = DELTA_PATTERNS.get(key, [])
    return [p.format(m=metric) for p in patterns]