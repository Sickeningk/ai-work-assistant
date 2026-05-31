"""
payslip_calibration_utils.py
-----------------------------
Pure Python helpers for the Payslip Calibration tab.
No Streamlit, no OpenAI, no side effects.

All outputs are estimates. This is not tax advice.

Line item dict structure:
{
    "description":            str,
    "hours":                  float,   # 0.0 if not applicable
    "rate":                   float,   # 0.0 if not applicable
    "amount":                 float,   # always required
    "category":               str,     # see CATEGORIES below
    "applicable_period_ending": str,   # ISO date string or "" — for adjustments
    "adjustment_note":        str,     # free-text note, e.g. "Double Time Super Applies"
}

CATEGORIES (canonical strings used throughout):
    "current period earning"
    "previous period adjustment"
    "marginal tax"
    "help"
    "deduction"
    "allowance"
    "super"
"""

CATEGORIES = [
    "current period earning",
    "previous period adjustment",
    "marginal tax",
    "help",
    "deduction",
    "allowance",
    "super",
]

EARNING_CATEGORIES = {"current period earning", "previous period adjustment"}
TAX_CATEGORIES     = {"marginal tax"}
HELP_CATEGORIES    = {"help"}
DEDUCTION_CATS     = {"deduction"}
ALLOWANCE_CATS     = {"allowance"}
SUPER_CATS         = {"super"}


# ---------------------------------------------------------------------------
# Categorisation helpers
# ---------------------------------------------------------------------------

def _lines_by_category(items: list, categories: set) -> list:
    return [i for i in items if i.get("category", "").lower() in categories]


def current_period_lines(items: list) -> list:
    return _lines_by_category(items, {"current period earning"})


def adjustment_lines(items: list) -> list:
    return _lines_by_category(items, {"previous period adjustment"})


def tax_lines(items: list) -> list:
    return _lines_by_category(items, TAX_CATEGORIES)


def help_lines(items: list) -> list:
    return _lines_by_category(items, HELP_CATEGORIES)


def deduction_lines(items: list) -> list:
    return _lines_by_category(items, DEDUCTION_CATS)


def allowance_lines(items: list) -> list:
    return _lines_by_category(items, ALLOWANCE_CATS)


def super_lines(items: list) -> list:
    return _lines_by_category(items, SUPER_CATS)


# ---------------------------------------------------------------------------
# Aggregate calculations
# ---------------------------------------------------------------------------

def _sum_amounts(lines: list) -> float:
    return round(sum(i.get("amount", 0.0) for i in lines), 2)


def current_period_gross(items: list) -> float:
    """Sum of current-period earning lines only (excludes adjustments)."""
    return _sum_amounts(current_period_lines(items))


def adjustment_gross(items: list) -> float:
    """Sum of previous-period adjustment earning lines."""
    return _sum_amounts(adjustment_lines(items))


def total_gross(items: list) -> float:
    """Sum of all earning lines (current + adjustments)."""
    return round(current_period_gross(items) + adjustment_gross(items), 2)


def marginal_tax_withheld(items: list) -> float:
    """Sum of all marginal tax lines (positive = withheld)."""
    return _sum_amounts(tax_lines(items))


def help_withheld(items: list) -> float:
    """Sum of HELP withholding lines."""
    return _sum_amounts(help_lines(items))


def total_withheld(items: list) -> float:
    """Marginal tax + HELP combined."""
    return round(marginal_tax_withheld(items) + help_withheld(items), 2)


def total_deductions(items: list) -> float:
    return _sum_amounts(deduction_lines(items))


def total_allowances(items: list) -> float:
    return _sum_amounts(allowance_lines(items))


def total_super(items: list) -> float:
    return _sum_amounts(super_lines(items))


def calculated_net_pay(items: list) -> float:
    """
    gross − marginal_tax − HELP − deductions + allowances.
    Does not include super (employer contribution, not deducted from net).
    """
    gross   = total_gross(items)
    tax     = marginal_tax_withheld(items)
    help_   = help_withheld(items)
    ded     = total_deductions(items)
    allow   = total_allowances(items)
    return round(gross - tax - help_ - ded + allow, 2)


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile(items: list, entered_net: float, tolerance: float = 0.50) -> dict:
    """
    Compare calculated net pay to the entered net pay from the payslip.
    Returns a dict:
        {
            "calculated_net": float,
            "entered_net":    float,
            "difference":     float,
            "is_ok":          bool,
        }
    """
    calc = calculated_net_pay(items)
    diff = round(abs(calc - entered_net), 2)
    return {
        "calculated_net": calc,
        "entered_net":    entered_net,
        "difference":     diff,
        "is_ok":          diff <= tolerance,
    }


# ---------------------------------------------------------------------------
# Effective rates
# ---------------------------------------------------------------------------

def effective_rates(items: list, entered_net: float) -> dict:
    """
    Compute effective rates based on total gross.
    All rates are expressed as decimals (multiply by 100 for %).
    Returns None for a rate if total_gross is zero.
    """
    gross = total_gross(items)
    if gross <= 0:
        return {
            "effective_tax_rate":      None,
            "effective_help_rate":     None,
            "effective_combined_rate": None,
            "effective_net_rate":      None,
        }
    mt   = marginal_tax_withheld(items)
    hlp  = help_withheld(items)
    comb = total_withheld(items)
    return {
        "effective_tax_rate":      round(mt / gross, 6),
        "effective_help_rate":     round(hlp / gross, 6),
        "effective_combined_rate": round(comb / gross, 6),
        "effective_net_rate":      round(entered_net / gross, 6),
    }


# ---------------------------------------------------------------------------
# Adjustment detection
# ---------------------------------------------------------------------------

def has_adjustments(items: list) -> bool:
    return len(adjustment_lines(items)) > 0


def adjustment_period_labels(items: list) -> list:
    """
    Return a list of human-readable labels for each adjustment line,
    e.g. ["Double Time Super Applies — Period ending 17/05/2026"].
    """
    labels = []
    for line in adjustment_lines(items):
        desc   = line.get("description", "Adjustment")
        period = line.get("applicable_period_ending", "")
        note   = line.get("adjustment_note", "")
        parts  = [desc]
        if note and note != desc:
            parts.append(note)
        if period:
            parts.append(f"Period ending {period}")
        labels.append(" — ".join(parts))
    return labels


# ---------------------------------------------------------------------------
# Hourly rate derivation
# ---------------------------------------------------------------------------

def derived_hourly_rate(items: list) -> float | None:
    """
    Derive an effective hourly rate from current-period earning lines only.
    Returns None if total hours across current-period lines is zero.
    Useful for calibrating the app's sidebar hourly_rate.
    """
    lines = current_period_lines(items)
    total_hours  = sum(i.get("hours", 0.0) for i in lines)
    total_amount = sum(i.get("amount", 0.0) for i in lines)
    if total_hours <= 0:
        return None
    return round(total_amount / total_hours, 4)
