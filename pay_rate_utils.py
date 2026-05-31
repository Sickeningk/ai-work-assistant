"""
pay_rate_utils.py
-----------------
Pure calculation helpers for the Casual Pay Estimator tab.
No Streamlit, no OpenAI, no side effects.

All rates are estimates.
Check Fair Work, your award, enterprise agreement, or payslip for exact rates.
"""

from datetime import date, datetime
from calendar import monthrange


# ---------------------------------------------------------------------------
# Day-type detection
# ---------------------------------------------------------------------------

def classify_days(scheduled_days: list, month_name: str, year: int) -> dict:
    """
    Given a list of day-of-month integers (e.g. [1, 3, 8, 15]),
    return a dict with keys 'weekday', 'saturday', 'sunday' whose
    values are lists of day numbers.

    Public holidays are NOT auto-detected. They are handled separately
    via a manual input in the UI.
    """
    month_num = datetime.strptime(month_name, "%B").month

    weekday_days = []
    saturday_days = []
    sunday_days = []

    for day in scheduled_days:
        try:
            d = date(year, month_num, int(day))
        except ValueError:
            continue  # skip invalid days
        wd = d.weekday()  # 0=Mon, 5=Sat, 6=Sun
        if wd == 5:
            saturday_days.append(day)
        elif wd == 6:
            sunday_days.append(day)
        else:
            weekday_days.append(day)

    return {
        "weekday": weekday_days,
        "saturday": saturday_days,
        "sunday": sunday_days,
    }


# ---------------------------------------------------------------------------
# Per-shift calculations
# ---------------------------------------------------------------------------

def shift_gross(base_rate: float, hours: float, multiplier: float = 1.0, allowance: float = 0.0) -> float:
    """Gross pay for a single shift."""
    return round(base_rate * hours * multiplier + allowance, 2)


def shift_after_fuel(base_rate: float, hours: float, multiplier: float = 1.0,
                     allowance: float = 0.0, fuel_cost: float = 0.0) -> float:
    """Gross minus fuel for a single shift."""
    return round(shift_gross(base_rate, hours, multiplier, allowance) - fuel_cost, 2)


# ---------------------------------------------------------------------------
# Roster-level calculations
# ---------------------------------------------------------------------------

def roster_gross(
    weekday_count: int,
    saturday_count: int,
    sunday_count: int,
    ph_count: int,
    base_rate: float,
    hours: float,
    weekday_mult: float = 1.0,
    saturday_mult: float = 1.0,
    sunday_mult: float = 1.0,
    ph_mult: float = 2.5,
    allowance: float = 0.0,
) -> float:
    """Total gross income for a roster based on shift-type counts."""
    total = (
        shift_gross(base_rate, hours, weekday_mult, allowance) * weekday_count
        + shift_gross(base_rate, hours, saturday_mult, allowance) * saturday_count
        + shift_gross(base_rate, hours, sunday_mult, allowance) * sunday_count
        + shift_gross(base_rate, hours, ph_mult, allowance) * ph_count
    )
    return round(total, 2)


def roster_after_fuel(
    weekday_count: int,
    saturday_count: int,
    sunday_count: int,
    ph_count: int,
    base_rate: float,
    hours: float,
    fuel_cost: float,
    weekday_mult: float = 1.0,
    saturday_mult: float = 1.0,
    sunday_mult: float = 1.0,
    ph_mult: float = 2.5,
    allowance: float = 0.0,
) -> float:
    """Total gross minus all fuel costs for a full roster."""
    total_shifts = weekday_count + saturday_count + sunday_count + ph_count
    gross = roster_gross(
        weekday_count, saturday_count, sunday_count, ph_count,
        base_rate, hours, weekday_mult, saturday_mult, sunday_mult, ph_mult, allowance
    )
    return round(gross - fuel_cost * total_shifts, 2)


# ---------------------------------------------------------------------------
# Skip cost
# ---------------------------------------------------------------------------

def skip_cost(
    base_rate: float,
    hours: float,
    multiplier: float = 1.0,
    fuel_cost: float = 0.0,
    allowance: float = 0.0,
) -> dict:
    """
    What you lose if you skip one shift.
    Returns gross lost and net loss (gross lost minus fuel saved).
    """
    g = shift_gross(base_rate, hours, multiplier, allowance)
    net_loss = round(g - fuel_cost, 2)
    return {"gross_lost": g, "net_loss": net_loss, "fuel_saved": round(fuel_cost, 2)}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def compare_shift_types(
    base_rate: float,
    hours: float,
    fuel_cost: float,
    weekday_mult: float = 1.0,
    saturday_mult: float = 1.0,
    sunday_mult: float = 1.0,
    ph_mult: float = 2.5,
    allowance: float = 0.0,
) -> list:
    """
    Return a list of dicts, one per shift type, with gross and after-fuel values.
    Suitable for rendering as a table.
    """
    types = [
        ("Weekday", weekday_mult),
        ("Saturday", saturday_mult),
        ("Sunday", sunday_mult),
        ("Public Holiday", ph_mult),
    ]
    rows = []
    for label, mult in types:
        g = shift_gross(base_rate, hours, mult, allowance)
        af = shift_after_fuel(base_rate, hours, mult, allowance, fuel_cost)
        rows.append({
            "Shift Type": label,
            "Multiplier": f"{mult:.2f}x",
            "Effective Rate ($/hr)": f"${base_rate * mult:.2f}",
            "Gross / Shift": f"${g:,.2f}",
            "After Fuel / Shift": f"${af:,.2f}",
        })
    return rows
