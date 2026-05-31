"""
week_entry_utils.py
-------------------
Pure Python helpers for the Weekly Roster Entry tab.
No Streamlit, no OpenAI, no side effects.

Week runs Sunday to Saturday (Amazon work week).
"""

from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Week navigation
# ---------------------------------------------------------------------------

def next_sunday(from_date: date = None) -> date:
    """
    Return the next upcoming Sunday.
    If today IS Sunday, return today.
    If from_date is None, uses today.
    """
    if from_date is None:
        from_date = date.today()
    # weekday(): Mon=0 ... Sun=6
    days_until_sunday = (6 - from_date.weekday()) % 7
    return from_date + timedelta(days=days_until_sunday)


def week_dates(week_start: date) -> list:
    """
    Given a Sunday start date, return a list of 7 date objects: Sun, Mon, ..., Sat.
    """
    return [week_start + timedelta(days=i) for i in range(7)]


def day_label(d: date) -> str:
    """
    Return a human-readable label for a date, e.g. 'Sunday 7 Jun'.
    """
    return d.strftime("%A %-d %b")


def week_label(week_start: date) -> str:
    """
    Return a range label for the week, e.g. 'Sun 7 Jun — Sat 13 Jun 2026'.
    """
    week_end = week_start + timedelta(days=6)
    return (
        f"Sun {week_start.strftime('%-d %b')} "
        f"— Sat {week_end.strftime('%-d %b %Y')}"
    )


# ---------------------------------------------------------------------------
# Summary calculations
# ---------------------------------------------------------------------------

DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
WEEKEND_DAYS = {"Saturday", "Sunday"}


def weekly_summary(
    selected_dates: list,
    hourly_rate: float,
    hours_per_shift: float,
    fuel_cost: float,
) -> dict:
    """
    Given a list of selected date objects, return a summary dict:
      - total_shifts
      - weekend_shifts
      - gross_income
      - after_fuel_income
      - per_shift_gross
      - per_shift_after_fuel
    """
    total = len(selected_dates)
    weekend = sum(1 for d in selected_dates if d.strftime("%A") in WEEKEND_DAYS)

    gross = round(total * hourly_rate * hours_per_shift, 2)
    fuel_total = round(total * fuel_cost, 2)
    after_fuel = round(gross - fuel_total, 2)

    per_shift_gross = round(gross / total, 2) if total > 0 else 0.0
    per_shift_af    = round(after_fuel / total, 2) if total > 0 else 0.0

    return {
        "total_shifts": total,
        "weekend_shifts": weekend,
        "gross_income": gross,
        "fuel_total": fuel_total,
        "after_fuel_income": after_fuel,
        "per_shift_gross": per_shift_gross,
        "per_shift_after_fuel": per_shift_af,
    }
