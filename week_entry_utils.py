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
      - total_hours
      - gross_income
      - after_fuel_income
      - per_shift_gross
      - per_shift_after_fuel
    """
    total = len(selected_dates)
    weekend = sum(1 for d in selected_dates if d.strftime("%A") in WEEKEND_DAYS)

    gross       = round(total * hourly_rate * hours_per_shift, 2)
    fuel_total  = round(total * fuel_cost, 2)
    after_fuel  = round(gross - fuel_total, 2)
    total_hours = round(total * hours_per_shift, 2)

    per_shift_gross = round(gross / total, 2) if total > 0 else 0.0
    per_shift_af    = round(after_fuel / total, 2) if total > 0 else 0.0

    return {
        "total_shifts": total,
        "weekend_shifts": weekend,
        "total_hours": total_hours,
        "gross_income": gross,
        "fuel_total": fuel_total,
        "after_fuel_income": after_fuel,
        "per_shift_gross": per_shift_gross,
        "per_shift_after_fuel": per_shift_af,
    }


# ---------------------------------------------------------------------------
# Natural language shift input parser
# ---------------------------------------------------------------------------

# Map of tokens → weekday index (0=Sun, 1=Mon, ..., 6=Sat) within a Sun-Sat week
_DAY_TOKENS: dict[str, int] = {
    # Full names
    "sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
    "thursday": 4, "friday": 5, "saturday": 6,
    # Common abbreviations
    "sun": 0, "mon": 1, "tue": 2, "tues": 2, "wed": 3, "weds": 3,
    "thu": 4, "thur": 4, "thurs": 4, "fri": 5, "sat": 6,
}

# Tokens that mean "no shifts this week"
_NO_SHIFT_TOKENS = {"no", "off", "none", "nothing", "nope", "nil", "zero", "rest", "rostered"}

# Range keywords
_RANGE_WORDS = {"to", "through", "thru", "-"}


def parse_shift_input(text: str, week_dates_list: list) -> dict:
    """
    Parse a natural language shift description and return matching dates
    from the given week (a list of 7 date objects, Sun–Sat).

    Handles:
    - Day names/abbreviations: "Mon Tue Fri", "monday tuesday friday"
    - Ranges: "Monday to Friday", "Mon-Fri"
    - No shifts: "no shifts", "off", "nothing this week"
    - Mixed: "Mon, Wed, Fri"

    Returns:
        {
            "selected_dates": list[date],   # matched date objects
            "unrecognised":   list[str],    # tokens not understood
            "no_shifts":      bool,         # True if user said no shifts
        }
    """
    import re

    if not text or not text.strip():
        return {"selected_dates": [], "unrecognised": [], "no_shifts": False}

    # Normalise: lowercase, replace hyphens with spaces for range detection
    raw = text.lower().strip()

    # Explicit "no shifts" check before tokenising
    if any(tok in raw.split() for tok in _NO_SHIFT_TOKENS):
        # Only treat as no-shifts if no day names are also present
        has_day = any(d in raw for d in _DAY_TOKENS)
        if not has_day:
            return {"selected_dates": [], "unrecognised": [], "no_shifts": True}

    # Expand hyphenated ranges like "mon-fri" → "mon to fri" before tokenising
    raw = re.sub(r'([a-z]+)-([a-z]+)', r'\1 to \2', raw)

    # Tokenise: split on whitespace and common punctuation
    tokens = re.split(r'[\s,/]+', raw)
    tokens = [t.strip("-") for t in tokens if t.strip("-")]

    # Build a day_index → date lookup for this week
    idx_to_date = {i: week_dates_list[i] for i in range(7)}

    selected_indices: set[int] = set()
    unrecognised: list[str] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        # Skip filler words
        if tok in ("this", "week", "next", "my", "shifts", "and", "also", "plus", "the"):
            i += 1
            continue

        # Ignore no-shift tokens if mixed with day names
        if tok in _NO_SHIFT_TOKENS:
            i += 1
            continue

        if tok in _DAY_TOKENS:
            start_idx = _DAY_TOKENS[tok]

            # Peek ahead for range word + day
            if (i + 2 < len(tokens)
                    and tokens[i + 1] in _RANGE_WORDS
                    and tokens[i + 2] in _DAY_TOKENS):
                end_idx = _DAY_TOKENS[tokens[i + 2]]
                # Range: start → end (inclusive), wrapping Sun=0
                if end_idx >= start_idx:
                    for idx in range(start_idx, end_idx + 1):
                        selected_indices.add(idx)
                else:
                    # Wrap-around range (e.g. "Thu to Mon") — uncommon but handle
                    for idx in range(start_idx, 7):
                        selected_indices.add(idx)
                    for idx in range(0, end_idx + 1):
                        selected_indices.add(idx)
                i += 3
                continue

            selected_indices.add(start_idx)
            i += 1
        elif tok in _RANGE_WORDS:
            i += 1  # skip bare range words
        else:
            unrecognised.append(tok)
            i += 1

    selected_dates = [idx_to_date[idx] for idx in sorted(selected_indices)]
    return {
        "selected_dates": selected_dates,
        "unrecognised": unrecognised,
        "no_shifts": False,
    }
