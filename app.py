import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import os

from database import (
    init_db,
    load_settings,
    save_settings,
    save_roster,
    load_rosters_dataframe,
    save_weekly_roster,
    load_weekly_rosters_dataframe,
    load_weekly_roster_by_start,
    clear_monthly_rosters,
)
from week_entry_utils import (
    next_sunday,
    week_dates,
    day_label,
    week_label,
    weekly_summary,
    parse_shift_input,
)

from analytics import (
    calculate_analytics_from_days,
    enrich_saved_rosters
)

from ai_helpers import analyze_roster_image
from time_utils import get_next_shift, get_days_until, get_today, build_week_breakdown
from calendar_view import render_calendar
from export_utils import build_export_dataframe
from payslip_calibration_utils import (
    CATEGORIES,
    current_period_gross,
    adjustment_gross,
    total_gross as pc_total_gross,
    marginal_tax_withheld,
    help_withheld,
    total_withheld,
    total_deductions,
    total_allowances,
    total_super,
    calculated_net_pay,
    reconcile as pc_reconcile,
    effective_rates as pc_effective_rates,
    has_adjustments,
    adjustment_period_labels,
    derived_hourly_rate,
)
from pay_rate_utils import (
    classify_days,
    shift_gross,
    shift_after_fuel,
    roster_gross,
    roster_after_fuel,
    skip_cost,
    compare_shift_types,
)
from forecasting import (
    gross_per_shift,
    net_per_shift,
    project_income,
    shifts_needed_for_target,
    cost_of_skipping_shift,
    projected_monthly_income
)


load_dotenv()


def get_openai_client():
    try:
        import streamlit as st
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("your_"):
        st.error(
            "⚠️ OpenAI API key is missing or not configured. "
            "Add your key to Streamlit Cloud secrets or your local .env file as OPENAI_API_KEY."
        )
        st.stop()
    return OpenAI(api_key=key)




def safe_md(text):
    """Escape bare dollar signs so Streamlit does not render them as LaTeX."""
    import re
    # Replace $ not already preceded by a backslash
    return re.sub(r'(?<!\\)\$', r'\\$', text)



def detect_week_question(question):
    """
    Check if the question references a specific week number (e.g. 'week 4').
    Returns the week number as an int, or None if not found.
    """
    import re
    match = re.search(r'\bweek\s*(\d)\b', question, re.IGNORECASE)
    return int(match.group(1)) if match else None


def asks_about_income(question):
    """Return True if the question mentions income, money, or earnings."""
    import re
    keywords = r'\b(income|earn|gross|net|pay|paid|money|make|made|how much)\b'
    return bool(re.search(keywords, question, re.IGNORECASE))


def detect_forecasting_question(question):
    """
    Detect questions that can be answered deterministically by forecasting.py.
    Returns (question_type, extracted_value) or None.

    Types:
        "shifts_for_target"  — how many shifts to reach $X net
        "project_shifts"     — if I work N shifts, what do I earn
        "skip_cost"          — what do I lose skipping a shift
        "net_per_shift"      — net or gross per shift
    """
    import re

    # Shifts needed for target income: "make $5000", "earn 5,000", "hit $3000 net"
    target_match = re.search(
        r'\b(make|earn|need|reach|hit|get to)\b.*?\$?([\d,]+)',
        question, re.IGNORECASE
    )
    if target_match and re.search(
        r'\b(how many shifts|shifts.*(need|take|require))\b', question, re.IGNORECASE
    ):
        raw = target_match.group(2).replace(",", "")
        try:
            return ("shifts_for_target", float(raw))
        except ValueError:
            pass

    # Project income for N shifts: "if I work 4 shifts", "work 10 shifts"
    proj_match = re.search(
        r'\b(if i work|work|do)\s+(\d+)\s+shifts?\b', question, re.IGNORECASE
    )
    if proj_match:
        try:
            return ("project_shifts", int(proj_match.group(2)))
        except ValueError:
            pass

    # Cost of skipping: "skip", "miss", "lose" + "shift"
    if re.search(r'\b(skip|miss|not work|skip a shift|miss a shift)\b', question, re.IGNORECASE):
        if re.search(r'\b(lose|cost|worth|income|earn|make|pay)\b', question, re.IGNORECASE):
            return ("skip_cost", None)

    # Net or gross per shift
    if re.search(
        r'\b(net|gross|earn|make|pay|get)\b.*\b(per shift|a shift|each shift|per day)\b',
        question, re.IGNORECASE
    ):
        return ("net_per_shift", None)

    return None


def format_forecasting_answer(question_type, value, hourly_rate, hours_per_shift, fuel_cost, roster_label):
    """Build a deterministic forecasting answer. Never calls OpenAI."""
    gps = gross_per_shift(hourly_rate, hours_per_shift)

    if question_type == "shifts_for_target":
        target = value
        needed = shifts_needed_for_target(target, hourly_rate, hours_per_shift, fuel_cost)
        if needed is None:
            return (
                "⚠️ Gross income per shift is zero — check your hourly rate and hours per shift settings."
            )
        proj = project_income(needed, hourly_rate, hours_per_shift, fuel_cost)
        lines = [
            f"To reach **\\${target:,.2f} gross**, you need **{needed} shifts**.",
            "",
            f"- Gross per shift: \\${gps:,.2f}",
            f"- {needed} shifts gross: \\${proj['gross']:,.2f}",
        ]
        return "\n".join(lines)

    if question_type == "project_shifts":
        n = int(value)
        proj = project_income(n, hourly_rate, hours_per_shift, fuel_cost)
        lines = [
            f"If you work **{n} shift{'s' if n != 1 else ''}**:",
            "",
            f"- Estimated gross income: \\${proj['gross']:,.2f}",
            f"- Gross per shift: \\${gps:,.2f}",
        ]
        return "\n".join(lines)

    if question_type == "skip_cost":
        skip = cost_of_skipping_shift(hourly_rate, hours_per_shift, fuel_cost)
        lines = [
            "If you skip one shift:",
            "",
            f"- **Gross income lost: \\${skip['lost_gross']:,.2f}**",
        ]
        return "\n".join(lines)

    if question_type == "net_per_shift":
        lines = [
            "Per shift breakdown:",
            "",
            f"- **Gross per shift: \\${gps:,.2f}**",
            f"- Use **🧾 Payslip Calibration** to estimate take-home after tax and HELP.",
        ]
        return "\n".join(lines)

    return None


def detect_app_question(question):
    """
    Return True if the question is about app behavior, not roster data.
    These questions are out of scope for Current Roster chat.
    """
    import re
    app_keywords = (
        r"\b(app|deploy|cloud|export|api.?key|secret|database|sqlite|storage|"
        r"bug|error|button|tab|streamlit|install|setting|config|server|"
        r"load properly|restart|reboot|disappear|reset|crash|ui|interface)\b"
    )
    return bool(re.search(app_keywords, question, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Weekly Assistant — deterministic question detection and formatting
# ---------------------------------------------------------------------------

# Shared day-token map (Sun=0 … Sat=6 within a Sun-Sat week)
_WA_DAY_TOKENS = {
    "sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
    "thursday": 4, "friday": 5, "saturday": 6,
    "sun": 0, "mon": 1, "tue": 2, "tues": 2, "wed": 3, "weds": 3,
    "thu": 4, "thur": 4, "thurs": 4, "fri": 5, "sat": 6,
}


def detect_wa_question(question: str, week_dates_list: list):
    """
    Detect Weekly Assistant specific questions before sending to OpenAI.

    Returns (type, value) or None.
      "skip_day"           — value: date object
      "add_day"            — value: date object
      "shifts_for_takehome"— value: float target $
      "shifts_for_gross"   — value: float target $
      "projected_payslip"  — value: None
      "can_afford"         — value: float amount $
    """
    import re
    q = question.lower().strip()

    # --- Skip a day -------------------------------------------------------
    skip_m = re.search(r'\bskip\s+(\w+)', q)
    if skip_m:
        tok = skip_m.group(1)
        if tok in _WA_DAY_TOKENS:
            return ("skip_day", week_dates_list[_WA_DAY_TOKENS[tok]])

    # --- Add / work a day -------------------------------------------------
    for _pat in [
        r'\badd\s+(\w+)',
        r'\bif\s+i\s+(?:add|work|do)\s+(\w+)',
        r'\bwhat\s+if\s+i\s+(?:add|work)\s+(\w+)',
        r'\binclude\s+(\w+)',
    ]:
        _m = re.search(_pat, q)
        if _m:
            tok = _m.group(1)
            if tok in _WA_DAY_TOKENS:
                return ("add_day", week_dates_list[_WA_DAY_TOKENS[tok]])

    # --- Shifts for take-home target --------------------------------------
    # Must mention take-home / net / after tax AND a dollar amount
    if re.search(r'\b(take.?home|net pay|after.?tax|after tax)\b', q):
        if re.search(r'\b(how many|shifts|need)\b', q):
            money_m = re.search(r'\$?([\d,]+)', q)
            if money_m:
                try:
                    val = float(money_m.group(1).replace(",", ""))
                    if val > 0:
                        return ("shifts_for_takehome", val)
                except ValueError:
                    pass

    # --- Shifts for gross target ------------------------------------------
    # "how many shifts to make $X" / "shifts needed for $X" (no take-home mention)
    if re.search(r'\b(how many shifts|shifts.*(need|take|required?))\b', q):
        money_m = re.search(r'\$?([\d,]+)', q)
        if money_m:
            try:
                val = float(money_m.group(1).replace(",", ""))
                if val > 0:
                    return ("shifts_for_gross", val)
            except ValueError:
                pass

    # --- Projected payslip ------------------------------------------------
    if re.search(r'\b(payslip|pay slip|pay breakdown|breakdown|projected pay)\b', q):
        return ("projected_payslip", None)

    # --- Can I afford X ---------------------------------------------------
    afford_m = re.search(r'\bafford\b.*?\$?([\d,]+)', q)
    if not afford_m:
        afford_m = re.search(r'\$?([\d,]+).*\bafford\b', q)
    if afford_m:
        try:
            val = float(afford_m.group(1).replace(",", ""))
            if val > 0:
                return ("can_afford", val)
        except ValueError:
            pass

    return None


def format_wa_answer(
    q_type: str,
    value,
    selected_dates: list,
    hourly_rate: float,
    hours_per_shift: float,
    pc_rates: dict | None,
    week_label_str: str,
) -> str | None:
    """
    Build a fully deterministic answer for a Weekly Assistant question.
    Never calls OpenAI. All money is Python-calculated.
    """
    from math import ceil

    def _sum(dates):
        return weekly_summary(dates, hourly_rate, hours_per_shift, 0.0)

    def _gps():
        return round(hourly_rate * hours_per_shift, 2)

    net_rate      = (pc_rates or {}).get("effective_net_rate")
    combined_rate = (pc_rates or {}).get("effective_combined_rate")

    # Refuse to calculate if no context
    no_shifts_msg = (
        "No shifts are entered for this week yet — "
        "type your shifts in the box above first."
    )

    # ------------------------------------------------------------------
    if q_type == "skip_day":
        day = value
        day_str = day.strftime("%A %-d %b")
        if not selected_dates:
            return no_shifts_msg
        if day not in selected_dates:
            return f"You don't have a shift on **{day_str}** this week, so there's nothing to skip."
        new_dates = [d for d in selected_dates if d != day]
        old_s = _sum(selected_dates)
        new_s = _sum(new_dates) if new_dates else {
            "total_shifts": 0, "total_hours": 0.0, "gross_income": 0.0
        }
        lines = [
            f"If you skip **{day_str}**:",
            "",
            f"- Shifts: {new_s['total_shifts']} (was {old_s['total_shifts']})",
            f"- Total hours: {new_s.get('total_hours', 0.0):.1f} h "
            f"(was {old_s.get('total_hours', 0.0):.1f} h)",
            f"- Gross: \\${new_s['gross_income']:,.2f} (was \\${old_s['gross_income']:,.2f})",
            f"- Gross lost: \\${_gps():,.2f}",
        ]
        if net_rate:
            old_net = round(old_s["gross_income"] * net_rate, 2)
            new_net = round(new_s["gross_income"] * net_rate, 2)
            lines.append(
                f"- Est. take-home: \\${new_net:,.2f} (was \\${old_net:,.2f})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    if q_type == "add_day":
        day = value
        day_str = day.strftime("%A %-d %b")
        if day in selected_dates:
            return f"You're already working **{day_str}** this week."
        new_dates = sorted(selected_dates + [day])
        old_s = _sum(selected_dates) if selected_dates else {
            "total_shifts": 0, "total_hours": 0.0, "gross_income": 0.0
        }
        new_s = _sum(new_dates)
        lines = [
            f"If you add **{day_str}**:",
            "",
            f"- Shifts: {new_s['total_shifts']} (was {old_s['total_shifts']})",
            f"- Total hours: {new_s.get('total_hours', 0.0):.1f} h "
            f"(was {old_s.get('total_hours', 0.0):.1f} h)",
            f"- Gross: \\${new_s['gross_income']:,.2f} (was \\${old_s['gross_income']:,.2f})",
            f"- Extra gross: \\${_gps():,.2f}",
        ]
        if net_rate:
            old_net = round(old_s["gross_income"] * net_rate, 2)
            new_net = round(new_s["gross_income"] * net_rate, 2)
            lines.append(
                f"- Est. take-home: \\${new_net:,.2f} (was \\${old_net:,.2f})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    if q_type == "shifts_for_takehome":
        target = value
        gps = _gps()
        if not net_rate or net_rate <= 0:
            n = ceil(target / gps) if gps > 0 else None
            if n is None:
                return "⚠️ Gross per shift is zero — check your hourly rate and hours per shift settings."
            proj_gross = round(n * gps, 2)
            return (
                f"No payslip calibration found — can't estimate take-home directly.\n\n"
                f"To earn \\${target:,.2f} **gross** (closest proxy), "
                f"you need **{n} shifts** (≈ \\${proj_gross:,.2f} gross).\n\n"
                "Add your payslip in **🧾 Payslip Calibration** "
                "to calculate shifts needed for a take-home target."
            )
        net_per_shift = round(gps * net_rate, 2)
        if net_per_shift <= 0:
            return "⚠️ Net per shift is zero — check your settings and payslip calibration."
        n = ceil(target / net_per_shift)
        proj_gross = round(n * gps, 2)
        proj_net   = round(n * net_per_shift, 2)
        return (
            f"To take home **\\${target:,.2f}**, you need **{n} shifts**.\n\n"
            f"- Gross per shift: \\${gps:,.2f}\n"
            f"- Est. take-home per shift: \\${net_per_shift:,.2f} "
            f"(net rate {net_rate*100:.1f}%)\n"
            f"- {n} shifts → gross \\${proj_gross:,.2f} → est. take-home \\${proj_net:,.2f}\n\n"
            "*Based on Payslip Calibration effective net rate.*"
        )

    # ------------------------------------------------------------------
    if q_type == "shifts_for_gross":
        target = value
        gps = _gps()
        if gps <= 0:
            return "⚠️ Gross per shift is zero — check your hourly rate and hours per shift settings."
        n = ceil(target / gps)
        proj_gross = round(n * gps, 2)
        lines = [
            f"To earn **\\${target:,.2f} gross**, you need **{n} shifts**.",
            "",
            f"- Gross per shift: \\${gps:,.2f}",
            f"- {n} shifts → \\${proj_gross:,.2f} gross",
        ]
        if net_rate:
            proj_net = round(proj_gross * net_rate, 2)
            lines.append(f"- Est. take-home: \\${proj_net:,.2f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    if q_type == "projected_payslip":
        if not selected_dates:
            return no_shifts_msg
        s = _sum(selected_dates)
        gross = s["gross_income"]
        lines = [
            f"**Projected payslip — {week_label_str}**",
            "",
            f"| Line | Amount |",
            f"|---|---:|",
            f"| Gross ({s['total_shifts']} shifts × {hours_per_shift} h @ \\${hourly_rate}/hr) "
            f"| \\${gross:,.2f} |",
        ]
        if combined_rate is not None and combined_rate > 0:
            withheld = round(gross * combined_rate, 2)
            lines.append(
                f"| Est. tax & HELP withheld ({combined_rate*100:.1f}%) "
                f"| −\\${withheld:,.2f} |"
            )
        if net_rate is not None and net_rate > 0:
            net = round(gross * net_rate, 2)
            lines.append(f"| **Est. take-home** | **\\${net:,.2f}** |")
        lines.append("")
        if not net_rate:
            lines.append(
                "⚠️ No payslip calibration — gross only. "
                "Add your payslip in **🧾 Payslip Calibration** for tax/HELP estimates."
            )
        else:
            lines.append(
                "*Estimates only. Based on Payslip Calibration effective rates. "
                "Not tax advice.*"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    if q_type == "can_afford":
        amount = value
        if not selected_dates:
            return no_shifts_msg
        s = _sum(selected_dates)
        gross = s["gross_income"]
        if net_rate and net_rate > 0:
            income       = round(gross * net_rate, 2)
            income_label = "estimated take-home"
        else:
            income       = gross
            income_label = "estimated gross (no payslip calibration)"
        if income >= amount:
            left = round(income - amount, 2)
            return (
                f"Your {income_label} this week is **\\${income:,.2f}**. "
                f"\\${amount:,.2f} would leave you **\\${left:,.2f}**. ✅"
            )
        else:
            short = round(amount - income, 2)
            return (
                f"Your {income_label} this week is **\\${income:,.2f}**. "
                f"\\${amount:,.2f} would put you **\\${short:,.2f} short**. ⚠️"
            )

    return None



def detect_saved_week_question(question):
    """
    Extract week number and month name(s) from a saved-roster question.
    Returns (week_num: int | None, months: list[str])
    Months are returned as full names e.g. ["April", "May"].
    """
    import re
    MONTHS = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    week_match = re.search(r'\bweek\s*(\d)\b', question, re.IGNORECASE)
    week_num = int(week_match.group(1)) if week_match else None

    found_months = [m for m in MONTHS if re.search(r'\b' + m + r'\b', question, re.IGNORECASE)]
    return week_num, found_months



def format_week_answer(week_data, include_income=False, roster_label=None):
    """
    Build a deterministic markdown answer from a week_breakdown entry.
    Never calls OpenAI.
    Uses \\$ so dollar signs render correctly after safe_md is applied.
    roster_label: optional string like "April 2026" shown in the header.
    """
    label_str = f" ({roster_label})" if roster_label else ""
    lines = [
        f"**Week {week_data['week_number']}{label_str}: {week_data['week_start']} to {week_data['week_end']}**",
        "",
    ]
    if week_data["shift_count"] == 0:
        lines.append("There were no scheduled shifts this week.")
    else:
        lines.append(f"You worked {week_data['shift_count']} shift{'s' if week_data['shift_count'] != 1 else ''}:")
        for d in week_data["shift_dates"]:
            lines.append(f"- {d}")
        if include_income:
            lines.append("")
            lines.append(f"- Gross income: \\${week_data['gross_income']:,.2f}")
    return "\n".join(lines)


init_db()

saved_hourly_rate, saved_hours_per_shift, saved_fuel_cost = load_settings()

# Pass A: fuel_cost removed from UI. Fixed at 0.0 so all existing function
# signatures continue to work without modification. Pass B will clean up
# helper signatures and the DB column when approved.
fuel_cost = 0.0

st.set_page_config(
    page_title="AI Work Assistant",
    layout="wide"
)

st.title("AI Work Assistant")

st.sidebar.header("Settings")

hourly_rate = st.sidebar.number_input(
    "Hourly Rate ($)",
    min_value=0.0,
    value=float(saved_hourly_rate),
    step=1.0
)

hours_per_shift = st.sidebar.number_input(
    "Hours Per Shift",
    min_value=0.0,
    value=float(saved_hours_per_shift),
    step=0.5
)

if st.sidebar.button("Save Settings"):
    save_settings(
        hourly_rate,
        hours_per_shift,
        fuel_cost   # preserves DB column; always saves 0.0 going forward
    )

    st.sidebar.success("Settings saved.")

st.sidebar.write("Current Settings")
st.sidebar.write(f"Hourly Rate: ${hourly_rate}")
st.sidebar.write(f"Hours Per Shift: {hours_per_shift}")

st.sidebar.divider()
show_legacy = st.sidebar.checkbox("Show Advanced / Legacy Tools", value=False)
show_debug = st.sidebar.checkbox("Show Developer Debug", value=False)
show_older_answers = st.sidebar.checkbox("Show older roster answers", value=False)

if "roster_data" not in st.session_state:
    st.session_state.roster_data = None

if "roster_chat" not in st.session_state:
    st.session_state.roster_chat = []

if "saved_chat" not in st.session_state:
    st.session_state.saved_chat = []

if "active_roster_key" not in st.session_state:
    st.session_state.active_roster_key = None

if "casual_pay_settings" not in st.session_state:
    st.session_state.casual_pay_settings = {}

if "weekly_entry" not in st.session_state:
    st.session_state.weekly_entry = {}  # stores last saved weekly roster in session

if "payslip_lines" not in st.session_state:
    st.session_state.payslip_lines = []  # list of line item dicts for payslip calibration

if "weekly_assistant_chat" not in st.session_state:
    st.session_state.weekly_assistant_chat = []

if "weekly_assistant_week_key" not in st.session_state:
    st.session_state.weekly_assistant_week_key = None

# Stores the last NL text that was parsed (for checkbox-sync detection)
if "wa_last_nl_parsed" not in st.session_state:
    st.session_state.wa_last_nl_parsed = ""

# Stores the last week key seen (for detecting week change)
if "wa_last_week_key" not in st.session_state:
    st.session_state.wa_last_week_key = None

# Entered net pay from Payslip Calibration — used to compute take-home rate
if "wa_entered_net" not in st.session_state:
    st.session_state.wa_entered_net = 0.0

# ---------------------------------------------------------------------------
# Tab layout — 3 main tabs always visible; legacy tools shown when toggled
# ---------------------------------------------------------------------------
if show_legacy:
    tab_wa, tab_tw, tab_pc, tab_we, tab1, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗓 Weekly Assistant",
        "📊 This Week",
        "🧾 Payslip Calibration",
        "🗓 Weekly Entry",
        "📅 Current Roster",
        "🤖 AI Assistant",
        "📚 History",
        "📈 Forecasting",
        "📥 Export",
        "💰 Casual Pay Estimator",
    ])
else:
    tab_wa, tab_tw, tab_pc = st.tabs([
        "🗓 Weekly Assistant",
        "📊 This Week",
        "🧾 Payslip Calibration",
    ])


# ---------------------------------------------------------------------------
# Tab: Weekly Assistant (main)
# ---------------------------------------------------------------------------
with tab_wa:
    import datetime as _dt_wa

    st.header("🗓 Weekly Assistant")

    # ── Week selector ──────────────────────────────────────────────────────
    _wa_col_left, _wa_col_right = st.columns([2, 5])
    with _wa_col_left:
        _wa_default_sunday = next_sunday(_dt_wa.date.today())
        _wa_week_start = st.date_input(
            "Week starting (Sunday)",
            value=_wa_default_sunday,
            key="wa_week_start",
        )

    if _wa_week_start.weekday() != 6:
        st.warning(
            f"⚠️ {_wa_week_start.strftime('%A %-d %b')} is not a Sunday. "
            "Please select a Sunday as the week start."
        )
    else:
        _wa_week_end  = _wa_week_start + _dt_wa.timedelta(days=6)
        _wa_all_dates = week_dates(_wa_week_start)
        _wa_week_key  = _wa_week_start.isoformat()

        # ── Detect week change → reset NL sync state ──────────────────────
        if st.session_state.wa_last_week_key != _wa_week_key:
            st.session_state.wa_last_nl_parsed = ""
            for _d in _wa_all_dates:
                _k = f"wa_day_{_d.isoformat()}"
                if _k in st.session_state:
                    del st.session_state[_k]
            if (
                st.session_state.weekly_assistant_week_key is not None
                and st.session_state.weekly_assistant_week_key != _wa_week_key
                and st.session_state.weekly_assistant_chat
            ):
                st.warning(
                    "⚠️ Week changed. Earlier chat messages refer to a different week."
                )
            st.session_state.wa_last_week_key = _wa_week_key
        st.session_state.weekly_assistant_week_key = _wa_week_key

        # ── Load saved roster for this week ───────────────────────────────
        _wa_existing      = load_weekly_roster_by_start(_wa_week_key)
        _wa_existing_isos = set(_wa_existing["scheduled_dates"]) if _wa_existing else set()

        # ── Conversational prompt ─────────────────────────────────────────
        st.markdown(
            f"#### What shifts did Amazon give you for the week starting "
            f"**{_wa_week_start.strftime('Sunday %-d %b')}**?"
        )

        _wa_nl = st.text_input(
            "shifts_input",
            placeholder="Mon Tue Fri  ·  Monday to Friday  ·  No shifts this week",
            key="wa_nl_input",
            label_visibility="collapsed",
        )

        # ── Sync checkboxes when NL input changes ─────────────────────────
        if _wa_nl != st.session_state.wa_last_nl_parsed:
            if _wa_nl.strip():
                _wa_p      = parse_shift_input(_wa_nl, _wa_all_dates)
                _wa_p_isos = {d.isoformat() for d in _wa_p["selected_dates"]}
                _wa_p_no   = _wa_p["no_shifts"]
                if _wa_p["unrecognised"]:
                    st.warning(
                        f"⚠️ Unrecognised: {', '.join(_wa_p['unrecognised'])}"
                    )
                for _d in _wa_all_dates:
                    st.session_state[f"wa_day_{_d.isoformat()}"] = (
                        False if _wa_p_no else (_d.isoformat() in _wa_p_isos)
                    )
            else:
                # NL cleared — restore from saved roster
                for _d in _wa_all_dates:
                    st.session_state[f"wa_day_{_d.isoformat()}"] = (
                        _d.isoformat() in _wa_existing_isos
                    )
            st.session_state.wa_last_nl_parsed = _wa_nl

        # ── Fine-tune expander with checkboxes ────────────────────────────
        with st.expander("Fine-tune shifts", expanded=False):
            st.caption("Tick to adjust individual days. Overrides the text input above.")
            _wa_selected = []
            _wa_chk_cols = st.columns(7)
            for _i, _d in enumerate(_wa_all_dates):
                _iso     = _d.isoformat()
                _default = st.session_state.get(f"wa_day_{_iso}", _iso in _wa_existing_isos)
                _checked = _wa_chk_cols[_i].checkbox(
                    day_label(_d), value=_default, key=f"wa_day_{_iso}"
                )
                if _checked:
                    _wa_selected.append(_d)

        # ── Payslip calibration rates ─────────────────────────────────────
        _wa_pc_items    = st.session_state.payslip_lines
        _wa_pc_gross    = pc_total_gross(_wa_pc_items) if _wa_pc_items else 0.0
        _wa_entered_net = st.session_state.wa_entered_net  # set by Payslip Calibration tab

        # Full rates (including take-home): requires entered_net > 0
        _wa_pc_rates = (
            pc_effective_rates(_wa_pc_items, _wa_entered_net)
            if _wa_pc_items and _wa_pc_gross > 0 and _wa_entered_net > 0
            else None
        )
        # Tax/HELP-only rates: valid even without entered_net
        _wa_pc_tax_rates = (
            pc_effective_rates(_wa_pc_items, 0.0)
            if _wa_pc_items and _wa_pc_gross > 0
            else None
        )

        # ── Summary card ──────────────────────────────────────────────────
        if not _wa_selected:
            if not _wa_nl.strip() and not _wa_existing_isos:
                st.caption(
                    "Type your shifts above — e.g. **Mon Tue Fri** or **Monday to Friday** — "
                    "to see your weekly summary."
                )
            else:
                st.info("No shifts this week.")
        else:
            _wa_sum = weekly_summary(_wa_selected, hourly_rate, hours_per_shift, fuel_cost)

            st.divider()
            st.markdown("**Your week at a glance**")

            # Shift date list
            _wa_date_str = "  ·  ".join(
                d.strftime("%a %-d %b") for d in sorted(_wa_selected)
            )
            st.markdown(f"📅 {_wa_date_str}")
            st.caption(
                f"Week: **{_wa_week_start.strftime('%-d %b')}** — "
                f"**{_wa_week_end.strftime('%-d %b %Y')}**"
            )

            # Metrics row
            # Defensive fallback for total_hours in case of stale bytecode
            _wa_total_hours = _wa_sum.get(
                "total_hours",
                round(_wa_sum["total_shifts"] * hours_per_shift, 2)
            )
            _wm1, _wm2, _wm3, _wm4 = st.columns(4)
            _wm1.metric("Shifts",       _wa_sum["total_shifts"])
            _wm2.metric("Total Hours",  f"{_wa_total_hours:.1f} h")
            _wm3.metric("Gross (est.)", f"${_wa_sum['gross_income']:,.2f}")

            # Take-home / withheld
            if _wa_pc_rates and _wa_pc_rates.get("effective_net_rate"):
                _wa_net_rate = _wa_pc_rates["effective_net_rate"]
                _wa_net_est  = round(_wa_sum["gross_income"] * _wa_net_rate, 2)
                _wa_withheld = round(_wa_sum["gross_income"] - _wa_net_est, 2)
                _wm4.metric("Est. Take-Home", f"${_wa_net_est:,.2f}")
                st.caption(
                    f"Tax & HELP withheld (est.): −\\${_wa_withheld:,.2f}  ·  "
                    f"Net rate: {_wa_net_rate*100:.1f}%  ·  "
                    "Recalibrate after each payslip."
                )
            elif _wa_pc_tax_rates and _wa_pc_tax_rates.get("effective_combined_rate"):
                _wa_cr        = _wa_pc_tax_rates["effective_combined_rate"]
                _wa_w_est     = round(_wa_sum["gross_income"] * _wa_cr, 2)
                _wm4.metric("Est. Withheld", f"−${_wa_w_est:,.2f}")
                st.caption(
                    f"Tax & HELP withheld (est.): −\\${_wa_w_est:,.2f} ({_wa_cr*100:.1f}%)  ·  "
                    "Enter your net pay in **🧾 Payslip Calibration → Calculate** "
                    "to unlock take-home estimate."
                )
            else:
                st.caption(
                    "Add your payslip in **🧾 Payslip Calibration** "
                    "to estimate tax/HELP and take-home pay."
                )

            # Save button
            if _wa_existing:
                st.caption(
                    f"⚠️ Roster already saved ({_wa_existing['created_at']}). "
                    "Saving will overwrite."
                )
            if st.button("💾 Save Weekly Roster", key="wa_save_btn"):
                _iso_list = [d.isoformat() for d in _wa_selected]
                save_weekly_roster(
                    week_start_date=_wa_week_key,
                    week_end_date=_wa_week_end.isoformat(),
                    scheduled_dates=_iso_list,
                    source="manual",
                )
                st.session_state.weekly_entry = {
                    "week_start": _wa_week_key,
                    "week_end":   _wa_week_end.isoformat(),
                    "dates":      _iso_list,
                    "summary":    _wa_sum,
                }
                st.success(
                    f"✅ Saved {_wa_sum['total_shifts']} shifts "
                    f"for {week_label(_wa_week_start)}."
                )
                st.rerun()

        st.divider()

        # ── Weekly chat ───────────────────────────────────────────────────
        st.subheader("Ask about this week")
        st.caption(
            "Skip a shift  ·  Add a shift  ·  Projected payslip  ·  Income goals  ·  Savings"
        )

        for _msg in st.session_state.weekly_assistant_chat:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])

        _wa_q = st.chat_input(
            "e.g.  What if I skip Tuesday?  ·  How many shifts for $1,000 take-home?",
            key="wa_chat_input",
        )

        if _wa_q:
            st.session_state.weekly_assistant_chat.append(
                {"role": "user", "content": _wa_q}
            )

            # Context: current checkboxes → fall back to saved roster
            _wa_ctx_dates = _wa_selected or [
                _dt_wa.date.fromisoformat(d) for d in sorted(_wa_existing_isos)
            ]
            _wa_ctx_sum = (
                weekly_summary(_wa_ctx_dates, hourly_rate, hours_per_shift, fuel_cost)
                if _wa_ctx_dates else None
            )

            # ── Deterministic routing: WA-specific questions ───────────────
            _wa_det = detect_wa_question(_wa_q, _wa_all_dates)
            if _wa_det is not None:
                _wa_det_type, _wa_det_val = _wa_det
                _wa_det_reply = format_wa_answer(
                    _wa_det_type, _wa_det_val,
                    _wa_ctx_dates,
                    hourly_rate, hours_per_shift,
                    _wa_pc_rates,
                    week_label(_wa_week_start),
                )
                if _wa_det_reply is not None:
                    st.session_state.weekly_assistant_chat.append(
                        {"role": "assistant", "content": _wa_det_reply}
                    )
                    st.rerun()

            # ── Deterministic routing: gross forecasting fallback ──────────
            _wa_fc = detect_forecasting_question(_wa_q)
            if _wa_fc is not None:
                _ftype, _fval = _wa_fc
                _wa_fc_reply = format_forecasting_answer(
                    _ftype, _fval, hourly_rate, hours_per_shift, fuel_cost,
                    week_label(_wa_week_start)
                )
                if _wa_fc_reply:
                    st.session_state.weekly_assistant_chat.append(
                        {"role": "assistant", "content": _wa_fc_reply}
                    )
                    st.rerun()

            # ── OpenAI fallback ────────────────────────────────────────────
            _wa_sys = (
                "You are a helpful personal work assistant for an Amazon casual warehouse worker.\n"
                f"Week: {week_label(_wa_week_start)}\n"
                f"Settings: ${hourly_rate}/hr, {hours_per_shift} hrs/shift.\n"
            )
            if _wa_ctx_sum:
                _wa_sys += (
                    f"Shifts: {_wa_ctx_sum['total_shifts']} "
                    f"({_wa_ctx_sum['weekend_shifts']} weekend), "
                    f"{_wa_ctx_sum.get('total_hours', round(_wa_ctx_sum['total_shifts'] * hours_per_shift, 2)):.1f} h total\n"
                    f"Dates: {[d.strftime('%A %-d %b') for d in sorted(_wa_ctx_dates)]}\n"
                    f"Gross (est.): ${_wa_ctx_sum['gross_income']:,.2f}\n"
                )
            else:
                _wa_sys += "No shifts entered for this week.\n"

            if _wa_pc_rates and _wa_ctx_sum and _wa_pc_rates.get("effective_net_rate"):
                _r = _wa_pc_rates["effective_net_rate"]
                _wa_sys += (
                    f"Payslip net rate: {_r*100:.1f}%\n"
                    f"Est. take-home: "
                    f"${round(_wa_ctx_sum['gross_income'] * _r, 2):,.2f}\n"
                )
            elif _wa_pc_tax_rates and _wa_pc_tax_rates.get("effective_combined_rate"):
                _cr = _wa_pc_tax_rates["effective_combined_rate"]
                _wa_sys += (
                    f"Est. tax & HELP rate: {_cr*100:.1f}% "
                    "(take-home not calibrated)\n"
                )
            else:
                _wa_sys += "No payslip calibration available.\n"

            _wa_sys += (
                "\nCritical rules:\n"
                "- NEVER calculate shifts, dates, gross pay, tax, HELP, or take-home yourself.\n"
                "- If asked about pay/shifts/income, say the summary above shows the numbers "
                "or ask the user to enter shifts first.\n"
                "- Keep answers short and practical.\n"
                "- Format money as \\$X,XXX.XX."
            )

            try:
                _wa_resp = get_openai_client().chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "system", "content": _wa_sys}]
                    + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.weekly_assistant_chat[-10:]
                    ],
                )
                _wa_reply = safe_md(_wa_resp.choices[0].message.content)
                st.session_state.weekly_assistant_chat.append(
                    {"role": "assistant", "content": _wa_reply}
                )
            except Exception as _e:
                st.error(f"⚠️ AI error: {_e}")
                st.session_state.weekly_assistant_chat.pop()

            st.rerun()

        if st.session_state.weekly_assistant_chat:
            if st.button("Clear chat", key="wa_clear_chat"):
                st.session_state.weekly_assistant_chat = []
                st.rerun()


# ---------------------------------------------------------------------------
# Tab: Weekly Entry (legacy — visible when Show Advanced / Legacy Tools is ON)
# ---------------------------------------------------------------------------
if show_legacy:
 with tab_we:

    import datetime as _dt

    st.header("🗓 Weekly Roster Entry")
    st.caption(
        "Enter your Amazon shifts for the coming week. "
        "Your work week runs **Sunday to Saturday**. "
        "Use this every Friday when your roster drops."
    )

    # --- Week selector ---
    st.subheader("Select week")

    _default_sunday = next_sunday(_dt.date.today())

    _we_week_start = st.date_input(
        "Week starting (Sunday)",
        value=_default_sunday,
        key="we_week_start",
        help="Defaults to the next upcoming Sunday. You can pick a different week.",
    )

    # Enforce Sunday — if user picks a non-Sunday, warn and nudge
    if _we_week_start.weekday() != 6:
        st.warning(
            f"⚠️ {_we_week_start.strftime('%A %-d %b')} is not a Sunday. "
            "Please select a Sunday as the week start."
        )
    else:
        _we_week_end = _we_week_start + _dt.timedelta(days=6)
        _we_all_dates = week_dates(_we_week_start)

        st.markdown(f"**Week:** {week_label(_we_week_start)}")

        # --- Check if a saved roster exists for this week ---
        _existing = load_weekly_roster_by_start(_we_week_start.isoformat())
        _existing_dates = set(_existing["scheduled_dates"]) if _existing else set()

        st.divider()
        st.subheader("Select your shifts")
        st.caption("Tick each day you are working. Dates are shown next to each day.")

        _selected_dates = []
        _cols = st.columns(7)
        _day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        for i, _d in enumerate(_we_all_dates):
            _iso = _d.isoformat()
            _label = day_label(_d)          # e.g. "Sunday 7 Jun"
            _default_checked = _iso in _existing_dates
            _checked = _cols[i].checkbox(
                _label,
                value=_default_checked,
                key=f"we_day_{_iso}",
            )
            if _checked:
                _selected_dates.append(_d)

        st.divider()

        # --- Weekly summary ---
        st.subheader("Weekly summary")

        if not _selected_dates:
            st.info("No shifts selected yet. Tick the days above to see your estimate.")
        else:
            _summary = weekly_summary(
                _selected_dates,
                hourly_rate,
                hours_per_shift,
                fuel_cost,
            )

            _s_col1, _s_col2, _s_col3 = st.columns(3)
            _s_col1.metric("Total Shifts",   _summary["total_shifts"])
            _s_col2.metric("Weekend Shifts", _summary["weekend_shifts"])
            _s_col3.metric("Gross Income",   f"${_summary['gross_income']:,.2f}")

            _s_col5, _s_col6 = st.columns(2)
            _s_col5.metric("Gross / Shift",  f"${_summary['per_shift_gross']:,.2f}")
            _s_col6.metric("Est. Shifts This Week", _summary["total_shifts"])

            st.caption(
                f"Based on: \\${hourly_rate}/hr × {hours_per_shift} hrs/shift. "
                "Gross income estimate only — does not include tax or HELP withholding. "
                "Use **🧾 Payslip Calibration** to see estimated take-home."
            )

            # Shift date list
            with st.expander("Selected shift dates"):
                for _sd in _selected_dates:
                    st.write(f"- {_sd.strftime('%A %-d %B %Y')}")

            st.divider()

            # --- Save button ---
            if _existing:
                st.caption(
                    f"⚠️ A roster for this week is already saved "
                    f"(saved {_existing['created_at']}). Saving again will overwrite it."
                )

            if st.button("💾 Save Weekly Roster", key="we_save_btn"):
                _iso_dates = [d.isoformat() for d in _selected_dates]
                save_weekly_roster(
                    week_start_date=_we_week_start.isoformat(),
                    week_end_date=_we_week_end.isoformat(),
                    scheduled_dates=_iso_dates,
                    source="manual",
                )
                st.session_state.weekly_entry = {
                    "week_start": _we_week_start.isoformat(),
                    "week_end":   _we_week_end.isoformat(),
                    "dates":      _iso_dates,
                    "summary":    _summary,
                }
                st.success(
                    f"✅ Saved {_summary['total_shifts']} shifts for "
                    f"{week_label(_we_week_start)}."
                )
                st.rerun()

    st.divider()
    st.caption(
        "💡 **Tip:** Come back every Friday when your Amazon roster drops "
        "and tick your shifts for the coming week."
    )


if show_legacy:
 with tab1:

    st.header("Current Roster")

    uploaded_file = st.file_uploader(
        "Upload roster screenshot",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            caption="Uploaded Screenshot"
        )

        try:
            data, ai_response = analyze_roster_image(
                uploaded_file
            )

            if "summary" not in data:
                data["summary"] = ""

            st.session_state.roster_data = data

            st.success("Roster loaded successfully.")

            with st.expander("Developer Debug"):
                st.subheader("Raw AI Response")
                st.code(
                    ai_response,
                    language="json"
                )

                st.subheader("Roster JSON")
                st.json(data)

        except Exception as e:
            st.error("Could not analyze roster image.")
            st.write(e)

    if st.session_state.roster_data:

        roster_data = st.session_state.roster_data

        st.subheader("Human Validation")

        current_days_text = ",".join(
            str(day)
            for day in roster_data["scheduled_days"]
        )

        edited_days = st.text_input(
            "Edit scheduled work days separated by commas",
            value=current_days_text
        )

        if st.button("Update Schedule"):

            corrected_days = [
                int(day.strip())
                for day in edited_days.split(",")
                if day.strip()
            ]

            roster_data["scheduled_days"] = sorted(
                corrected_days
            )

            st.session_state.roster_data = roster_data

            st.success("Schedule updated successfully.")

            st.rerun()

        month_name = roster_data["month"]
        year = int(roster_data["year"])
        scheduled_days = roster_data["scheduled_days"]
        summary = roster_data.get("summary", "")

        current_analytics = calculate_analytics_from_days(
            month_name,
            year,
            scheduled_days,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        work_schedule = current_analytics["work_schedule"]
        total_shifts = current_analytics["total_shifts"]
        weekend_shifts = current_analytics["weekend_shifts"]
        gross_income = current_analytics["gross_income"]
        net_income = current_analytics["net_income"]
        average_weekly_workload = current_analytics["average_weekly_workload"]

        st.subheader("Quick Summary")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Shifts", total_shifts)
        col2.metric("Weekend Shifts", weekend_shifts)
        col3.metric("Avg Weekly Load", average_weekly_workload)
        col4.metric("Gross Income", f"${round(gross_income, 2)}")

        st.subheader("Work Calendar")
        st.markdown(
            render_calendar(month_name, year, scheduled_days),
            unsafe_allow_html=True
        )

        st.subheader("Next Shift")

        next_shift_date = get_next_shift(scheduled_days, month_name, year)

        if next_shift_date:
            days_away = get_days_until(next_shift_date)
            weekday_label = next_shift_date.strftime("%A")
            date_label = next_shift_date.strftime("%Y-%m-%d")

            if days_away == 0:
                countdown = "today"
            elif days_away == 1:
                countdown = "tomorrow"
            else:
                countdown = f"in {days_away} days"

            st.write(f"{weekday_label}, {date_label} ({countdown})")
        else:
            st.write("No upcoming shifts found in this roster.")

        if st.button("Save Corrected Roster to Database"):

            save_roster(
                month_name,
                year,
                scheduled_days,
                summary
            )

            st.success("Corrected roster saved. Old version replaced.")

    else:
        st.info("Upload a roster screenshot to begin.")


with tab_tw:

    st.header("📊 This Week")

    # -----------------------------------------------------------------------
    # Section A — Weekly Dashboard (primary)
    # -----------------------------------------------------------------------
    _tw_df = load_weekly_rosters_dataframe()

    if _tw_df.empty:
        st.info(
            "No weekly roster selected yet. "
            "Go to **🗓 Weekly Entry** to select this week's shifts."
        )
    else:
        import json as _json_tw
        import datetime as _dt_tw

        # Build display labels and sort most-recent first
        _tw_df = _tw_df.sort_values("week_start_date", ascending=False).reset_index(drop=True)

        _tw_labels = []
        for _, _r in _tw_df.iterrows():
            try:
                _ws = _dt_tw.date.fromisoformat(_r["week_start_date"])
                _we = _dt_tw.date.fromisoformat(_r["week_end_date"])
                _tw_labels.append(
                    f"Sun {_ws.strftime('%-d %b')} — Sat {_we.strftime('%-d %b %Y')}"
                )
            except Exception:
                _tw_labels.append(_r["week_start_date"])

        _tw_selected_label = st.selectbox(
            "View week:",
            options=_tw_labels,
            index=0,
            key="tw_week_selector",
        )
        _tw_idx = _tw_labels.index(_tw_selected_label)
        _tw_row = _tw_df.iloc[_tw_idx]

        # Parse saved dates and compute summary
        _tw_dates_iso = _json_tw.loads(_tw_row["scheduled_dates"])
        _tw_dates = [_dt_tw.date.fromisoformat(d) for d in _tw_dates_iso]
        _tw_summary = weekly_summary(_tw_dates, hourly_rate, hours_per_shift, fuel_cost)

        st.caption(f"Source: {_tw_row['source']} · Saved: {_tw_row['created_at']}")
        st.divider()

        # Metrics row 1
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Week Start", _tw_row["week_start_date"])
        _m2.metric("Week End",   _tw_row["week_end_date"])
        _m3.metric("Total Shifts",   _tw_summary["total_shifts"])
        _m4.metric("Weekend Shifts", _tw_summary["weekend_shifts"])

        # Metrics row 2
        _m5, _m6, _m7 = st.columns(3)
        _m5.metric("Gross Income",       f"${_tw_summary['gross_income']:,.2f}")
        _m6.metric("Gross / Shift",      f"${_tw_summary['per_shift_gross']:,.2f}")
        _m7.metric("Total Shifts",       _tw_summary["total_shifts"])

        # Shift list
        if _tw_dates:
            with st.expander("Shift dates this week"):
                for _sd in sorted(_tw_dates):
                    st.write(f"- {_sd.strftime('%A %-d %B %Y')}")

        # Shifts-per-week trend chart (all saved weeks, oldest → newest)
        if len(_tw_df) > 1:
            st.divider()
            st.subheader("Shifts per week — trend")
            _chart_df = _tw_df.copy()
            _chart_df["shifts"] = _chart_df["scheduled_dates"].apply(
                lambda x: len(_json_tw.loads(x))
            )
            _chart_df["label"] = _chart_df["week_start_date"].apply(
                lambda x: _dt_tw.date.fromisoformat(x).strftime("%-d %b")
            )
            _chart_df = _chart_df.sort_values("week_start_date")

            _fig_tw, _ax_tw = plt.subplots(figsize=(10, 3))
            _ax_tw.bar(_chart_df["label"], _chart_df["shifts"], color="#4C9BE8")
            _ax_tw.set_title("Shifts by Week")
            _ax_tw.set_ylabel("Shifts")
            _ax_tw.set_xlabel("Week starting")
            _ax_tw.tick_params(axis="x", rotation=45)
            _fig_tw.tight_layout()
            st.pyplot(_fig_tw)



if show_legacy:
 with tab3:

    st.header("AI Assistant")

    if st.session_state.roster_data:

        # --- Explicitly named current-roster variables ---
        # All Current Roster chat logic must use ONLY these variables.
        # Never use saved roster database records inside this section.
        current_roster_data = st.session_state.roster_data
        current_month_name = current_roster_data["month"]
        current_year = int(current_roster_data["year"])
        current_scheduled_days = list(current_roster_data["scheduled_days"])

        current_analytics = calculate_analytics_from_days(
            current_month_name,
            current_year,
            current_scheduled_days,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        current_work_schedule = current_analytics["work_schedule"]

        current_week_breakdown = build_week_breakdown(
            current_month_name,
            current_year,
            current_scheduled_days,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        # Human-readable context label attached to every answer
        current_roster_label = (
            f"{current_month_name} {current_year} — "
            f"{len(current_scheduled_days)} shift{'s' if len(current_scheduled_days) != 1 else ''}"
        )

        # Build a unique key for the current roster using month, year, and exact days
        current_roster_key = (
            f"{current_month_name}-{current_year}-"
            f"{','.join(str(d) for d in sorted(current_scheduled_days))}"
        )

        # Detect if the loaded roster has changed while chat history exists
        if (
            st.session_state.active_roster_key is not None
            and st.session_state.active_roster_key != current_roster_key
            and st.session_state.roster_chat
        ):
            st.warning(
                "⚠️ The loaded roster has changed. "
                "Earlier chat messages may refer to a different roster."
            )
        st.session_state.active_roster_key = current_roster_key

        st.subheader("Ask About Current Roster")
        st.caption(f"📋 Current loaded roster: {current_roster_label}")

        # Developer Debug expander — shown only when sidebar checkbox is enabled
        if show_debug:
            with st.expander("🛠 Developer Debug", expanded=True):
                st.markdown(f"**Loaded roster:** {current_month_name} {current_year}")
                st.markdown(f"**Shift count:** {len(current_scheduled_days)}")
                st.markdown(f"**current_scheduled_days:** `{current_scheduled_days}`")
                st.markdown("**current_week_breakdown:**")
                st.json(current_week_breakdown)
                wk4 = next((w for w in current_week_breakdown if w["week_number"] == 4), None)
                if wk4:
                    st.markdown("**Week 4 entry:**")
                    st.json(wk4)
                else:
                    st.markdown("**Week 4 entry:** not found")

        # Render chat history
        for msg in st.session_state.roster_chat:
            msg_key = msg.get("roster_key")
            is_stale = (msg_key is not None and msg_key != current_roster_key)
            # Hide stale messages (both user and assistant) when toggle is off
            if is_stale and not show_older_answers:
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if msg_key is None:
                        pass  # old format — render silently
                    elif msg_key == current_roster_key:
                        st.caption(f"✅ Based on: {msg_label if (msg_label := msg.get('roster_label', '')) else msg_key}")
                    else:
                        msg_label = msg.get("roster_label", msg_key)
                        st.caption(f"⚠️ Older answer — based on: {msg_label}")

        user_question = st.chat_input(
            "Ask about the currently loaded roster",
            key="current_roster_chat_input"
        )

        if user_question:

            st.session_state.roster_chat.append(
                {"role": "user", "content": user_question,
                 "roster_key": current_roster_key, "roster_label": current_roster_label}
            )

            # --- Python-first: answer week questions without calling OpenAI ---
            week_num = detect_week_question(user_question)
            if week_num is not None:
                _, mentioned_months = detect_saved_week_question(user_question)

                # Scope check: redirect if user asks about a different month
                if len(mentioned_months) > 1:
                    reply = (
                        f"This chat only uses the currently loaded roster: "
                        f"**{current_roster_label}**. "
                        f"For questions about multiple months, use the **Saved Rosters chat** below."
                    )
                    st.session_state.roster_chat.append({"role": "assistant", "content": reply, "roster_key": current_roster_key, "roster_label": current_roster_label})
                    st.rerun()
                elif len(mentioned_months) == 1 and mentioned_months[0].lower() != current_month_name.lower():
                    reply = (
                        f"This chat only uses the currently loaded roster: "
                        f"**{current_roster_label}**. "
                        f"For {mentioned_months[0]} or other saved months, "
                        f"use the **Saved Rosters chat** below."
                    )
                    st.session_state.roster_chat.append({"role": "assistant", "content": reply, "roster_key": current_roster_key, "roster_label": current_roster_label})
                    st.rerun()

                # No month mentioned, or month matches loaded roster → answer using current_week_breakdown
                week_match = next(
                    (w for w in current_week_breakdown if w["week_number"] == week_num),
                    None
                )
                if week_match:
                    reply = format_week_answer(
                        week_match,
                        include_income=asks_about_income(user_question),
                        roster_label=f"{current_month_name} {current_year}"
                    )
                else:
                    reply = f"I could not find week {week_num} for this roster."
                st.session_state.roster_chat.append({"role": "assistant", "content": reply, "roster_key": current_roster_key, "roster_label": current_roster_label})
                st.rerun()

            # --- Python-first: forecasting questions ---
            forecast_result = detect_forecasting_question(user_question)
            if forecast_result is not None:
                ftype, fvalue = forecast_result
                reply = format_forecasting_answer(
                    ftype, fvalue, hourly_rate, hours_per_shift, fuel_cost, current_roster_label
                )
                if reply:
                    st.session_state.roster_chat.append({"role": "assistant", "content": reply, "roster_key": current_roster_key, "roster_label": current_roster_label})
                    st.rerun()

            # --- Scope check: app/system questions ---
            if detect_app_question(user_question):
                reply = (
                    "This Current Roster chat only answers questions about the loaded roster "
                    f"(**{current_roster_label}**). "
                    "For app behavior, deployment, export, cloud storage, or API questions, "
                    "use **App Help / Diagnostics** (coming soon) or check the sidebar settings."
                )
                st.session_state.roster_chat.append({"role": "assistant", "content": reply, "roster_key": current_roster_key, "roster_label": current_roster_label})
                st.rerun()

            # --- OpenAI for all other questions ---
            system_prompt = f"""
You are a concise scheduling assistant for a single loaded roster month: {current_month_name} {current_year}.

SCOPE: You only know this one month. If asked to compare months, tell the user to use the Saved Rosters chat below.

SOURCE OF TRUTH: The scheduled_days list has been human-validated. If asked about confidence or accuracy, say the source of truth is the human-validated scheduled_days, but note that the original AI image extraction may still contain errors worth reviewing.

WORK SCHEDULE (individual shift details — use for ALL date lookups):
{json.dumps(current_work_schedule)}

MONTHLY ANALYTICS (month totals only — do NOT use for week-level answers):
{json.dumps(current_analytics)}

SETTINGS:
- Hourly rate: {hourly_rate}
- Hours per shift: {hours_per_shift}

FORMATTING RULES:
- Never list shifts as raw day numbers like "17, 18, 19" or "day 17".
- Always format shift dates as: Weekday DD Month YYYY (e.g. Monday 18 May 2026).
- When listing multiple shifts, use bullet points, one per line.
- For count questions: give the count first, then list each shift as a bullet point.
- Format all money with commas and 2 decimal places (e.g. \\$1,088.80).
- Keep answers concise.
"""

            recent_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.roster_chat[-10:]
            ]

            try:
                assistant_response = get_openai_client().chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt}
                    ] + recent_messages
                )
                reply = safe_md(assistant_response.choices[0].message.content)
                st.session_state.roster_chat.append({"role": "assistant", "content": reply, "roster_key": current_roster_key, "roster_label": current_roster_label})
            except Exception as e:
                st.error(f"⚠️ AI Assistant error: {e}")
                st.session_state.roster_chat.pop()

            st.rerun()

        if st.session_state.roster_chat:
            if st.button("Clear Current Roster Chat"):
                st.session_state.roster_chat = []
                st.rerun()

    else:

        st.info(
            "Upload a roster first to ask questions about the current roster."
        )

    st.subheader("Ask About Saved Rosters")

    raw_df = load_rosters_dataframe()

    if raw_df.empty:

        st.write("No saved roster data yet.")

    else:

        saved_df = enrich_saved_rosters(
            raw_df,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        # Render saved rosters chat history
        for msg in st.session_state.saved_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        database_question = st.chat_input(
            "Ask about all saved rosters",
            key="saved_rosters_chat_input"
        )

        if database_question:

            st.session_state.saved_chat.append(
                {"role": "user", "content": database_question}
            )

            # --- Python-first: handle week questions for saved rosters ---
            saved_week_num, saved_months = detect_saved_week_question(database_question)
            if saved_week_num is not None:
                if not saved_months:
                    # No month specified — ask for clarification
                    clarification = (
                        "Which month and year do you mean? "
                        "For example: *'week 4 of April'* or *'April week 4'*."
                    )
                    st.session_state.saved_chat.append(
                        {"role": "assistant", "content": clarification}
                    )
                    st.rerun()

                # Month(s) specified — answer in Python for each
                include_inc = asks_about_income(database_question)
                week_replies = []
                for month in saved_months:
                    # Find matching saved roster(s) for this month
                    matching_rows = [
                        row for _, row in raw_df.iterrows()
                        if row["month"].lower() == month.lower()
                    ]
                    if not matching_rows:
                        week_replies.append(
                            f"I could not find a saved roster for {month.capitalize()}."
                        )
                        continue
                    for row in matching_rows:
                        import json as _json
                        row_days = _json.loads(row["scheduled_days"])
                        row_year = int(row["year"])
                        wb = build_week_breakdown(
                            row["month"], row_year, row_days,
                            hourly_rate, hours_per_shift, fuel_cost
                        )
                        wk = next((w for w in wb if w["week_number"] == saved_week_num), None)
                        if wk:
                            week_replies.append(
                                format_week_answer(
                                    wk,
                                    include_income=include_inc,
                                    roster_label=f"{row['month']} {row_year}"
                                )
                            )
                        else:
                            week_replies.append(
                                f"I could not find week {saved_week_num} "
                                f"for the saved {row['month']} {row_year} roster."
                            )
                combined_reply = "\n\n---\n\n".join(week_replies)
                st.session_state.saved_chat.append(
                    {"role": "assistant", "content": combined_reply}
                )
                st.rerun()

            # --- Python-first: forecasting questions for saved rosters ---
            forecast_result = detect_forecasting_question(database_question)
            if forecast_result is not None:
                ftype, fvalue = forecast_result
                settings_label = (
                    f"Settings used: hourly rate {hourly_rate:.2f}/hr, "
                    f"{hours_per_shift}h/shift"
                )
                reply = format_forecasting_answer(
                    ftype, fvalue, hourly_rate, hours_per_shift, fuel_cost, settings_label
                )
                if reply:
                    st.session_state.saved_chat.append(
                        {"role": "assistant", "content": reply}
                    )
                    st.rerun()

            system_prompt_db = f"""
You are a workforce analytics assistant. Answer clearly using clean Markdown.

Saved roster data:
{json.dumps(saved_df, default=str)}

Current settings:
- Hourly rate: ${hourly_rate}
- Hours per shift: {hours_per_shift}

Formatting rules — always follow these:
- For comparison questions (e.g. "compare April and May"): give a concise summary per month only — total shifts, weekend shifts, gross income, net income, and a partial-roster note if fewer than 10 shifts. Do NOT list individual shift dates unless the user explicitly asks for them.
- Never list shifts as raw day numbers like "17, 18, 19" or "day 17".
- Always format shift dates as: Weekday DD Month YYYY (e.g. Monday 18 May 2026) if dates are requested.
- Use short Markdown headings (###) for each section or month.
- Format all money with commas and 2 decimal places (e.g. $1,088.80).
- When comparing months, give each month its own heading.
- If a month has fewer than 10 shifts, note it may be a partial roster.
- State clearly if data is missing rather than guessing.
- Do not invent missing months or shifts.
- Keep answers concise and practical.
"""

            recent_db_messages = st.session_state.saved_chat[-10:]

            try:
                database_response = get_openai_client().chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt_db}
                    ] + recent_db_messages
                )
                db_reply = safe_md(database_response.choices[0].message.content)
                st.session_state.saved_chat.append(
                    {"role": "assistant", "content": db_reply}
                )
            except Exception as e:
                st.error(f"⚠️ AI Assistant error: {e}")
                st.session_state.saved_chat.pop()

            st.rerun()

        if st.session_state.saved_chat:
            if st.button("Clear Saved Rosters Chat"):
                st.session_state.saved_chat = []
                st.rerun()


if show_legacy:
 with tab4:

    st.header("Saved Rosters")
    st.info(
        "☁️ **Cloud storage notice:** Saved rosters are stored in a local SQLite database. "
        "On Streamlit Cloud, this data may reset when the app is rebooted or redeployed. "
        "Export your data regularly using the Export tab to avoid losing it."
    )
    

    raw_df = load_rosters_dataframe()

    if raw_df.empty:

        st.write("No saved rosters yet.")

    else:

        saved_df = enrich_saved_rosters(
            raw_df,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        saved_df = pd.DataFrame(saved_df)

        for _, row in saved_df.sort_values(
            ["year", "month"],
            ascending=False
        ).iterrows():

            st.markdown(
                f"{row['month']} {row['year']} | "
                f"{row['total_shifts']} shifts | "
                f"{row['weekend_shifts']} weekend shifts | "
                f"Gross: \\${row['gross_income']:,.2f} | "
                f"Saved: {row['created_at']}"
            )

    # --- Danger Zone ---
    st.divider()
    with st.expander("⚠️ Danger Zone — Clear Legacy Monthly Rosters"):
        st.warning(
            "This will permanently delete **all saved monthly rosters** from the database. "
            "Weekly Entry data and app settings will **not** be affected. "
            "This cannot be undone."
        )
        _confirm_clear = st.checkbox(
            "Yes, I want to permanently delete all saved monthly rosters",
            key="confirm_clear_monthly",
        )
        _clear_btn = st.button(
            "🗑 Delete All Monthly Rosters",
            key="clear_monthly_btn",
            disabled=not _confirm_clear,
        )
        if _clear_btn and _confirm_clear:
            clear_monthly_rosters()
            st.success("✅ All monthly rosters have been cleared.")
            st.rerun()

if show_legacy:
 with tab5:

    st.header("Forecasting & Goal Planning")

    st.subheader("Per-Shift Breakdown")

    gps = gross_per_shift(hourly_rate, hours_per_shift)

    col1, col2 = st.columns(2)
    col1.metric("Gross Per Shift", f"${gps:,.2f}")
    col2.metric("Hours Per Shift", hours_per_shift)

    st.divider()

    st.subheader("Project My Income")

    shift_count = st.number_input(
        "How many shifts?",
        min_value=1,
        max_value=31,
        value=10,
        step=1
    )

    projection = project_income(shift_count, hourly_rate, hours_per_shift, fuel_cost)

    st.metric("Projected Gross Income", f"${projection['gross']:,.2f}")

    st.divider()

    st.subheader("Shifts Needed for a Target Gross Income")

    target = st.number_input(
        "Target gross income ($)",
        min_value=0.0,
        value=4000.0,
        step=100.0
    )

    needed = shifts_needed_for_target(target, hourly_rate, hours_per_shift, fuel_cost)

    if needed is None:
        st.warning("Gross income per shift is zero — check your hourly rate and hours per shift settings.")
    else:
        st.metric("Shifts Needed", needed)
        st.caption(
            f"At \\${gps:,.2f} gross per shift, "
            f"{needed} shifts earns \\${needed * gps:,.2f} gross."
        )

    st.divider()

    st.subheader("Cost of Skipping One Shift")

    skip = cost_of_skipping_shift(hourly_rate, hours_per_shift, fuel_cost)

    st.metric("Gross Income Lost", f"${skip['lost_gross']:,.2f}")

    st.divider()

    st.subheader("Monthly Income Projection")

    raw_df = load_rosters_dataframe()

    if raw_df.empty:
        st.info("No saved rosters yet. Save at least one full roster to see projections.")
    else:
        saved_records = enrich_saved_rosters(
            raw_df,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        monthly_proj = projected_monthly_income(
            saved_records,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        if monthly_proj is None:
            st.info("No full rosters found yet (need at least one month with 10+ shifts).")
        else:
            st.caption(
                f"Based on {monthly_proj['based_on_months']} full roster(s), "
                f"averaging {monthly_proj['avg_shifts']} shifts/month."
            )

            col1, col2 = st.columns(2)
            col1.metric("Projected Monthly Gross", f"${monthly_proj['gross']:,.2f}")
            col2.metric("Projected Monthly Net", f"${monthly_proj['net']:,.2f}")

            partial = [r for r in saved_records if r["total_shifts"] < 10]
            if partial:
                names = ", ".join(f"{r['month']} {r['year']}" for r in partial)
                st.caption(f"⚠️ Excluded as partial: {names}")


if show_legacy:
 with tab6:

    st.header("Export Roster Data")

    raw_df = load_rosters_dataframe()

    if raw_df.empty:
        st.info(
            "No saved rosters yet. "
            "Upload and save a roster first before exporting."
        )
    else:
        saved_records = enrich_saved_rosters(
            raw_df,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        export_df = build_export_dataframe(saved_records)

        st.subheader("Preview")
        st.dataframe(export_df, use_container_width=True, height=140)

        csv_data = export_df.to_csv(index=False).encode("utf-8")
        filename = f"roster_export_{get_today()}.csv"

        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )

        st.caption(
            f"Export includes {len(export_df)} roster(s). "
            f"Settings used — hourly rate: {hourly_rate}, "
            f"hours per shift: {hours_per_shift}."
        )

if show_legacy:
 with tab7:

    st.header("💰 Casual Pay Estimator")
    st.caption(
        "Estimate your gross income based on your actual pay rate, shift hours, "
        "day-type multipliers, and allowances. **Rates are estimates.** "
        "Check Fair Work, your award, enterprise agreement, or payslip for exact rates."
    )

    st.divider()
    st.subheader("Step 1 — Pay Rate Settings")
    st.caption(
        "Defaults come from your sidebar settings. "
        "Adjust multipliers to match your actual award, EBA, or employer rate."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        cpe_base_rate = st.number_input(
            "Base hourly rate ($)",
            min_value=0.0,
            value=float(hourly_rate),
            step=0.25,
            key="cpe_base_rate",
            help="Your ordinary-time hourly rate. Defaults to your sidebar setting."
        )
        cpe_hours = st.number_input(
            "Hours per shift",
            min_value=0.0,
            value=float(hours_per_shift),
            step=0.25,
            key="cpe_hours",
            help="Paid hours per shift. Defaults to your sidebar setting."
        )
        # Pass A: fuel removed from UI. Fixed at 0.0.
        cpe_fuel = 0.0
        cpe_allowance = st.number_input(
            "Allowance per shift ($ optional)",
            min_value=0.0,
            value=0.0,
            step=0.50,
            key="cpe_allowance",
            help="Any flat allowance paid per shift (e.g. tool allowance, meal allowance). Leave at 0 if none."
        )

    with col_b:
        st.markdown("**Day-type multipliers**")
        st.caption(
            "Set to 1.0 if you are unsure. "
            "Do not apply penalty rates unless your award, EBA, or payslip confirms them."
        )
        cpe_weekday_mult = st.number_input(
            "Weekday multiplier",
            min_value=0.0,
            value=1.0,
            step=0.05,
            key="cpe_weekday_mult"
        )
        cpe_sat_mult = st.number_input(
            "Saturday multiplier",
            min_value=0.0,
            value=1.0,
            step=0.05,
            key="cpe_sat_mult"
        )
        cpe_sun_mult = st.number_input(
            "Sunday multiplier",
            min_value=0.0,
            value=1.0,
            step=0.05,
            key="cpe_sun_mult"
        )
        cpe_ph_mult = st.number_input(
            "Public holiday multiplier",
            min_value=0.0,
            value=2.5,
            step=0.05,
            key="cpe_ph_mult",
            help="Only applied to shifts you manually mark as public holidays below."
        )
        cpe_ot_mult = st.number_input(
            "Overtime multiplier (reference only)",
            min_value=0.0,
            value=1.5,
            step=0.05,
            key="cpe_ot_mult",
            help="Not applied automatically — shown in the comparison table for reference."
        )

    st.divider()
    st.subheader("Step 2 — Per-Shift Estimates")

    wd_gross = shift_gross(cpe_base_rate, cpe_hours, cpe_weekday_mult, cpe_allowance)
    wd_af    = shift_after_fuel(cpe_base_rate, cpe_hours, cpe_weekday_mult, cpe_allowance, cpe_fuel)
    sat_gross = shift_gross(cpe_base_rate, cpe_hours, cpe_sat_mult, cpe_allowance)
    sat_af    = shift_after_fuel(cpe_base_rate, cpe_hours, cpe_sat_mult, cpe_allowance, cpe_fuel)
    sun_gross = shift_gross(cpe_base_rate, cpe_hours, cpe_sun_mult, cpe_allowance)
    sun_af    = shift_after_fuel(cpe_base_rate, cpe_hours, cpe_sun_mult, cpe_allowance, cpe_fuel)
    ph_gross  = shift_gross(cpe_base_rate, cpe_hours, cpe_ph_mult, cpe_allowance)
    ph_af     = shift_after_fuel(cpe_base_rate, cpe_hours, cpe_ph_mult, cpe_allowance, cpe_fuel)
    ot_gross  = shift_gross(cpe_base_rate, cpe_hours, cpe_ot_mult, cpe_allowance)
    ot_af     = shift_after_fuel(cpe_base_rate, cpe_hours, cpe_ot_mult, cpe_allowance, cpe_fuel)

    cpe_table_rows = compare_shift_types(
        cpe_base_rate, cpe_hours, cpe_fuel,
        weekday_mult=cpe_weekday_mult,
        saturday_mult=cpe_sat_mult,
        sunday_mult=cpe_sun_mult,
        ph_mult=cpe_ph_mult,
        allowance=cpe_allowance,
    )
    # Append overtime row for reference
    cpe_table_rows.append({
        "Shift Type": "Overtime (ref)",
        "Multiplier": f"{cpe_ot_mult:.2f}x",
        "Effective Rate ($/hr)": f"${cpe_base_rate * cpe_ot_mult:.2f}",
        "Gross / Shift": f"${ot_gross:,.2f}",
    })
    # Drop the "After Fuel / Shift" column — fuel removed in Pass A
    for _r in cpe_table_rows:
        _r.pop("After Fuel / Shift", None)

    st.dataframe(pd.DataFrame(cpe_table_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Step 3 — Roster Summary")

    if st.session_state.roster_data:
        _rd = st.session_state.roster_data
        _month = _rd["month"]
        _year  = int(_rd["year"])
        _days  = list(_rd["scheduled_days"])

        _classified = classify_days(_days, _month, _year)
        _wd_days  = _classified["weekday"]
        _sat_days = _classified["saturday"]
        _sun_days = _classified["sunday"]

        _wd_count  = len(_wd_days)
        _sat_count = len(_sat_days)
        _sun_count = len(_sun_days)

        st.caption(
            f"Loaded roster: **{_month} {_year}** — "
            f"{len(_days)} total shifts: "
            f"{_wd_count} weekday, {_sat_count} Saturday, {_sun_count} Sunday"
        )
        if _sat_days:
            st.caption(f"Saturday shifts: days {sorted(_sat_days)}")
        if _sun_days:
            st.caption(f"Sunday shifts: days {sorted(_sun_days)}")

        cpe_ph_count = st.number_input(
            "Public holiday shifts in this roster (manual)",
            min_value=0,
            max_value=len(_days),
            value=0,
            step=1,
            key="cpe_ph_count",
            help="Public holidays are not auto-detected. Enter how many of your shifts fall on a public holiday."
        )

        # Adjust weekday count down by PH count (user confirms PH shifts were weekdays)
        _wd_adj = max(0, _wd_count - cpe_ph_count)

        _r_gross = roster_gross(
            _wd_adj, _sat_count, _sun_count, cpe_ph_count,
            cpe_base_rate, cpe_hours,
            weekday_mult=cpe_weekday_mult,
            saturday_mult=cpe_sat_mult,
            sunday_mult=cpe_sun_mult,
            ph_mult=cpe_ph_mult,
            allowance=cpe_allowance,
        )
        _r_af = roster_after_fuel(
            _wd_adj, _sat_count, _sun_count, cpe_ph_count,
            cpe_base_rate, cpe_hours, cpe_fuel,
            weekday_mult=cpe_weekday_mult,
            saturday_mult=cpe_sat_mult,
            sunday_mult=cpe_sun_mult,
            ph_mult=cpe_ph_mult,
            allowance=cpe_allowance,
        )
        _avg_per_shift = round(_r_af / len(_days), 2) if len(_days) > 0 else 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Gross", f"${_r_gross:,.2f}")
        col2.metric("Estimated Gross", f"${_r_af:,.2f}")
        col3.metric("Gross / Shift", f"${_avg_per_shift:,.2f}")

    else:
        st.info("No roster loaded. Load a roster in the Current Roster tab to see a roster summary here.")
        cpe_manual_shifts = st.number_input(
            "Number of shifts (manual estimate)",
            min_value=1, max_value=31, value=10, step=1, key="cpe_manual_shifts"
        )
        cpe_ph_count = st.number_input(
            "Public holiday shifts (manual)",
            min_value=0, max_value=cpe_manual_shifts, value=0, step=1, key="cpe_ph_count_manual"
        )
        _wd_adj  = max(0, cpe_manual_shifts - cpe_ph_count)
        _r_gross = roster_gross(
            _wd_adj, 0, 0, cpe_ph_count,
            cpe_base_rate, cpe_hours,
            weekday_mult=cpe_weekday_mult,
            ph_mult=cpe_ph_mult,
            allowance=cpe_allowance,
        )
        _r_af = roster_after_fuel(
            _wd_adj, 0, 0, cpe_ph_count,
            cpe_base_rate, cpe_hours, cpe_fuel,
            weekday_mult=cpe_weekday_mult,
            ph_mult=cpe_ph_mult,
            allowance=cpe_allowance,
        )
        _avg_per_shift = round(_r_af / cpe_manual_shifts, 2) if cpe_manual_shifts > 0 else 0.0
        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Gross", f"${_r_gross:,.2f}")
        col2.metric("Estimated Gross", f"${_r_af:,.2f}")
        col3.metric("Gross / Shift", f"${_avg_per_shift:,.2f}")

    st.divider()
    st.subheader("Step 4 — Skip Cost")
    st.caption("What you lose (in gross) if you skip one shift of a chosen type.")

    cpe_skip_type = st.selectbox(
        "Shift type to skip",
        options=["Weekday", "Saturday", "Sunday", "Public Holiday"],
        key="cpe_skip_type"
    )
    _skip_mult_map = {
        "Weekday": cpe_weekday_mult,
        "Saturday": cpe_sat_mult,
        "Sunday": cpe_sun_mult,
        "Public Holiday": cpe_ph_mult,
    }
    _skip_result = skip_cost(
        cpe_base_rate, cpe_hours,
        multiplier=_skip_mult_map[cpe_skip_type],
        fuel_cost=cpe_fuel,
        allowance=cpe_allowance,
    )

    st.metric("Gross Income Lost", f"${_skip_result['gross_lost']:,.2f}")

    st.divider()
    st.warning(
        "⚠️ **Estimates only.** Figures are gross income based on entered rate and hours. "
        "They do not include tax, HELP, super, deductions, or award penalty rates. "
        "Check your payslip, Fair Work, your award/enterprise agreement, "
        "or a registered tax agent for exact figures."
    )


# ---------------------------------------------------------------------------
# Tab: Payslip Calibration (main)
# ---------------------------------------------------------------------------
with tab_pc:
    import datetime as _dt_ps
    import pandas as _pd_ps

    st.header("🧾 Payslip Calibration")
    st.caption(
        "Enter your payslip line items manually to break down earnings, tax, and HELP. "
        "Use this to understand your effective rates and flag previous-period adjustments."
    )
    st.info("📋 **Manual entry only.** Payslip image/OCR upload is not available yet.")

    # -----------------------------------------------------------------------
    # Section 1 — Pay Period
    # -----------------------------------------------------------------------
    st.subheader("Step 1 — Pay Period")

    _ps_col1, _ps_col2 = st.columns(2)
    with _ps_col1:
        _ps_week_start = st.date_input(
            "Pay week start (Sunday)",
            value=_dt_ps.date.today() - _dt_ps.timedelta(days=_dt_ps.date.today().weekday() + 1),
            key="ps_week_start",
        )
    with _ps_col2:
        _ps_week_end = st.date_input(
            "Pay week end (Saturday)",
            value=_ps_week_start + _dt_ps.timedelta(days=6),
            key="ps_week_end",
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2 — Line Item Entry
    # -----------------------------------------------------------------------
    st.subheader("Step 2 — Payslip Line Items")
    st.caption(
        "Add one row per line on your payslip. "
        "Set the **Category** for each line. "
        "Use **previous period adjustment** for any lines that belong to an earlier pay week."
    )

    # Seed with Amazon example structure if empty
    _default_lines = [
        {
            "description": "Night Shift",
            "hours": 0.0,
            "rate": 0.0,
            "amount": 0.0,
            "category": "current period earning",
            "applicable_period_ending": "",
            "adjustment_note": "",
        },
    ]

    if not st.session_state.payslip_lines:
        st.session_state.payslip_lines = _default_lines.copy()

    # Build editable dataframe
    _lines_df = _pd_ps.DataFrame(st.session_state.payslip_lines)

    # Ensure all expected columns exist (guard for older session states)
    for _col, _default in [
        ("description", ""),
        ("hours", 0.0),
        ("rate", 0.0),
        ("amount", 0.0),
        ("category", "current period earning"),
        ("applicable_period_ending", ""),
        ("adjustment_note", ""),
    ]:
        if _col not in _lines_df.columns:
            _lines_df[_col] = _default

    _edited_df = st.data_editor(
        _lines_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "description": st.column_config.TextColumn("Description", width="large"),
            "hours": st.column_config.NumberColumn("Hours", min_value=0.0, step=0.001, format="%.4f"),
            "rate": st.column_config.NumberColumn("Rate ($/hr)", min_value=0.0, step=0.01, format="%.4f"),
            "amount": st.column_config.NumberColumn("Amount ($)", min_value=0.0, step=0.01, format="%.2f"),
            "category": st.column_config.SelectboxColumn(
                "Category",
                options=CATEGORIES,
                required=True,
            ),
            "applicable_period_ending": st.column_config.TextColumn(
                "Period Ending (YYYY-MM-DD)",
                help="Fill in only for previous-period adjustment lines, e.g. 2026-05-17",
                width="medium",
            ),
            "adjustment_note": st.column_config.TextColumn(
                "Adjustment Note",
                help="Optional note for adjustment lines, e.g. 'Double Time Super Applies'",
                width="large",
            ),
        },
        key="ps_line_editor",
    )

    _ps_col_a, _ps_col_b = st.columns([1, 4])
    with _ps_col_a:
        if st.button("💾 Save Line Items", key="ps_save_lines"):
            st.session_state.payslip_lines = _edited_df.to_dict("records")
            st.success("Line items saved.")
            st.rerun()
    with _ps_col_b:
        if st.button("🗑 Clear All Lines", key="ps_clear_lines"):
            st.session_state.payslip_lines = _default_lines.copy()
            st.rerun()

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3 — Payslip Summary
    # -----------------------------------------------------------------------
    st.subheader("Step 3 — Payslip Summary")

    _items = st.session_state.payslip_lines

    # Entered net pay for reconciliation
    _ps_entered_net = st.number_input(
        "Entered Net Pay (from payslip, $)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.2f",
        key="ps_entered_net",
        help="Enter the net pay figure shown on your payslip. Used for reconciliation only.",
    )

    if st.button("📊 Calculate", key="ps_calculate"):

        # Store entered net pay so Weekly Assistant can use the effective net rate
        st.session_state.wa_entered_net = _ps_entered_net

        # Derive totals
        _cp_gross  = current_period_gross(_items)
        _adj_gross = adjustment_gross(_items)
        _tot_gross = pc_total_gross(_items)
        _marg_tax  = marginal_tax_withheld(_items)
        _help      = help_withheld(_items)
        _tot_with  = total_withheld(_items)
        _ded       = total_deductions(_items)
        _allow     = total_allowances(_items)
        _sup       = total_super(_items)
        _calc_net  = calculated_net_pay(_items)
        _rates     = pc_effective_rates(_items, _ps_entered_net)
        _recon     = pc_reconcile(_items, _ps_entered_net)
        _hr        = derived_hourly_rate(_items)

        # Adjustment warning banner
        if has_adjustments(_items):
            _adj_labels = adjustment_period_labels(_items)
            _adj_text = "\n".join(f"- {l}" for l in _adj_labels)
            st.warning(
                "⚠️ **This payslip includes previous-period adjustments.** "
                "Current-period estimates may differ from the total shown on your payslip.\n\n"
                + _adj_text
            )

        # --- Earnings breakdown ---
        st.subheader("Earnings")
        _e1, _e2, _e3 = st.columns(3)
        _e1.metric("Current Period Gross",    f"${_cp_gross:,.2f}")
        _e2.metric("Previous Period Adjustments", f"${_adj_gross:,.2f}")
        _e3.metric("Total Gross Earnings",    f"${_tot_gross:,.2f}")

        # --- Tax & withholding ---
        st.subheader("Tax & Withholding")
        _t1, _t2, _t3 = st.columns(3)
        _t1.metric("Marginal Tax Withheld",   f"${_marg_tax:,.2f}")
        _t2.metric("HELP Withheld",           f"${_help:,.2f}")
        _t3.metric("Total Withheld",          f"${_tot_with:,.2f}")

        # --- Other lines ---
        if _ded > 0 or _allow > 0 or _sup > 0:
            st.subheader("Other")
            _o1, _o2, _o3 = st.columns(3)
            _o1.metric("Deductions",  f"${_ded:,.2f}")
            _o2.metric("Allowances",  f"${_allow:,.2f}")
            _o3.metric("Super",       f"${_sup:,.2f}")

        # --- Net pay ---
        st.subheader("Net Pay")
        _n1, _n2, _n3 = st.columns(3)
        _n1.metric("Calculated Net Pay",  f"${_calc_net:,.2f}")
        _n2.metric("Entered Net Pay",     f"${_ps_entered_net:,.2f}")
        _n3.metric("Difference",          f"${_recon['difference']:,.2f}")

        if not _recon["is_ok"]:
            st.warning(
                f"⚠️ **Reconciliation mismatch:** Calculated net pay is **${_calc_net:,.2f}** "
                f"but your entered net pay is **${_ps_entered_net:,.2f}** "
                f"(difference: ${_recon['difference']:,.2f}). "
                "This may be due to unlisted deductions, rounding, or items not entered above. "
                "Check your payslip carefully."
            )
        else:
            st.success(
                f"✅ Net pay reconciles within tolerance "
                f"(calculated ${_calc_net:,.2f} vs entered ${_ps_entered_net:,.2f})."
            )

        # --- Effective rates ---
        st.subheader("Effective Rates")
        st.caption("Rates are based on total gross earnings including any adjustments.")

        _r1, _r2, _r3, _r4 = st.columns(4)
        _r1.metric(
            "Effective Tax Rate",
            f"{_rates['effective_tax_rate'] * 100:.2f}%" if _rates["effective_tax_rate"] is not None else "N/A",
        )
        _r2.metric(
            "Effective HELP Rate",
            f"{_rates['effective_help_rate'] * 100:.2f}%" if _rates["effective_help_rate"] is not None else "N/A",
        )
        _r3.metric(
            "Combined Withholding Rate",
            f"{_rates['effective_combined_rate'] * 100:.2f}%" if _rates["effective_combined_rate"] is not None else "N/A",
        )
        _r4.metric(
            "Effective Net Rate",
            f"{_rates['effective_net_rate'] * 100:.2f}%" if _rates["effective_net_rate"] is not None else "N/A",
        )

        # --- Derived hourly rate ---
        if _hr is not None:
            st.divider()
            st.subheader("Derived Hourly Rate")
            st.caption(
                "Calculated from current-period earning lines only (hours × rate = amount). "
                "Excludes previous-period adjustments."
            )
            st.metric("Effective Hourly Rate (current period)", f"${_hr:,.4f}/hr")
            st.caption(
                f"This compares to your sidebar setting of **${hourly_rate}/hr**. "
                "You can update the sidebar if your rate has changed."
            )

        st.divider()
        st.warning(
            "⚠️ **This is an estimate only and not tax advice.** "
            "Actual tax, HELP, and net pay depend on your full-year income, "
            "ATO assessments, and individual circumstances. "
            "Consult a registered tax agent for personalised advice."
        )
