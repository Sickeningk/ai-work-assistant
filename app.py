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
)

from analytics import (
    calculate_analytics_from_days,
    enrich_saved_rosters
)

from ai_helpers import analyze_roster_image
from time_utils import get_next_shift, get_days_until, get_today, build_week_breakdown
from calendar_view import render_calendar
from export_utils import build_export_dataframe
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
    nps = net_per_shift(hourly_rate, hours_per_shift, fuel_cost)

    if question_type == "shifts_for_target":
        target = value
        needed = shifts_needed_for_target(target, hourly_rate, hours_per_shift, fuel_cost)
        if needed is None:
            return (
                "⚠️ Your net per shift is zero or negative — check your settings. "
                "Cannot calculate shifts needed."
            )
        proj = project_income(needed, hourly_rate, hours_per_shift, fuel_cost)
        lines = [
            f"To reach **\\${target:,.2f} net**, you need **{needed} shifts**.",
            "",
            f"- Net per shift: \\${nps:,.2f}",
            f"- {needed} shifts gross: \\${proj['gross']:,.2f}",
            f"- {needed} shifts net: \\${proj['net']:,.2f}",
        ]
        return "\n".join(lines)

    if question_type == "project_shifts":
        n = int(value)
        proj = project_income(n, hourly_rate, hours_per_shift, fuel_cost)
        lines = [
            f"If you work **{n} shift{'s' if n != 1 else ''}**:",
            "",
            f"- Gross income: \\${proj['gross']:,.2f}",
            f"- Net income: \\${proj['net']:,.2f}",
            f"- Net per shift: \\${nps:,.2f}",
        ]
        return "\n".join(lines)

    if question_type == "skip_cost":
        skip = cost_of_skipping_shift(hourly_rate, hours_per_shift, fuel_cost)
        lines = [
            "If you skip one shift:",
            "",
            f"- Lost gross income: \\${skip['lost_gross']:,.2f}",
            f"- Fuel saved: \\${skip['saved_fuel']:,.2f}",
            f"- **Net income lost: \\${skip['lost_net']:,.2f}**",
        ]
        return "\n".join(lines)

    if question_type == "net_per_shift":
        lines = [
            "Per shift breakdown:",
            "",
            f"- Gross per shift: \\${gps:,.2f}",
            f"- Fuel cost: \\${fuel_cost:,.2f}",
            f"- **Net per shift: \\${nps:,.2f}**",
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
            lines.append(f"- Net income: \\${week_data['net_income']:,.2f}")
    return "\n".join(lines)


init_db()

saved_hourly_rate, saved_hours_per_shift, saved_fuel_cost = load_settings()

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

fuel_cost = st.sidebar.number_input(
    "Fuel Cost Per Shift ($)",
    min_value=0.0,
    value=float(saved_fuel_cost),
    step=5.0
)

if st.sidebar.button("Save Settings"):
    save_settings(
        hourly_rate,
        hours_per_shift,
        fuel_cost
    )

    st.sidebar.success("Settings saved.")

st.sidebar.write("Current Settings")
st.sidebar.write(f"Hourly Rate: ${hourly_rate}")
st.sidebar.write(f"Hours Per Shift: {hours_per_shift}")
st.sidebar.write(f"Fuel Cost Per Shift: ${fuel_cost}")

st.sidebar.divider()
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

tab_we, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗓 Weekly Entry",
    "📅 Current Roster",
    "📊 Analytics",
    "🤖 AI Assistant",
    "📚 History",
    "📈 Forecasting",
    "📥 Export",
    "💰 Casual Pay Estimator"
])


# ---------------------------------------------------------------------------
# Tab: Weekly Entry
# ---------------------------------------------------------------------------
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

            _s_col1, _s_col2, _s_col3, _s_col4 = st.columns(4)
            _s_col1.metric("Total Shifts",    _summary["total_shifts"])
            _s_col2.metric("Weekend Shifts",  _summary["weekend_shifts"])
            _s_col3.metric("Gross Income",    f"${_summary['gross_income']:,.2f}")
            _s_col4.metric("After-Fuel",      f"${_summary['after_fuel_income']:,.2f}")

            _s_col5, _s_col6 = st.columns(2)
            _s_col5.metric("Gross / Shift",     f"${_summary['per_shift_gross']:,.2f}")
            _s_col6.metric("After-Fuel / Shift", f"${_summary['per_shift_after_fuel']:,.2f}")

            st.caption(
                f"Based on: \\${hourly_rate}/hr × {hours_per_shift} hrs/shift, "
                f"\\${fuel_cost} fuel/shift. "
                "This is an estimate only — rates may vary by day type."
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
        col5.metric("Net Income", f"${round(net_income, 2)}")

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


with tab2:

    st.header("Analytics Dashboard")

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

        saved_df = pd.DataFrame(saved_df)

        total_rosters = len(saved_df)
        total_shifts_history = saved_df["total_shifts"].sum()
        total_gross_history = saved_df["gross_income"].sum()
        total_net_history = saved_df["net_income"].sum()
        avg_income_history = saved_df["gross_income"].mean()

        best_month_row = saved_df.sort_values(
            "total_shifts",
            ascending=False
        ).iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rosters Saved", total_rosters)
        col2.metric("Total Shifts", int(total_shifts_history))
        col3.metric("Total Gross", f"${round(total_gross_history, 2)}")
        col4.metric("Total Net", f"${round(total_net_history, 2)}")

        st.write(
            f"Average Monthly Income: ${round(avg_income_history, 2)}"
        )

        st.write(
            f"Best Month: {best_month_row['month']} "
            f"{best_month_row['year']} "
            f"({best_month_row['total_shifts']} shifts)"
        )

        st.subheader("Charts & Trends")

        saved_df["label"] = (
            saved_df["month"] +
            " " +
            saved_df["year"].astype(str)
        )

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(
            saved_df["label"],
            saved_df["total_shifts"]
        )
        ax1.set_title("Shifts by Month")
        ax1.set_ylabel("Shifts")
        ax1.set_xlabel("Month")
        ax1.tick_params(axis="x", rotation=45)
        fig1.tight_layout()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(
            saved_df["label"],
            saved_df["gross_income"]
        )
        ax2.set_title("Gross Income by Month")
        ax2.set_ylabel("Income ($)")
        ax2.set_xlabel("Month")
        ax2.tick_params(axis="x", rotation=45)
        fig2.tight_layout()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.bar(
            saved_df["label"],
            saved_df["net_income"]
        )
        ax3.set_title("Net Income by Month")
        ax3.set_ylabel("Income ($)")
        ax3.set_xlabel("Month")
        ax3.tick_params(axis="x", rotation=45)
        fig3.tight_layout()
        st.pyplot(fig3)


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
- Fuel cost per shift: {fuel_cost}

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
                    f"{hours_per_shift}h/shift, fuel {fuel_cost:.2f}/shift"
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
- Fuel cost per shift: ${fuel_cost}

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
                f"Net: \\${row['net_income']:,.2f} | "
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

with tab5:

    st.header("Forecasting & Goal Planning")

    st.subheader("Per-Shift Breakdown")

    gps = gross_per_shift(hourly_rate, hours_per_shift)
    nps = net_per_shift(hourly_rate, hours_per_shift, fuel_cost)

    col1, col2, col3 = st.columns(3)
    col1.metric("Gross Per Shift", f"${gps:,.2f}")
    col2.metric("Fuel Cost Per Shift", f"${fuel_cost:,.2f}")
    col3.metric("Net Per Shift", f"${nps:,.2f}")

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

    col1, col2 = st.columns(2)
    col1.metric("Projected Gross", f"${projection['gross']:,.2f}")
    col2.metric("Projected Net", f"${projection['net']:,.2f}")

    st.divider()

    st.subheader("Shifts Needed for a Target Income")

    target = st.number_input(
        "Target net income ($)",
        min_value=0.0,
        value=4000.0,
        step=100.0
    )

    needed = shifts_needed_for_target(target, hourly_rate, hours_per_shift, fuel_cost)

    if needed is None:
        st.warning("Net income per shift is zero or negative — check your settings.")
    else:
        st.metric("Shifts Needed", needed)
        st.caption(
            f"At ${nps:,.2f} net per shift, "
            f"{needed} shifts earns ${needed * nps:,.2f} net."
        )

    st.divider()

    st.subheader("Cost of Skipping One Shift")

    skip = cost_of_skipping_shift(hourly_rate, hours_per_shift, fuel_cost)

    col1, col2, col3 = st.columns(3)
    col1.metric("Gross Lost", f"${skip['lost_gross']:,.2f}")
    col2.metric("Fuel Saved", f"${skip['saved_fuel']:,.2f}")
    col3.metric("Net Lost", f"${skip['lost_net']:,.2f}")

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
            f"hours per shift: {hours_per_shift}, fuel per shift: {fuel_cost}."
        )

with tab7:

    st.header("💰 Casual Pay Estimator")
    st.caption(
        "Estimate your gross income based on your actual pay rate, shift hours, fuel cost, "
        "and day-type multipliers. **Rates are estimates.** "
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
        cpe_fuel = st.number_input(
            "Fuel cost per shift ($)",
            min_value=0.0,
            value=float(fuel_cost),
            step=0.50,
            key="cpe_fuel",
            help="Out-of-pocket fuel cost per shift. Defaults to your sidebar setting."
        )
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
        "After Fuel / Shift": f"${ot_af:,.2f}",
    })

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
        col2.metric("After-Fuel Gross", f"${_r_af:,.2f}")
        col3.metric("Avg After-Fuel / Shift", f"${_avg_per_shift:,.2f}")

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
        col2.metric("After-Fuel Gross", f"${_r_af:,.2f}")
        col3.metric("Avg After-Fuel / Shift", f"${_avg_per_shift:,.2f}")

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

    col1, col2, col3 = st.columns(3)
    col1.metric("Gross Lost", f"${_skip_result['gross_lost']:,.2f}")
    col2.metric("Fuel Saved", f"${_skip_result['fuel_saved']:,.2f}")
    col3.metric("Net Loss (gross − fuel)", f"${_skip_result['net_loss']:,.2f}")

    st.divider()
    st.warning(
        "⚠️ **Estimate only.** These figures are based on the values you entered above. "
        "They do not account for tax, super, deductions, or award entitlements. "
        "Check Fair Work, your award, enterprise agreement, or payslip for exact rates."
    )
