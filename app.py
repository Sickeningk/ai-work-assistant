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
    load_rosters_dataframe
)

from analytics import (
    calculate_analytics_from_days,
    enrich_saved_rosters
)

from ai_helpers import analyze_roster_image
from time_utils import get_next_shift, get_days_until, get_today
from calendar_view import render_calendar
from export_utils import build_export_dataframe
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

if "roster_data" not in st.session_state:
    st.session_state.roster_data = None

if "roster_chat" not in st.session_state:
    st.session_state.roster_chat = []

if "saved_chat" not in st.session_state:
    st.session_state.saved_chat = []

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📅 Current Roster",
    "📊 Analytics",
    "🤖 AI Assistant",
    "📚 History",
    "📈 Forecasting",
    "📥 Export"
])


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

        roster_data = st.session_state.roster_data

        month_name = roster_data["month"]
        year = int(roster_data["year"])
        scheduled_days = roster_data["scheduled_days"]

        current_analytics = calculate_analytics_from_days(
            month_name,
            year,
            scheduled_days,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        work_schedule = current_analytics["work_schedule"]

        st.subheader("Ask About Current Roster")

        # Render chat history
        for msg in st.session_state.roster_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input(
            "Ask about the currently loaded roster",
            key="current_roster_chat_input"
        )

        if user_question:

            st.session_state.roster_chat.append(
                {"role": "user", "content": user_question}
            )

            system_prompt = f"""
You are a concise scheduling assistant.

Use only the data below. Never refer to shifts as raw day numbers alone.

Work schedule (use this as the source of truth for all dates):
{json.dumps(work_schedule)}

Analytics:
{json.dumps(current_analytics)}

Settings:
- Hourly rate: {hourly_rate}
- Hours per shift: {hours_per_shift}
- Fuel cost per shift: {fuel_cost}

Formatting rules — always follow these:
- Never list a shift as just a number like "day 17" or "17, 18, 19".
- Always format shift dates as: Weekday DD Month YYYY (e.g. Monday 18 May 2026).
- Use the weekday and date from work_schedule for every shift mentioned.
- When listing multiple shifts, use bullet points, one per line.
- For count questions: state the count first, then list each shift date as a bullet point.
- Format all money as $1,234.56 (comma-separated, 2 decimal places).
- Keep answers concise and practical.

Example format for shift lists:
You worked 5 shifts in week 4:
- Sunday 17 May 2026
- Monday 18 May 2026
- Tuesday 19 May 2026
- Wednesday 20 May 2026
- Thursday 21 May 2026
"""

            recent_messages = st.session_state.roster_chat[-10:]

            try:
                assistant_response = get_openai_client().chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt}
                    ] + recent_messages
                )
                reply = assistant_response.choices[0].message.content
                reply = reply.replace("$", "\$")
                st.session_state.roster_chat.append(
                    {"role": "assistant", "content": reply}
                )
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

            system_prompt_db = f"""
You are a workforce analytics assistant. Answer clearly using clean Markdown.

Saved roster data:
{json.dumps(saved_df, default=str)}

Current settings:
- Hourly rate: ${hourly_rate}
- Hours per shift: {hours_per_shift}
- Fuel cost per shift: ${fuel_cost}

Formatting rules — always follow these:
- Never list shifts as raw day numbers like "17, 18, 19" or "day 17".
- Always format shift dates as: Weekday DD Month YYYY (e.g. Monday 18 May 2026).
- To build a full date, combine the day number with the month and year from the roster record.
- When listing multiple shifts, use bullet points, one per line.
- For count questions: state the count first, then list each shift date as a bullet point.
- Use short Markdown headings (###) for each section or month.
- Format all money as $1,234.56 (comma-separated, 2 decimal places).
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
                db_reply = database_response.choices[0].message.content
                db_reply = db_reply.replace("$", "\$")
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
