# Payslip Projector

A personal Streamlit app for Amazon casual warehouse workers to project their weekly pay estimate from a shift roster. Enter your shifts, see estimated gross pay, income tax, student loan / HELP repayment, and take-home — with a downloadable projected payslip in Excel.

> **This app produces estimates only. It is not an official payslip and does not constitute tax advice.**

---

## What It Does

- Enter shifts for the week using plain English: `Mon Tue Fri`, `Monday to Friday`, `No shifts this week`
- Instantly see estimated gross pay, income tax, HELP repayment, total withheld, and take-home
- Ask natural-language questions: *"What if I skip Tuesday?"*, *"How much will I take home?"*, *"What's my projected payslip?"*
- Download a formatted Projected Payslip Estimate as an Excel (.xlsx) file
- Save weekly rosters to a local SQLite database
- All pay calculations are deterministic Python — OpenAI is only used for open-ended questions not covered by the built-in handlers

---

## How to Run Locally

```bash
# 1. Clone or download the project
cd ai_assistant_project

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env            # if .env.example exists, otherwise create .env manually
# Add your OpenAI API key (see Required Environment Variables below)

# 5. Run the app (employee mode — default)
streamlit run app.py

# 6. Run in admin mode
PAYSLIP_ADMIN=1 streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

---

## Required Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

OpenAI is used only as a fallback for questions that are not handled deterministically. Most pay-related questions are answered by Python without calling OpenAI at all.

> Never commit `.env` to version control. It is already excluded in `.gitignore`.

---

## Employee Mode vs Admin Mode

### Employee Mode (default)

Run with `streamlit run app.py`.

- Single tab: **💰 Payslip Projector**
- No editable rate inputs — pay rates load automatically from saved settings
- No developer debug controls
- No legacy tools
- Clean, uncluttered interface for daily use

### Admin Mode

Run with `PAYSLIP_ADMIN=1 streamlit run app.py`.

Adds:
- Editable **Hourly Rate** and **Hours Per Shift** inputs in the sidebar
- **Save Settings** button
- **Show Advanced / Legacy Tools** toggle (historical roster tabs, payslip calibration, etc.)
- **Show Developer Debug** toggle
- **📊 This Week** tab

Admin mode is intended for setup, configuration, and testing — not for everyday employee use.

---

## PAYSLIP_ADMIN=1

Set this environment variable to `1` to enable admin controls:

```bash
# Mac/Linux — inline
PAYSLIP_ADMIN=1 streamlit run app.py

# Or export it for the session
export PAYSLIP_ADMIN=1
streamlit run app.py

# Windows (Command Prompt)
set PAYSLIP_ADMIN=1
streamlit run app.py
```

Unset it (or set to `0`) to return to employee mode.

---

## Disclaimer

The pay figures produced by this app are **estimates only**, calculated from your entered shift count and the withholding rates derived from a previous payslip.

- This is **not an official payslip**
- Actual pay may vary due to overtime, allowances, adjustments, tax bracket changes, HELP repayment thresholds, and employer payroll rules
- This app does **not** constitute tax advice
- Always refer to your official payslip from your employer for accurate figures

---

## Project Structure

```
app.py                      — Main Streamlit app
database.py                 — SQLite functions (init, load, save rosters & settings)
week_entry_utils.py         — Shift parsing, week navigation, summary calculations
pay_rate_utils.py           — Shift gross, net, skip cost calculations
forecasting.py              — Income projection and target planning
payslip_calibration_utils.py— Payslip line-item processing (legacy / admin)
analytics.py                — Historical roster analytics
ai_helpers.py               — OpenAI image analysis (legacy roster upload)
time_utils.py               — Date helpers
calendar_view.py            — HTML calendar renderer
export_utils.py             — CSV export builder
requirements.txt            — Python dependencies
rosters.db                  — Local SQLite database (auto-created, not committed)
.streamlit/config.toml      — Streamlit config (toolbar mode, usage stats)
backups/                    — Manual module backups
.env                        — API keys (not committed)
```

---

## Beta Testing Checklist

Quick test to verify the app is working correctly:

1. **Open the app** — confirm you see "💰 Payslip Projector" with a single tab and no developer controls
2. **Enter shifts** — type `Mon Tue Fri` in the shift input box and press Enter
   - Confirm 3 shifts, correct gross pay, income tax, HELP, and take-home are shown
   - Confirm the breakdown table adds up: tax + HELP = Total withheld; gross − withheld = take-home
3. **Ask a chat question** — type *"How much will I take home?"*
   - Confirm the answer matches the take-home shown in the summary card
4. **Try another question** — type *"What if I add Saturday?"*
   - Confirm shifts increase to 4 and pay updates accordingly
5. **Open Projected Payslip Estimate** — expand the "📄 Projected Payslip Estimate" section
   - Confirm numbers match the summary card
   - Download the Excel file and open it
   - Confirm it says "PROJECTED PAYSLIP ESTIMATE" (not "Official Payslip")
   - Confirm the disclaimer is present at the bottom
6. **Test admin mode** — restart with `PAYSLIP_ADMIN=1 streamlit run app.py`
   - Confirm sidebar shows editable Hourly Rate, Hours Per Shift, and Save Settings
   - Confirm "📊 This Week" tab appears
   - Confirm advanced tools toggle works

---

## Deployment Notes

When deploying to Streamlit Cloud:

- Set `OPENAI_API_KEY` in **Streamlit Cloud Secrets** (not in `.env`)
- `rosters.db` is auto-created by `init_db()` on first run — ensure the deployment environment has write access to the working directory, or replace with a cloud-compatible database (e.g. Supabase)
- Do not upload `.env`, `venv/`, or `Images/` to the repository
- `.streamlit/config.toml` should be committed — it controls the toolbar appearance
