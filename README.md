# AI Work Assistant — Roster Intelligence App

A personal Streamlit app that reads work roster screenshots using OpenAI Vision, extracts scheduled days, and provides analytics, forecasting, and export tools.

---

## What It Does

- Upload a roster screenshot and let AI extract your scheduled work days
- Validate and correct the extracted days manually
- View a visual calendar with highlighted work days
- Track income across months with live recalculation based on your settings
- Ask the AI assistant questions about your current or historical rosters
- Forecast income targets and shift costs
- Export saved roster data as CSV

---

## How to Run Locally

```bash
# 1. Clone or download the project
cd ai_assistant_project

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file
cp .env.example .env
# Add your OpenAI API key to .env

# 5. Run the app
streamlit run app.py
```

---

## Required Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

> Never commit `.env` to version control. It is already excluded in `.gitignore`.

---

## Current Features

| Tab | Description |
|---|---|
| 📅 Current Roster | Upload screenshot, AI extracts days, validate manually, view calendar |
| 📊 Analytics | Historical charts — shifts, gross income, net income by month |
| 🤖 AI Assistant | Ask questions about current or saved rosters |
| 📚 History | View all saved rosters with income summary |
| 📈 Forecasting | Project income, calculate shifts needed for a target, skip-shift cost |
| 📥 Export | Download saved roster data as CSV |

---

## Project Structure

```
app.py              — Main Streamlit app (modular entry point)
database.py         — SQLite functions (init, load, save)
analytics.py        — Schedule and income calculations
ai_helpers.py       — OpenAI image analysis logic
time_utils.py       — Date helpers (today, week ranges, next shift)
calendar_view.py    — HTML calendar grid renderer
forecasting.py      — Income projection and goal planning
export_utils.py     — CSV export builder
requirements.txt    — Python dependencies
rosters.db          — Local SQLite database (not committed)
backups/            — Manual backups of all modules
Images/             — Roster screenshots (not committed)
.env                — API keys (not committed)
```

---

## Settings

Configured via the sidebar in the app:

- **Hourly Rate** — your pay rate per hour
- **Hours Per Shift** — typical shift length
- **Fuel Cost Per Shift** — travel cost deducted per shift worked

All income figures are recalculated live from these settings. The database stores only raw shift facts (month, year, scheduled days).

---

## Deployment Notes

> The app is not yet deployed. When deploying to Streamlit Cloud:
> - Set `OPENAI_API_KEY` in Streamlit Cloud secrets
> - Replace `rosters.db` with a cloud-compatible database (e.g. Supabase, PlanetScale)
> - Do not upload `.env`, `venv/`, or `Images/` to the repository
