import sqlite3
import json
from datetime import datetime

DB_NAME = "rosters.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rosters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT,
            year INTEGER,
            scheduled_days TEXT,
            summary TEXT,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY,
            hourly_rate REAL,
            hours_per_shift REAL,
            fuel_cost REAL
        )
    """)

    cursor.execute("""
        INSERT OR IGNORE INTO settings (
            id, hourly_rate, hours_per_shift, fuel_cost
        )
        VALUES (1, 32.0, 8.0, 80.0)
    """)

    # Weekly rosters table — safe to add to an existing DB
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weekly_rosters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start_date TEXT UNIQUE,
            week_end_date TEXT,
            scheduled_dates TEXT,
            source TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def load_settings():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT hourly_rate, hours_per_shift, fuel_cost
        FROM settings
        WHERE id = 1
    """)

    row = cursor.fetchone()
    conn.close()

    return row


def save_settings(hourly_rate, hours_per_shift, fuel_cost):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE settings
        SET hourly_rate = ?,
            hours_per_shift = ?,
            fuel_cost = ?
        WHERE id = 1
    """, (hourly_rate, hours_per_shift, fuel_cost))

    conn.commit()
    conn.close()


def save_roster(month, year, scheduled_days, summary):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM rosters
        WHERE month = ? AND year = ?
    """, (month, year))

    cursor.execute("""
        INSERT INTO rosters (
            month, year, scheduled_days, summary, created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        month,
        year,
        json.dumps(scheduled_days),
        summary,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def save_weekly_roster(week_start_date: str, week_end_date: str, scheduled_dates: list, source: str = "manual"):
    """
    Upsert a weekly roster by week_start_date.
    scheduled_dates: list of ISO date strings, e.g. ["2026-06-07", "2026-06-09"]
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO weekly_rosters (
            week_start_date, week_end_date, scheduled_dates, source, created_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(week_start_date) DO UPDATE SET
            week_end_date   = excluded.week_end_date,
            scheduled_dates = excluded.scheduled_dates,
            source          = excluded.source,
            created_at      = excluded.created_at
    """, (
        week_start_date,
        week_end_date,
        json.dumps(scheduled_dates),
        source,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))

    conn.commit()
    conn.close()


def load_weekly_rosters_dataframe():
    """Return all weekly rosters sorted by week_start_date descending."""
    import pandas as pd

    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("""
        SELECT id, week_start_date, week_end_date, scheduled_dates, source, created_at
        FROM weekly_rosters
        ORDER BY week_start_date DESC
    """, conn)
    conn.close()
    return df


def load_weekly_roster_by_start(week_start_date: str):
    """
    Fetch a single weekly roster by its start date string (ISO format).
    Returns a dict or None if not found.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT week_start_date, week_end_date, scheduled_dates, source, created_at
        FROM weekly_rosters
        WHERE week_start_date = ?
    """, (week_start_date,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "week_start_date": row[0],
        "week_end_date": row[1],
        "scheduled_dates": json.loads(row[2]),
        "source": row[3],
        "created_at": row[4],
    }


def load_rosters_dataframe():
    import pandas as pd

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query("""
        SELECT month, year, scheduled_days, summary, created_at
        FROM rosters
    """, conn)

    conn.close()

    if not df.empty:
        month_order = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        df["month_number"] = df["month"].map(month_order)
        df = df.sort_values(["year", "month_number"])

    return df