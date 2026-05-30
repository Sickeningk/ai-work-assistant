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