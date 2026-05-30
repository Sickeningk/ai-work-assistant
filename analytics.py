import calendar
import json
from datetime import datetime


def build_work_schedule(month_name, year, scheduled_days):
    month_number = datetime.strptime(month_name, "%B").month

    work_schedule = {}

    for day in scheduled_days:
        date_obj = datetime(year, month_number, day)

        work_schedule[day] = {
            "weekday": calendar.day_name[date_obj.weekday()],
            "date": date_obj.strftime("%Y-%m-%d")
        }

    return work_schedule


def calculate_analytics_from_days(
    month,
    year,
    scheduled_days,
    hourly_rate,
    hours_per_shift,
    fuel_cost
):
    work_schedule = build_work_schedule(
        month,
        year,
        scheduled_days
    )

    total_shifts = len(scheduled_days)
    weekend_shifts = 0
    weekday_counts = {}

    for info in work_schedule.values():
        weekday = info["weekday"]

        if weekday in ["Saturday", "Sunday"]:
            weekend_shifts += 1

        weekday_counts[weekday] = weekday_counts.get(weekday, 0) + 1

    gross_income = total_shifts * hourly_rate * hours_per_shift
    net_income = gross_income - (fuel_cost * total_shifts)
    average_weekly_workload = round(total_shifts / 4.4, 1)

    return {
        "work_schedule": work_schedule,
        "total_shifts": total_shifts,
        "weekend_shifts": weekend_shifts,
        "weekday_counts": weekday_counts,
        "gross_income": gross_income,
        "net_income": net_income,
        "average_weekly_workload": average_weekly_workload
    }


def enrich_saved_rosters(
    df,
    hourly_rate,
    hours_per_shift,
    fuel_cost
):
    records = []

    for _, row in df.iterrows():
        scheduled_days = json.loads(row["scheduled_days"])

        analytics = calculate_analytics_from_days(
            row["month"],
            int(row["year"]),
            scheduled_days,
            hourly_rate,
            hours_per_shift,
            fuel_cost
        )

        records.append({
            "month": row["month"],
            "year": int(row["year"]),
            "scheduled_days": scheduled_days,
            "total_shifts": analytics["total_shifts"],
            "weekend_shifts": analytics["weekend_shifts"],
            "gross_income": analytics["gross_income"],
            "net_income": analytics["net_income"],
            "summary": row["summary"],
            "created_at": row["created_at"]
        })

    return records