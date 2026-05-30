import pandas as pd
import json


def build_export_dataframe(saved_records):
    """
    Takes the enriched list of roster records and returns a clean
    DataFrame ready for CSV export.

    scheduled_days is converted from a list to a readable string.
    gross_income and net_income are rounded to 2 decimal places.
    """
    rows = []

    for r in saved_records:
        days = r["scheduled_days"]
        if isinstance(days, str):
            days = json.loads(days)

        rows.append({
            "month": r["month"],
            "year": r["year"],
            "scheduled_days": ", ".join(str(d) for d in sorted(days)),
            "total_shifts": r["total_shifts"],
            "weekend_shifts": r["weekend_shifts"],
            "gross_income": round(r["gross_income"], 2),
            "net_income": round(r["net_income"], 2),
            "created_at": r["created_at"]
        })

    return pd.DataFrame(rows, columns=[
        "month", "year", "scheduled_days",
        "total_shifts", "weekend_shifts",
        "gross_income", "net_income", "created_at"
    ])
