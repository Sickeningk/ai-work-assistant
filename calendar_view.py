import calendar
from datetime import date
from time_utils import get_today


def render_calendar(month_name, year, scheduled_days):
    """
    Returns an HTML string of a styled monthly calendar grid.
    Highlights scheduled work days and marks today's date.
    Columns run Monday to Sunday.
    """
    month_number = list(calendar.month_name).index(month_name)
    today = get_today()
    scheduled_set = set(scheduled_days)

    # calendar.monthcalendar returns list of weeks [Mon..Sun], 0 = empty
    weeks = calendar.monthcalendar(year, month_number)

    day_headers = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # --- Styles ---
    table_style = (
        "border-collapse: collapse; "
        "width: 100%; "
        "font-family: sans-serif; "
        "font-size: 14px;"
    )
    header_cell_style = (
        "padding: 8px 4px; "
        "text-align: center; "
        "color: #aaaaaa; "
        "font-weight: 600; "
        "border-bottom: 1px solid #444;"
    )
    work_day_style = (
        "padding: 10px 4px; "
        "text-align: center; "
        "background-color: #1a6b9a; "
        "color: #ffffff; "
        "font-weight: 700; "
        "border-radius: 6px; "
        "border: 1px solid #1a6b9a;"
    )
    today_work_style = (
        "padding: 10px 4px; "
        "text-align: center; "
        "background-color: #1a6b9a; "
        "color: #ffffff; "
        "font-weight: 700; "
        "border-radius: 6px; "
        "border: 2px solid #f0c040;"
    )
    today_off_style = (
        "padding: 10px 4px; "
        "text-align: center; "
        "background-color: #2a2a2a; "
        "color: #f0c040; "
        "font-weight: 700; "
        "border-radius: 6px; "
        "border: 2px solid #f0c040;"
    )
    off_day_style = (
        "padding: 10px 4px; "
        "text-align: center; "
        "background-color: #2a2a2a; "
        "color: #cccccc; "
        "border-radius: 6px; "
        "border: 1px solid #333;"
    )
    empty_cell_style = (
        "padding: 10px 4px; "
        "text-align: center; "
        "background-color: transparent; "
        "color: transparent; "
        "border: 1px solid transparent;"
    )

    # --- Build HTML ---
    html = f'<table style="{table_style}"><thead><tr>'

    for header in day_headers:
        html += f'<th style="{header_cell_style}">{header}</th>'

    html += "</tr></thead><tbody>"

    for week in weeks:
        html += "<tr>"
        for day in week:
            if day == 0:
                html += f'<td style="{empty_cell_style}">·</td>'
            else:
                is_work = day in scheduled_set
                is_today = (
                    year == today.year and
                    month_number == today.month and
                    day == today.day
                )

                if is_today and is_work:
                    cell_style = today_work_style
                elif is_today:
                    cell_style = today_off_style
                elif is_work:
                    cell_style = work_day_style
                else:
                    cell_style = off_day_style

                html += f'<td style="{cell_style}">{day}</td>'
        html += "</tr>"

    html += "</tbody></table>"
    return html
