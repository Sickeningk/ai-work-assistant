from datetime import date, datetime, timedelta


def get_today():
    """Return today's date from the local system clock."""
    return date.today()


def get_week_range(offset=0):
    """
    Return (start, end) date objects for a week relative to today.
    offset=0  -> this week (Monday to Sunday)
    offset=-1 -> last week
    offset=1  -> next week
    """
    today = get_today()
    monday = today - timedelta(days=today.weekday()) + timedelta(weeks=offset)
    sunday = monday + timedelta(days=6)
    return monday, sunday


def get_next_shift(scheduled_days, month_name, year):
    """
    Return the next shift date on or after today, or None if the
    roster is entirely in the past.
    """
    today = get_today()
    month_number = datetime.strptime(month_name, "%B").month

    upcoming = []
    for day in scheduled_days:
        try:
            shift_date = date(year, month_number, day)
            if shift_date >= today:
                upcoming.append(shift_date)
        except ValueError:
            continue

    return min(upcoming) if upcoming else None


def get_days_until(target_date):
    """Return the number of days from today until target_date."""
    return (target_date - get_today()).days


def shifts_this_week(scheduled_days, month_name, year):
    """Return list of shift dates falling within the current week."""
    start, end = get_week_range(offset=0)
    month_number = datetime.strptime(month_name, "%B").month

    result = []
    for day in scheduled_days:
        try:
            shift_date = date(year, month_number, day)
            if start <= shift_date <= end:
                result.append(shift_date)
        except ValueError:
            continue

    return sorted(result)


def shifts_next_week(scheduled_days, month_name, year):
    """Return list of shift dates falling within next week."""
    start, end = get_week_range(offset=1)
    month_number = datetime.strptime(month_name, "%B").month

    result = []
    for day in scheduled_days:
        try:
            shift_date = date(year, month_number, day)
            if start <= shift_date <= end:
                result.append(shift_date)
        except ValueError:
            continue

    return sorted(result)

def build_week_breakdown(month_name, year, scheduled_days, hourly_rate, hours_per_shift, fuel_cost):
    """
    Break a roster month into Monday-to-Sunday weeks and calculate
    deterministic income figures for each week.

    Week numbering starts at 1 from the Monday on or before the first
    day of the month.  Weeks continue until the last day of the month
    is covered.  A week may span into the previous or next month — only
    shifts within the month's scheduled_days are counted.

    Returns a list of dicts, one per week:
        week_number  : int  (1-based)
        week_start   : str  "Monday 27 April 2026"
        week_end     : str  "Sunday 03 May 2026"
        shift_count  : int
        shift_dates  : list[str]  each formatted "Weekday DD Month YYYY"
        gross_income : float
        net_income   : float
    """
    import calendar as _calendar

    month_number = datetime.strptime(month_name, "%B").month
    first_day = date(year, month_number, 1)
    last_day = date(year, month_number, _calendar.monthrange(year, month_number)[1])

    # Monday on or before the 1st of the month
    first_monday = first_day - timedelta(days=first_day.weekday())

    shift_dates_set = set()
    for day in scheduled_days:
        try:
            shift_dates_set.add(date(year, month_number, day))
        except ValueError:
            continue

    weeks = []
    week_start = first_monday
    week_num = 1

    while week_start <= last_day:
        week_end = week_start + timedelta(days=6)
        shifts_in_week = sorted(
            [d for d in shift_dates_set if week_start <= d <= week_end]
        )
        shift_count = len(shifts_in_week)
        gross = round(shift_count * hourly_rate * hours_per_shift, 2)
        net = round(gross - (fuel_cost * shift_count), 2)

        weeks.append({
            "week_number": week_num,
            "week_start": week_start.strftime("%A %d %B %Y"),
            "week_end": week_end.strftime("%A %d %B %Y"),
            "shift_count": shift_count,
            "shift_dates": [d.strftime("%A %d %B %Y") for d in shifts_in_week],
            "gross_income": gross,
            "net_income": net,
        })

        week_start += timedelta(weeks=1)
        week_num += 1

    return weeks
