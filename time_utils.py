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
