import math


def gross_per_shift(hourly_rate, hours_per_shift):
    """Gross income for a single shift."""
    return hourly_rate * hours_per_shift


def net_per_shift(hourly_rate, hours_per_shift, fuel_cost):
    """Net income for a single shift after fuel."""
    return gross_per_shift(hourly_rate, hours_per_shift) - fuel_cost


def project_income(num_shifts, hourly_rate, hours_per_shift, fuel_cost):
    """
    Project gross and net income for a given number of shifts.
    Returns dict with gross and net.
    """
    gross = num_shifts * gross_per_shift(hourly_rate, hours_per_shift)
    net = num_shifts * net_per_shift(hourly_rate, hours_per_shift, fuel_cost)
    return {"shifts": num_shifts, "gross": round(gross, 2), "net": round(net, 2)}


def shifts_needed_for_target(target_net, hourly_rate, hours_per_shift, fuel_cost):
    """
    Return the minimum number of whole shifts needed to reach target_net income.
    Returns None if net per shift is zero or negative (can't reach target).
    """
    nps = net_per_shift(hourly_rate, hours_per_shift, fuel_cost)
    if nps <= 0:
        return None
    return math.ceil(target_net / nps)


def cost_of_skipping_shift(hourly_rate, hours_per_shift, fuel_cost):
    """
    Return the net income lost by skipping one shift.
    Also returns the fuel saved (a silver lining).
    """
    lost_gross = gross_per_shift(hourly_rate, hours_per_shift)
    saved_fuel = fuel_cost
    lost_net = lost_gross - saved_fuel
    return {
        "lost_gross": round(lost_gross, 2),
        "saved_fuel": round(saved_fuel, 2),
        "lost_net": round(lost_net, 2)
    }


def projected_monthly_income(saved_records, hourly_rate, hours_per_shift, fuel_cost):
    """
    Project a typical monthly income based on average shifts from full rosters.
    Rosters with fewer than 10 shifts are treated as partial and excluded.
    Returns None if no full rosters exist yet.
    """
    full_months = [r for r in saved_records if r["total_shifts"] >= 10]

    if not full_months:
        return None

    avg_shifts = sum(r["total_shifts"] for r in full_months) / len(full_months)
    projection = project_income(avg_shifts, hourly_rate, hours_per_shift, fuel_cost)
    projection["avg_shifts"] = round(avg_shifts, 1)
    projection["based_on_months"] = len(full_months)
    return projection
