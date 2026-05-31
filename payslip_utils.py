"""
payslip_utils.py — Deterministic payslip-based take-home estimator.

All functions return floats or dicts. No rounding inside helpers —
round at the display layer. No tax advice is implied.
"""


def effective_tax_rate(tax_withheld, gross_pay):
    """
    Effective tax rate from one payslip period.
    Returns None if gross_pay is zero or invalid.
    """
    if not gross_pay or gross_pay <= 0:
        return None
    return tax_withheld / gross_pay


def effective_net_rate(net_pay, gross_pay):
    """Net pay as a fraction of gross pay."""
    if not gross_pay or gross_pay <= 0:
        return None
    return net_pay / gross_pay


def estimate_tax_for_gross(gross, tax_rate):
    """Apply a payslip-derived tax rate to a new gross amount."""
    if tax_rate is None:
        return None
    return gross * tax_rate


def estimate_takehome(gross, tax_rate):
    """Gross minus estimated tax — no fuel deducted."""
    if tax_rate is None:
        return None
    return gross - estimate_tax_for_gross(gross, tax_rate)


def estimate_takehome_after_fuel(gross, tax_rate, fuel_cost, num_shifts):
    """Take-home after estimated tax and fuel. Fuel is NOT tax-deductible here."""
    th = estimate_takehome(gross, tax_rate)
    if th is None:
        return None
    return th - (fuel_cost * num_shifts)


def per_shift_takehome(takehome_after_fuel, num_shifts):
    """Average take-home per shift after tax and fuel."""
    if not num_shifts or num_shifts <= 0:
        return None
    return takehome_after_fuel / num_shifts


def reconciliation_check(gross, tax_withheld, net_pay, deductions, allowances, tolerance=0.50):
    """
    Check whether net_pay ≈ gross - tax_withheld - deductions + allowances.
    Returns (is_ok: bool, expected: float, difference: float).
    Tolerance is in dollars.
    """
    expected = gross - tax_withheld - deductions + allowances
    diff = abs(net_pay - expected)
    return diff <= tolerance, round(expected, 2), round(diff, 2)
