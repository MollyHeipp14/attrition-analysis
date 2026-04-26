import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def basic_df():
    # emp 1 & 3 left (Yes), emp 2 & 4 stayed (No)
    # overtime Yes: emp 1 & 3 (both left); overtime No: emp 2 & 4 (neither left)
    # leavers avg income: (4000 + 3000) / 2 = 3500; stayers: (6000 + 5000) / 2 = 5500
    return pd.DataFrame({
        "employee_id":    [1,       2,       3,    4],
        "department":     ["Sales", "Sales", "HR", "HR"],
        "overtime":       ["Yes",   "No",    "Yes", "No"],
        "monthly_income": [4000,    6000,    3000,  5000],
        "job_satisfaction": [1,     3,       2,     4],
        "attrition":      ["Yes",  "No",    "Yes", "No"],
    })


# --- attrition_rate ---

def test_attrition_rate_half(basic_df):
    assert attrition_rate(basic_df) == 50.0


def test_attrition_rate_zero():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(basic_df):
    result = attrition_by_department(basic_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(basic_df):
    result = attrition_by_department(basic_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    hr = result[result["department"] == "HR"].iloc[0]
    assert sales["attrition_rate"] == 50.0
    assert hr["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "department":  ["Sales", "Sales", "HR", "HR", "HR"],
        "attrition":   ["Yes",   "Yes",   "Yes", "No", "No"],
    })
    result = attrition_by_department(df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(basic_df):
    result = attrition_by_overtime(basic_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(basic_df):
    result = attrition_by_overtime(basic_df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(basic_df):
    result = average_income_by_attrition(basic_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(basic_df):
    result = average_income_by_attrition(basic_df)
    leavers = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    stayers = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert leavers == 3500.0
    assert stayers == 5500.0


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(basic_df):
    result = satisfaction_summary(basic_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_within_group_rate():
    # sat=1: 3 employees, 2 left  => 66.67%
    # sat=4: 2 employees, 0 left  => 0%
    df = pd.DataFrame({
        "employee_id":      [1,    2,    3,    4,    5],
        "job_satisfaction": [1,    1,    1,    4,    4],
        "attrition":        ["Yes","Yes","No", "No", "No"],
    })
    result = satisfaction_summary(df)
    sat1 = result[result["job_satisfaction"] == 1].iloc[0]
    sat4 = result[result["job_satisfaction"] == 4].iloc[0]
    assert sat1["attrition_rate"] == round(2 / 3 * 100, 2)
    assert sat4["attrition_rate"] == 0.0


def test_satisfaction_summary_sorted_ascending():
    df = pd.DataFrame({
        "employee_id":      [1,    2,    3],
        "job_satisfaction": [3,    1,    2],
        "attrition":        ["Yes","No", "Yes"],
    })
    result = satisfaction_summary(df)
    assert list(result["job_satisfaction"]) == [1, 2, 3]
