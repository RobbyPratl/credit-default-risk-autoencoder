import pandas as pd
import polars as pl

from credit_risk import select_numeric_features

COLUMNS = [
    "case_id",
    "target",
    "WEEK_NUM",
    "MONTH",
    "date_decision",
    "annuity_780A",
    "credamount_770A",
    "pmtnum_254L",
    "numinstlswithdpd5_4187116P",
    "maxdpdtolerance_374P",
    "description_5085714M",
    "isbidproduct_1095L",
]


def make_df(kind):
    data = {c: [0] for c in COLUMNS}
    return pl.DataFrame(data) if kind == "polars" else pd.DataFrame(data)


def test_selects_only_a_and_p_columns():
    for kind in ("polars", "pandas"):
        assert select_numeric_features(make_df(kind)) == [
            "annuity_780A",
            "credamount_770A",
            "numinstlswithdpd5_4187116P",
            "maxdpdtolerance_374P",
        ]


def test_excludes_id_and_meta_columns():
    selected = select_numeric_features(make_df("polars"))

    for meta in ("case_id", "target", "WEEK_NUM", "MONTH", "date_decision"):
        assert meta not in selected


def test_meta_columns_excluded_even_when_suffix_would_match():
    # "MONTH" ends in H, "WEEK_NUM" in M -- ask for those suffixes explicitly
    # and the exclusion list must still win.
    df = pl.DataFrame({c: [0] for c in ["MONTH", "WEEK_NUM", "description_5085714M"]})

    assert select_numeric_features(df, suffixes=("H", "M")) == ["description_5085714M"]


def test_custom_suffixes():
    df = make_df("polars")

    assert select_numeric_features(df, suffixes=("L",)) == [
        "pmtnum_254L",
        "isbidproduct_1095L",
    ]
    assert select_numeric_features(df, suffixes=("A",)) == [
        "annuity_780A",
        "credamount_770A",
    ]
    assert select_numeric_features(df, suffixes=()) == []


def test_preserves_dataframe_column_order():
    df = pl.DataFrame({c: [0] for c in ["credamount_770A", "case_id", "annuity_780A"]})

    assert select_numeric_features(df) == ["credamount_770A", "annuity_780A"]
