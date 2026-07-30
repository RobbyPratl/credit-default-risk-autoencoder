import polars as pl
import pytest

from credit_risk import temporal_train_val_split


def make_df(n_weeks=10, rows_per_week=3, week_col="WEEK_NUM"):
    weeks = [w for w in range(n_weeks) for _ in range(rows_per_week)]
    return pl.DataFrame(
        {
            "case_id": list(range(len(weeks))),
            week_col: weeks,
            "target": [i % 2 for i in range(len(weeks))],
        }
    )


def test_split_is_row_disjoint_and_complete():
    df = make_df()
    train, val = temporal_train_val_split(df)

    train_ids = set(train["case_id"].to_list())
    val_ids = set(val["case_id"].to_list())

    assert train_ids & val_ids == set()
    assert train_ids | val_ids == set(df["case_id"].to_list())
    assert len(train) + len(val) == len(df)


def test_every_val_week_is_strictly_later_than_every_train_week():
    df = make_df(n_weeks=17)
    train, val = temporal_train_val_split(df)

    assert train["WEEK_NUM"].max() < val["WEEK_NUM"].min()
    assert set(train["WEEK_NUM"].to_list()) & set(val["WEEK_NUM"].to_list()) == set()


@pytest.mark.parametrize("train_frac", [0.5, 0.8, 0.9])
def test_split_fraction_is_respected(train_frac):
    n_weeks = 20
    df = make_df(n_weeks=n_weeks)
    train, val = temporal_train_val_split(df, train_frac=train_frac)

    expected_train_weeks = int(n_weeks * train_frac)
    assert train["WEEK_NUM"].n_unique() == expected_train_weeks
    assert val["WEEK_NUM"].n_unique() == n_weeks - expected_train_weeks


def test_unsorted_and_irregular_weeks_still_split_chronologically():
    # Weeks arrive out of order and are not contiguous integers.
    df = pl.DataFrame(
        {
            "case_id": list(range(6)),
            "WEEK_NUM": [7, 0, 91, 13, 0, 44],
        }
    )
    train, val = temporal_train_val_split(df, train_frac=0.5)

    assert sorted(train["WEEK_NUM"].unique().to_list()) == [0, 7]
    assert sorted(val["WEEK_NUM"].unique().to_list()) == [13, 44, 91]
    assert train["WEEK_NUM"].max() < val["WEEK_NUM"].min()


def test_custom_week_column():
    df = make_df(week_col="week")
    train, val = temporal_train_val_split(df, week_col="week")

    assert train["week"].max() < val["week"].min()
