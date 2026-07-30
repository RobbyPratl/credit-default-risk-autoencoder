"""Out-of-time train/validation splitting."""

import polars as pl


def temporal_train_val_split(df, train_frac=0.8, week_col="WEEK_NUM"):
    """Split a polars DataFrame into train/validation by calendar week.

    Credit models are deployed on future applicants, so validation has to be
    out-of-time: the earliest ``train_frac`` of the distinct weeks go to train
    and every later week goes to validation. Rows are never split across the
    boundary, so no week appears in both halves.

    Parameters
    ----------
    df : polars.DataFrame
        Must contain ``week_col``.
    train_frac : float, default 0.8
        Fraction of distinct weeks assigned to training. The cut index is
        ``int(n_weeks * train_frac)``.
    week_col : str, default "WEEK_NUM"
        Name of the week column.

    Returns
    -------
    tuple[polars.DataFrame, polars.DataFrame]
        ``(train_df, val_df)``.
    """
    weeks = sorted(df[week_col].unique().to_list())
    n = len(weeks)

    train_weeks = weeks[: int(n * train_frac)]
    val_weeks = weeks[int(n * train_frac) :]

    train_df = df.filter(pl.col(week_col).is_in(train_weeks))
    val_df = df.filter(pl.col(week_col).is_in(val_weeks))

    return train_df, val_df
