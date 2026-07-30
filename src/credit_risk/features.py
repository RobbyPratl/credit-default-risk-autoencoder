"""Feature-column selection based on the dataset's naming convention."""

EXCLUDED_COLS = ("case_id", "target", "WEEK_NUM", "MONTH", "date_decision")


def select_numeric_features(df, suffixes=("A", "P")):
    """Return the feature columns whose name ends in one of ``suffixes``.

    Home Credit encodes a column's type in the last character of its name.
    The two numeric families used by the baseline are:

    * ``A`` — amounts (balances, annuities, credit limits, ...)
    * ``P`` — DPD buckets and counts

    Identifier and metadata columns (``case_id``, ``target``, ``WEEK_NUM``,
    ``MONTH``, ``date_decision``) are always dropped, even if a future suffix
    choice would otherwise match them.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        Any object exposing a ``columns`` sequence.
    suffixes : tuple[str, ...], default ("A", "P")
        Name suffixes to keep.

    Returns
    -------
    list[str]
        Matching column names, in the DataFrame's own column order.
    """
    suffixes = tuple(suffixes)
    return [
        c
        for c in df.columns
        if c.endswith(suffixes) and c not in EXCLUDED_COLS
    ]
