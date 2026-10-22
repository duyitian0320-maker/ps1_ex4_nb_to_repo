import pandas as pd


def load_data(file_path):
    """
    Load a CSV file and return it as a pandas DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(file_path)

def missing_summary(df):
    """
    Summarise missing values and data types for each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect.

    Returns
    -------
    pandas.DataFrame
        Summary containing Total, Percent and Types.
    """
    total = df.isnull().sum()
    percent = df.isnull().mean() * 100

    summary = pd.DataFrame({
        "Total": total,
        "Percent": percent,
        "Types": df.dtypes.astype(str)
    })

    return summary.T

def frequency_summary(df):
    """
    Summarise the most frequent value in each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect.

    Returns
    -------
    pandas.DataFrame
        Non-missing counts, most frequent values,
        frequencies and percentages by column.
    """
    total = df.count()
    items = []
    vals = []

    for col in df.columns:
        counts = df[col].value_counts()

        if not counts.empty:
            items.append(counts.index[0])
            vals.append(counts.iloc[0])
        else:
            items.append(0)
            vals.append(0)

    summary = pd.DataFrame({
        "Total": total,
        "Most frequent item": items,
        "Frequence": vals
    }, index=df.columns)

    summary["Percent from total"] = (
        summary["Frequence"]
        .div(summary["Total"].replace(0, float("nan")))
        .mul(100)
        .round(3)
    )

    return summary.T

def unique_summary(df):
    """
    Summarise non-missing and unique values in each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect.

    Returns
    -------
    pandas.DataFrame
        Non-missing counts and unique-value counts
        for each column.
    """
    summary = pd.DataFrame({
        "Total": df.count(),
        "Uniques": df.nunique()
    })

    return summary.T