
import pandas as pd


def add_family_size(df):
    """
    Create a Family Size feature from SibSp and Parch.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing SibSp and Parch.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataset with Family Size added.
    """
    result = df.copy()
    result["Family Size"] = result["SibSp"] + result["Parch"] + 1
    return result


def add_age_interval(df):
    """
    Categorise passenger ages into five intervals.

    Categories:
    0: Age <= 16
    1: 16 < Age <= 32
    2: 32 < Age <= 48
    3: 48 < Age <= 64
    4: Age > 64

    Missing ages remain missing.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing Age.

    Returns
    -------
    pandas.DataFrame
        Copy with the Age Interval feature added.
    """
    result = df.copy()

    result["Age Interval"] = pd.cut(
        result["Age"],
        bins=[float("-inf"), 16, 32, 48, 64, float("inf")],
        labels=[0, 1, 2, 3, 4]
    ).astype("float")

    return result

def add_fare_interval(df, bins, labels):
    """
    Categorise ticket fares into intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing Fare.
    bins : list
        Boundaries of the fare intervals.
    labels : list
        Category labels.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataset with Fare Interval added.
    """
    result = df.copy()

    result["Fare Interval"] = pd.cut(
        result["Fare"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return result


def add_sex_pclass(df):
    """
    Create a combined Sex and Pclass feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing Sex and Pclass.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataset with Sex_Pclass added.
    """
    result = df.copy()

    result["Sex_Pclass"] = (
        result["Sex"].str[0].str.upper()
        + result["Pclass"].astype(str)
    )

    return result


def parse_names(row):
    """
    Extract family name, title, given name and maiden name.

    Parameters
    ----------
    row : pandas.Series
        Passenger record containing a Name column.

    Returns
    -------
    pandas.Series
        Four extracted name components in this order:
        family name, title, given name, maiden name.
        Maiden name is None when unavailable.
    """
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]

        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()

        next_text = split_text[1]

        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")

            return pd.Series([
                family_name, title, given_name, maiden_name
            ])
        else:
            given_name = next_text

            return pd.Series([
                family_name, title, given_name, None
            ])

    except Exception as ex:
        print(f"Exception: {ex}")

def add_name_features(df):
    """
    Extract passenger name components into new columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing a Name column.

    Returns
    -------
    pandas.DataFrame
        Copy with Family Name, Title, Given Name,
        and Maiden Name columns added.
    """
    result = df.copy()

    columns = [
        "Family Name",
        "Title",
        "Given Name",
        "Maiden Name"
    ]

    result[columns] = result.apply(
        parse_names,
        axis=1
    )

    return result


def add_family_type(df):
    """
    Categorise passengers as Single, Small or Large families.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing Family Size.

    Returns
    -------
    pandas.DataFrame
        Copy with Family Type added.
    """
    result = df.copy()

    result["Family Type"] = result["Family Size"].map(
        lambda size: (
            "Single" if size == 1
            else "Small" if 1 < size < 5
            else "Large"
        )
    )

    return result


def aggregate_titles(df):
    """
    Consolidate passenger titles into common categories.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing Title.

    Returns
    -------
    pandas.DataFrame
        Copy with a standardised Titles column.
    """
    result = df.copy()
    result["Titles"] = result["Title"].copy()

    title_mapping = {
        "Mlle.": "Miss.",
        "Ms.": "Miss.",
        "Mme.": "Mrs.",
        "Lady.": "Rare",
        "the Countess.": "Rare",
        "Capt.": "Rare",
        "Col.": "Rare",
        "Don.": "Rare",
        "Dr.": "Rare",
        "Major.": "Rare",
        "Rev.": "Rare",
        "Sir.": "Rare",
        "Jonkheer.": "Rare",
        "Dona.": "Rare"
    }

    result["Titles"] = result["Titles"].replace(title_mapping)

    return result


def survival_by_title(df):
    """
    Calculate mean survival by passenger title and sex.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing Titles, Sex and Survived.

    Returns
    -------
    pandas.DataFrame
        Survival rates grouped by Titles and Sex.
    """
    return (
        df[["Titles", "Sex", "Survived"]]
        .groupby(["Titles", "Sex"], as_index=False)
        .mean()
    )


def encode_sex(df):
    """
    Encode Sex numerically: female=1, male=0.

    Accepts both original string values and already
    encoded numerical values.
    """
    result = df.copy()

    result["Sex"] = result["Sex"].replace({
        "female": 1,
        "male": 0
    })

    return result
