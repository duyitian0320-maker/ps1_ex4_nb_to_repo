
import matplotlib.pyplot as plt
import seaborn as sns


def plot_category_counts(df, column, hue="set"):
    """
    Plot category counts for training and test datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to plot.
    column : str
        Column to display on the x-axis.
    hue : str, optional
        Column used to separate groups.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the completed plot.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=column,
        hue=hue,
        ax=ax
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.5
    )

    ax.set_axisbelow(True)
    ax.set_title(f"Number of passengers / {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Number of passengers")

    plt.tight_layout()

    return ax

def plot_distribution(df, column, hue=None):
    """
    Plot the distribution of a numerical variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the variable.
    column : str
        Numerical column to visualise.
    hue : str or None, optional
        Variable used to group observations.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the histogram.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x=column,
        hue=hue,
        bins=30,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Number of passengers")

    plt.tight_layout()
    return ax