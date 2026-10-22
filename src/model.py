
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def prepare_model_data(train, valid, predictors, target):
    """
    Prepare predictor variables and target labels.

    Parameters
    ----------
    train, valid : pandas.DataFrame
        Training and validation datasets.
    predictors : list
        Names of predictor columns.
    target : str
        Name of the target column.

    Returns
    -------
    tuple
        train_X, train_Y, valid_X, valid_Y.
    """
    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values

    return train_X, train_Y, valid_X, valid_Y


def create_random_forest(
    n_estimators=100,
    random_state=42,
    criterion="gini"
):
    """
    Initialise a Random Forest classifier.

    Returns
    -------
    RandomForestClassifier
        Untrained Random Forest model.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        criterion=criterion,
        n_jobs=-1,
        verbose=False
    )


def train_model(model, X_train, y_train):
    """
    Fit the classifier using training data.

    Returns
    -------
    RandomForestClassifier
        Fitted classifier.
    """
    model.fit(X_train, y_train)
    return model


def predict_model(model, X):
    """
    Predict class labels using a fitted model.

    Returns
    -------
    numpy.ndarray
        Predicted labels.
    """
    return model.predict(X)


def evaluate_model(y_true, y_pred):
    """
    Calculate classification accuracy.

    Returns
    -------
    float
        Proportion of correct predictions.
    """
    return accuracy_score(y_true, y_pred)
