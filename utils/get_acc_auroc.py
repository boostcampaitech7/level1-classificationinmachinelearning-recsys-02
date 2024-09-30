from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def get_accuracy_and_auroc(valid_type, model, train_df):
    """
    Calculate accuracy and AUROC (Area Under the Receiver Operating Characteristic Curve) 
    for a given model and training data using different validation strategies.

    Parameters:
    -----------
    valid_type : valid_type(CV, random, time)
    model : object
        The model to be trained and evaluated. Expected to have `train`, `predict`, and `predict_proba` methods.
    train_df : pandas.DataFrame
        The training data containing features and a target column named "target".

    Returns:
    --------
    accuracy : float
        The average accuracy of the model over the validation splits.
    auroc : float
        The average AUROC of the model over the validation splits.

    Raises:
    -------
    ValueError
        If `args.valid_type` is not one of "CV", "random", or "time".
    """
    if valid_type == "CV":
        # cross validation import
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        accuracy = []
        auroc = []
        for train_index, valid_index in skf.split(train_df.drop(columns=["target"]), train_df["target"]):
            x_train, x_valid = train_df.drop(columns=["target"]).iloc[train_index], train_df.drop(columns=["target"]).iloc[valid_index]
            y_train, y_valid = train_df["target"].iloc[train_index], train_df["target"].iloc[valid_index]
            model.train(x_train, y_train, x_valid, y_valid)
            y_valid_pred_class = model.predict(x_valid)
            y_valid_pred = model.predict_proba(x_valid)
            accuracy.append(accuracy_score(y_valid, y_valid_pred_class))
            auroc.append(roc_auc_score(y_valid, y_valid_pred, multi_class="ovr"))
        accuracy = sum(accuracy) / len(accuracy)
        auroc = sum(auroc) / len(auroc)  
        
          
    elif valid_type == "random":
        # random split import
        x_train, x_valid, y_train, y_valid = train_test_split(
            train_df.drop(columns=["target"]), 
            train_df["target"], 
            test_size=0.2,
            random_state=42,
            stratify=train_df["target"]
        )
        model.train(x_train, y_train, x_valid, y_valid)
        y_valid_pred_class = model.predict(x_valid)
        y_valid_pred = model.predict_proba(x_valid)
        accuracy = accuracy_score(y_valid, y_valid_pred_class)
        auroc = roc_auc_score(y_valid, y_valid_pred, multi_class="ovr")
        
        
    elif valid_type == "time":
        # time series split import
        x_train, x_valid, y_train, y_valid = train_test_split(
            train_df.drop(columns=["target"]), 
            train_df["target"], 
            test_size=0.2,
            random_state=42,
            shuffle=False
        )
        model.train(x_train, y_train, x_valid, y_valid)
        y_valid_pred_class = model.predict(x_valid)
        y_valid_pred = model.predict_proba(x_valid)
        accuracy = accuracy_score(y_valid, y_valid_pred_class)
        auroc = roc_auc_score(y_valid, y_valid_pred, multi_class="ovr")
    else:
        raise ValueError("Invalid validation type")
    
    return accuracy, auroc