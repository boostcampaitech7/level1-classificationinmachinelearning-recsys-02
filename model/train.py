import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

def train_model(model_name, params, x_train, x_valid, y_train, y_valid, train_data, valid_data):
    if model_name == "LGBM":
        # lgb train
        model = LGBMClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        #predict
        y_valid_pred = model.predict_proba(x_valid)
        y_valid_pred_class = model.predict(x_valid)
    
    elif model_name == "FCLGBM":
        # Focal Loss lgb train
        model = lgb.train(params, train_data, valid_sets=valid_data)
        # predict
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-6)
        y_valid_pred = softmax(model.predict(x_valid))
        y_valid_pred_class = np.argmax(y_valid_pred, axis=1)
        
    elif model_name == "XGB":
        # XGBoost train
        model = XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        #predict
        y_valid_pred = model.predict_proba(x_valid)
        y_valid_pred_class = model.predict(x_valid)
        
    elif model_name == "RF":
        # RandomForestClassifier train
        model = RandomForestClassifier(**params)
        model.fit(x_train, y_train)
        #predict
        y_valid_pred_class = model.predict(x_valid)
        y_valid_pred = model.predict_proba(x_valid)

    elif model_name == "CatBoost":
        # CatBoost train
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid), verbose=0)

        #predict
        y_valid_pred_class = model.predict(x_valid)
        y_valid_pred = model.predict_proba(x_valid)

    elif model_name == "SVM":
        # SVM train
        model = SVC(**params)
        model.fit(x_train, y_train)

        # predict
        y_valid_pred_class = model.predict(x_valid)
        y_valid_pred = model.predict_proba(x_valid)

    else:
        print("Invalid model name. (def train_model)")

    # score check
    accuracy = accuracy_score(y_valid, y_valid_pred_class)
    try:
        auroc = roc_auc_score(y_valid, y_valid_pred, multi_class="ovr")
    except ValueError as e:
        print("Error calculating AUC:", e)
        auroc = 0.0

    return model, y_valid_pred_class, accuracy, auroc

def test(model_name, drop_colunm, model, test_df, submission_df):
    # LGBM predict
    if model_name == "LGBM":
        y_test_pred_class = model.predict(test_df.drop(drop_colunm, axis=1))
    
    # FCLGBM predict
    elif model_name == "FCLGBM":
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-6)
        y_test_pred = softmax(model.predict(test_df.drop(drop_colunm, axis = 1)))
        y_test_pred_class = np.argmax(y_test_pred, axis=1)

    # XGBoost predict
    elif model_name == "XGB":
        y_test_pred = model.predict_proba(test_df.drop(drop_colunm, axis=1))
        y_test_pred_class = model.predict(test_df.drop(drop_colunm, axis=1))

    elif model_name == "RF":
        y_test_pred_class = model.predict(test_df.drop(drop_colunm, axis = 1))

    elif model_name == "CatBoost":
        y_test_pred_class = model.predict(test_df.drop(drop_colunm, axis = 1))

    elif model_name == "SVM":
        y_test_pred_class = model.predict(test_df.drop(drop_colunm, axis = 1))

    else:
        print("Invalid model name. (Save Test)")

    # output file 할당후 save 
    submission_df = submission_df.assign(target = y_test_pred_class)
    return submission_df