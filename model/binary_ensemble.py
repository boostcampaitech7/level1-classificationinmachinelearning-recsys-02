from typing import Dict
import numpy as np
import lightgbm as lgb

# default params
binary_params = {
    "boosting_type": "gbdt",
    'objective': 'binary',
    "metric": "binary_logloss",
    "num_leaves": 12,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbose": 0,
} 
params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 4,
    "num_leaves": 97,
    "learning_rate": 0.015732043600075817,
    "n_estimators": 52,
    "random_state": 42,
    "verbose": 0,
}


class BinaryEnsemble:
    def __init__(self, params=params, binary_params=binary_params):
        self.params = params
        self.binary_params = binary_params
        self.lgb_model = None
        self.binary_model = None
    
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        if valid_data is None or y_valid is None:
            # main lgb model
            train_data = lgb.Dataset(x_train, label=y_train)
            lgb_model = lgb.train(
                params=self.params,
                train_set=train_data,
            )
            self.lgb_model = lgb_model
            
            # binary model
            y_train_binary = y_train.apply(lambda x: 1 if x >= 2 else 0)
            train_data_binary = lgb.Dataset(x_train, label=y_train_binary)
            binary_model = lgb.train(
                params=self.binary_params,
                train_set=train_data_binary,
            )
            self.binary_model = binary_model
        else:
            # main lgb model
            train_data = lgb.Dataset(x_train, label=y_train)
            valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)
            lgb_model = lgb.train(
                params=self.params,
                train_set=train_data,
                valid_sets=valid_data,
            )
            # binary model
            y_train_binary = y_train.apply(lambda x: 1 if x >= 2 else 0)
            y_valid_binary = y_valid.apply(lambda x: 1 if x >= 2 else 0)
            train_data_binary = lgb.Dataset(x_train, label=y_train_binary)
            valid_data_binary = lgb.Dataset(x_valid, label=y_valid_binary, reference=train_data_binary)
            binary_model = lgb.train(
                params=self.binary_params,
                train_set=train_data_binary,
                valid_sets=valid_data_binary,
            )
            self.lgb_model = lgb_model
            self.binary_model = binary_model
    
    def predict_proba(self, x_valid):
        if self.lgb_model is None or self.binary_model is None:
            raise ValueError("Train first")
        
        y_valid_pred = self.lgb_model.predict(x_valid)
        y_valid_pred_split = self.binary_model.predict(x_valid)
        
        y_valid_pred_split_onehot = np.array([[1-p, 1-p, p, p] for p in y_valid_pred_split])
        y_valid_pred_ensemble = (y_valid_pred + y_valid_pred_split_onehot) / 2
        
        return y_valid_pred_ensemble
        
        
    def predict(self, x_valid):
        y_valid_pred_ensemble = self.predict_proba(x_valid)
        y_valid_pred_class = np.argmax(y_valid_pred_ensemble, axis = 1)
        
        return y_valid_pred_class
    
    
    