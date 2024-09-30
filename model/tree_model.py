from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

class LGBM:
    def __init__(self, params):
        self.model = None
        self.params = params
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        model = LGBMClassifier(**self.params)
        if x_valid is None or y_valid is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        self.model = model
    
    def predict_proba(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        y_valid_pred = self.model.predict_proba(x_valid)
        return y_valid_pred
    
    def predict(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        pred = self.model.predict(x_valid)
        return pred
    
class XGB:
    def __init__(self, params):
        self.model = None
        self.params = params
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        model = XGBClassifier(**self.params)
        if x_valid is None or y_valid is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        self.model = model
    
    def predict_proba(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        y_valid_pred = self.model.predict_proba(x_valid)
        return y_valid_pred
    
    def predict(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        pred = self.model.predict(x_valid)
        return pred

class RF:
    def __init__(self, params):
        self.model = None
        self.params = params
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        model = RandomForestClassifier(**self.params)
        model.fit(x_train, y_train)
        self.model = model
    
    def predict_proba(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        y_valid_pred = self.model.predict_proba(x_valid)
        return y_valid_pred
    
    def predict(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        pred = self.model.predict(x_valid)
        return pred
    
class CatBoost:
    def __init__(self, params):
        self.model = None
        self.params = params
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        model = CatBoostClassifier(**self.params)
        if x_valid is None or y_valid is None:
            model.fit(x_train, y_train, verbose=0)
        else:
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid), verbose=0)
        self.model = model
    
    def predict_proba(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        y_valid_pred = self.model.predict_proba(x_valid)
        return y_valid_pred
    
    def predict(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        pred = self.model.predict(x_valid)
        return pred