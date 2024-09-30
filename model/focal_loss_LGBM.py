from scipy.misc import derivative
import numpy as np
import lightgbm as lgb

focal_loss = lambda x,y: focal_loss_lgb(x, y, 0.25, 2.0, 4)
fclgbm_params = { "num_class":4,
        "objective": focal_loss,
        "boosting_type": "gbdt",
        "num_class": 4,
        "num_leaves": 60,
        "learning_rate": 0.05,
        "n_estimators": 26,
        "random_state": 42,
        "verbose": 0,
}

class FocalLossLGBM:
    def __init__(self, alpha=0.25, gamma=2, params=fclgbm_params):
        self.alpha = alpha
        self.gamma = gamma
        self.model = None
        self.binary_model = None
        self.params = params
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        if x_valid is None or y_valid is None:
            train_data = lgb.Dataset(x_train, label=y_train)
            model = lgb.train(
                params=self.params,
                train_set=train_data,
            )
            self.model = model
        
        # main lgb model
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)
        model = lgb.train(
            params=self.params,
            train_set=train_data,
            valid_sets=valid_data,
        )
        self.model = model
        
    def predict_proba(self, x_valid):
        if self.lgb_model is None:
            raise ValueError("Train first")
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-6)
        y_valid_pred = softmax(self.model.predict(x_valid))
        y_valid_pred = self.model.predict(x_valid)
        return y_valid_pred
    
    def predict(self, x_valid):
        y_valid_pred = self.predict_proba(x_valid)
        y_valid_pred = np.argmax(y_valid_pred, axis=1)
        return y_valid_pred

# https://github.com/jrzaurin/LightGBM-with-Focal-Loss/blob/master/examples/Lightgbm_with_Focal_Loss_multiclass.ipynb 참고
def focal_loss_lgb(y_pred, dtrain, alpha, gamma, num_class):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    # N observations x num_class arrays
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1,num_class, order='F')
    # alpha and gamma multiplicative factors with BCEWithLogitsLoss
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    # flatten in column-major (Fortran-style) order
    return grad.flatten('F'), hess.flatten('F')