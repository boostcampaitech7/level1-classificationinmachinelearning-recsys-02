from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import scipy
from pytorch_tabnet.augmentations import ClassificationSMOTE
tabnet_params = {
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": {"lr": 2e-2},
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "scheduler_params":{
        "step_size": 30, # 학습률 스케줄러 설정
        "gamma": 0.9,
    },
    "mask_type":  'entmax', # "sparsemax"
    "n_d": 80,
    "n_a": 80,
    "n_steps": 4,
    "gamma": 1.1,
}

class TabNet:
    def __init__(self, params, max_epochs=100, augmentations=0):
        self.params = params
        self.model = None
        self.max_epochs = max_epochs
        self.augmentations = augmentations
    
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        sparse_X_train = scipy.sparse.csr_matrix(x_train)
        self.model = TabNetClassifier(**self.params)
        self.model.fit(
            X_train=sparse_X_train, y_train=y_train,
            eval_name = ["train", "valid"],
            eval_metric=['accuracy'], # balanced_accuracy
            max_epochs=self.max_epochs,
            patience=20,
            batch_size=256,
            virtual_batch_size=256,
            num_workers=0,
            weights = {0:1.11, 1:1.11, 2:1, 3:1},
            drop_last=False,
            augmentations=ClassificationSMOTE(p=self.augmentations),
        )
    
    def predict_proba(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        
        sparse_X_valid = scipy.sparse.csr_matrix(x_valid)
        y_valid_pred = self.model.predict_proba(sparse_X_valid)
        return y_valid_pred
    
    def predict(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        
        sparse_X_valid = scipy.sparse.csr_matrix(x_valid)
        pred = self.model.predict(sparse_X_valid)
        return pred