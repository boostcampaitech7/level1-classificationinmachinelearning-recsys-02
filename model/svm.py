from sklearn.svm import SVC

class CustomSVM:
    def __init__(self, params):
        self.model = None
        self.params = params
    
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        model = SVC(**self.params)
        model.fit(x_train, y_train)
        self.model = model
    
    def predict_proba(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        try:
            return self.model.predict_proba(x_valid)
        except AttributeError:
            raise ValueError("SVM does not have predict_proba method") 
            # SVM은 predict_proba 메소드를 실행을 위해선, probability=True로 설정해야함
            # 그러나, 이렇게 설정하면, SVM의 학습 시간이 길어짐
            
    def predict(self, x_valid):
        if self.model is None:
            raise ValueError("Train first")
        return self.model.predict(x_valid)