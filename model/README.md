# 모델 구현 가이드

이 저장소는 여러 모델 클래스를 포함하고 있으며, 각각은 공통된 구조를 따릅니다. 이 가이드는 모든 모델이 동일한 패턴을 따르도록 하여, 모델을 쉽게 교체하거나 앙상블 할 수 있도록 돕기 위해 작성되었습니다.

## 기본 모델 클래스 구조

각 모델 클래스는 아래에 설명된 동일한 구조를 따라야 합니다. 클래스는 초기화 시 파라미터를 받아들이고, 학습, 예측 라벨, 예측 확률을 위한 메서드를 포함합니다.

### 클래스 템플릿

```python
class ModelName:
    def __init__(self, params):
        self.model = None  # 모델 객체를 저장할 변수
        self.params = params  # 전달받은 파라미터 저장

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        주어진 학습 데이터를 이용해 모델을 학습합니다.
        
        인자:
            x_train: 학습 데이터 특징(feature)들
            y_train: 학습 데이터 라벨(label)
            x_valid: 검증 데이터 특징
            y_valid: 검증 데이터 라벨
        
        반환값:
            없음
        """
        model = ModelClass(**self.params)  # 전달받은 파라미터로 모델 초기화
        if x_valid is None or y_valid is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid))  # 학습
        self.model = model  # 학습된 모델 저장

    def predict_proba(self, x_valid):
        """
        주어진 검증 데이터에 대해 클래스 확률을 예측합니다.
        
        인자:
            x_valid: 검증 데이터 특징
        
        반환값:
            각 클래스에 대한 예측 확률 값
        """
        if self.model is None:
            raise ValueError("먼저 모델을 학습시키세요.")
        
        y_valid_pred = self.model.predict_proba(x_valid)
        return y_valid_pred

    def predict(self, x_valid):
        """
        주어진 검증 데이터에 대해 클래스 라벨을 예측합니다.
        
        인자:
            x_valid: 검증 데이터 특징
        
        반환값:
            예측된 클래스 라벨
        """
        if self.model is None:
            raise ValueError("먼저 모델을 학습시키세요.")
        
        pred = self.model.predict(x_valid)
        return pred