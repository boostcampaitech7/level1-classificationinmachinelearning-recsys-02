import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer  
import xgboost

class FillNaN:
  def __init__(self, df) -> None:
    self.df = df
    self.train_df = self.df[self.df._type == 'train'].drop(columns=['_type', 'target', 'ID'])
    self.test_df = self.df[self.df._type == 'test'].drop(columns=['_type', 'target', 'ID'])

  def MICE(self):
    temp = self.train_df.copy()

    iimp = IterativeImputer(
      estimator=xgboost.XGBRegressor(),
      random_state=42,
      verbose=2,
      max_iter=15
    )

    # IterativeImputer 적용
    final_train = iimp.fit_transform(self.train_df)
    final_test = iimp.transform(self.test_df)

    # 다시 DataFrame으로 변환
    train_df_filled = pd.DataFrame(final_train, columns=self.train_df.columns)
    test_df_filled = pd.DataFrame(final_test, columns=self.test_df.columns)

    # 원래 데이터에서 ID, target, _type 컬럼을 가져와서 결합
    df2 = pd.concat([train_df_filled, test_df_filled], axis=0).reset_index(drop=True)
    df2[['ID', 'target', '_type']] = self.df[['ID', 'target', '_type']]

    return df2
