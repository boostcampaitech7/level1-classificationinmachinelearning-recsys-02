import os
import pandas as pd
from typing import List

def data_loader(dataset_name: str, drop_column=None) -> pd.DataFrame:
    # 파일 호출
    data_path: str = "data"
    df: pd.DataFrame = pd.read_csv(os.path.join(data_path, dataset_name))
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    df = df.drop(columns=drop_column)
    submission_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")) # ID, target 열만 가진 데이터 미리 호출
    # 타겟 변수를 제외한 변수를 forwardfill, -999로 결측치 대체
    _target = df["target"]
    df = df.ffill().fillna(-999).assign(target = _target)

    # _type에 따라 train, test 분리
    train_df = df.loc[df["_type"]=="train"].drop(columns=["_type"])
    test_df = df.loc[df["_type"]=="test"].drop(columns=["_type", "target"])
    
    return train_df, test_df, submission_df