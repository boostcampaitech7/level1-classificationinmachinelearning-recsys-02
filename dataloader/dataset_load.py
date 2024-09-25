from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool

def data_split(split_type, train_df, drop_colunm, target_colunm):
    if split_type == "random":
        # train_test_split 으로 valid set, train set 분리
        x_train, x_valid, y_train, y_valid = train_test_split(
            train_df.drop(drop_colunm, axis = 1), 
            train_df[target_colunm].astype(int), 
            test_size=0.2,
            random_state=42,
            stratify=train_df[target_colunm]
        )
    elif split_type == "time":
        # time series valid, train set 분리
        x_train = train_df[train_df.ID < '2023-11-01'].drop(drop_colunm, axis = 1)
        y_train = train_df[train_df.ID < '2023-11-01'][target_colunm].astype(int)

        x_valid = train_df[train_df.ID >= '2023-11-01'].drop(drop_colunm, axis = 1)
        y_valid = train_df[train_df.ID >= '2023-11-01'][target_colunm].astype(int)
    
    # random CV
    elif split_type == "randomcv":
        # cross validation
        x_train = train_df.drop(drop_colunm, axis = 1)
        y_train = train_df[target_colunm].astype(int)
        x_train_list = []
        y_train_list = []
        x_valid_list = []
        y_valid_list = []
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        
        for train_index, valid_index in skf.split(x_train, y_train):
            x_train_0, x_valid_0 = x_train.iloc[train_index], x_train.iloc[valid_index]
            x_train_list.append(x_train_0)
            x_valid_list.append(x_valid_0)
            y_train_0, y_valid_0 = y_train.iloc[train_index], y_train.iloc[valid_index]
            y_train_list.append(y_train_0)
            y_valid_list.append(y_valid_0)
        return x_train_list, x_valid_list, y_train_list, y_valid_list    
        
    else:
        print("Invalid model name. (Dataset Split)")
        
    return x_train, x_valid, y_train, y_valid

def _Dataset(model_name, x_train, x_valid, y_train, y_valid):
    if model_name == "LGBM" or model_name == "FCLGBM":
        # lgb dataset
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)
    elif model_name == "XGB":
        # XGBoost dataset
        train_data = (x_train, y_train)
        valid_data = (x_valid, y_valid)
    elif model_name == "RF":
        # RandomForest doesn't require specific dataset format
        train_data = (x_train, y_train)
        valid_data = (x_valid, y_valid)
    elif model_name == "CatBoost":
        # CatBoost dataset
        train_data = Pool(x_train, y_train)
        valid_data = Pool(x_valid, y_valid)
    elif model_name == "SVM":
        # SVM doesn't require specific dataset format
        train_data = (x_train, y_train)
        valid_data = (x_valid, y_valid)
    else:
        print("Invalid model name. (Dataset Preprocessing)")

    return train_data, valid_data