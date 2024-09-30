from model.binary_ensemble import BinaryEnsemble
from model.focal_loss_LGBM import FocalLossLGBM
from model.svm import CustomSVM
from model.tree_model import LGBM, XGB, RF, CatBoost
import argparse
import yaml
from dataloader.data_loader import data_loader
from datetime import datetime, timedelta
import csv
from utils.get_acc_auroc import get_accuracy_and_auroc
import os


def set_model(args, params):
    model = None
    if args.model == "LGBM":
        model = LGBM(params)
    elif args.model == "XGB":
        model = XGB(params)
    elif args.model == "RF":
        model = RF(params)
    elif args.model == "CatBoost":
        model = CatBoost(params)
    elif args.model == "SVM":
        model = CustomSVM(params)
    elif args.model == "FCLGBM":
        model = FocalLossLGBM(params)
    elif args.model == "BinaryEnsemble":
        model = BinaryEnsemble()
    else:
        print("Invalid model name")
        exit(1)
    return model


def save_submission(model, test_df, submission_df):
    y_test_pred_class = model.predict(test_df)
    submission_df = submission_df.assign(target = y_test_pred_class)
    submission_df.to_csv(args.save_name, index=False)


def save_log(args, accuracy, auroc, params):
    with open("log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([args.model, args.dataset, args.valid_type, args.save_name, accuracy, auroc, params, datetime.now()])


def train_model(model, train_df):
    model.train(train_df.drop(columns=["target"]), train_df["target"])
    return model


if __name__ == "__main__":
    print("This is train.py")
    parser = argparse.ArgumentParser()
    # param path
    path = "config"
    parser.add_argument("--config", default=os.path.join(path, "lgbm_param.yaml"))
    parser.add_argument("--dataset", default="real_final_df.csv")
    parser.add_argument("--drop_column", default=["ID", "dir_prob_ts"])
    parser.add_argument("--valid_type", default="random")
    parser.add_argument("--model", default="LGBM")
    parser.add_argument("--save_name", default="submission.csv")
    args = parser.parse_args()
    params = yaml.safe_load(open(os.path.join(path, args.config)))
    params = params["params"]

    model = set_model(args, params)
    train_df, test_df, submission_df = data_loader(args.dataset ,drop_column=args.drop_column)
    accuracy, auroc = get_accuracy_and_auroc(args.valid_type, model, train_df)
    model = train_model(model, train_df)
    save_submission(model, test_df, submission_df)
    save_log(args, accuracy, auroc, params)  