# 암호화폐 가격 예측 AI 대회

## 💡Team

|강현구|서동준|이도걸|이수미|최윤혜|
|:---:|:---:|:---:|:---:|:---:|
|<img src="https://github.com/user-attachments/assets/e1405e2b-4606-4a66-9b0c-eb7a70e941d4" width="150" height="150"/>|<img src="https://github.com/user-attachments/assets/7c03fd89-73e1-4580-aec5-46fe806b613c" width="150" height="150"/>|<img src="https://github.com/user-attachments/assets/eb46b31c-8046-49a8-8404-9292982e4582" width="150" height="150"/>|<img src="https://github.com/user-attachments/assets/e317d6b7-953a-46b1-b65d-48dc1d1075af" width="150" height="150"/>|<img src="https://github.com/user-attachments/assets/57762658-ec2c-4914-a4db-5080c105da16" width="150" height="150"/>|
|Modeling, hyperparameter tunning|Modeling, model modularization, Train pipeline refactoring|Modeling, Modularization, hyperparameter tunning|EDA, Feature Engineering, Data Preprocessing, Modeling|EDA, Feature Engineering, Data Augmentation, Modeling|

</br>

## 🪙Introduction
암호화폐는 전 세계적으로 온라인 상거래나 금융 거래 시 사용되며, 중간 거래자나 수수료 없이 안전하고 신속한 거래를 가능하게 합니다. 일반적으로 암호화폐의 가격 변동성은 주가보다 더 크기 때문에 예측이 더욱 어렵습니다. 하지만 인공지능과 머신러닝의 발전으로 다양한 예측 모델이 성공을 거두고 있습니다. 비트코인의 가격 등락 예측이 정확하게 이루어진다면, 투자자들의 투자 전략 수립에 큰 도움이 될 것입니다. 또한 이러한 모델 개발을 통해 얻어진 인사이트는 다른 금융 상품의 예측에도 기여할 수 있습니다. 따라서 이번 대회에서는 **비트코인의 다음 시점(한 시간 뒤) 가격 등락** 예측을 목표로 합니다.

본 대회는 주어진 시점에서 가격의 상승 또는 하락 정도를 예측하는 **다중 범주 분류 문제**입니다. 평가 지표로는 **Accuracy**를 사용하며, 리더보드와 최종 평가도 오직 Accuracy를 기준으로 이루어집니다.

|class |description       |count  |
|------|------------------|-------|
| 0    | -0.5% 미만        | 740   |
| 1    | -0.5% 이상 0% 미만 | 3544  |
| 2    | 0% 이상 0.5% 미만  | 3671  |
| 3    | 0.5% 이상         | 805   |

</br>

## 💾Datasets
본 대회에서 제공된 데이터는 **CryptoQuant**에서 1차적으로 정제된 **블록체인의 온체인 데이터**입니다. 온체인 데이터는 블록체인 상에서 발생하는 활동을 보여주는 정보로, 두 가지로 나뉩니다:
- **네트워크 데이터(Network Data)**: 블록체인 내에서 파악할 수 있는 데이터 (예: 활성화된 지갑 수, 트랜잭션 수 등)
- **시장 데이터(Market Data)**: 가상화폐 거래소에서 생성되는 가격 등 시장 정보와 관련된 데이터 (예: 거래량, 청산량 등)


</br>

## ⭐Project Summary
- EDA: EDA를 통해 유의한 변수 파악 및 파생변수 생성
- Preprocessing : pca, 변수 로그변환, MICE 결측치 처리, 변수 선택 등을 통한 데이터 정리
- Modeling: RandomForest, XGBoost, CatBoost, LGBM, tabnet, autoML 등 
- Ensemble : soft / hard voting을 통한 다양한 모델 앙상블 시도
- Hyper-parameter tuning : bayes optimization을 통한 파라미터 튜닝

</br>

## 📑Wrap-up Report
[RecSys기초대회_RecSys_팀 리포트(02조).pdf](https://github.com/boostcampaitech7/level1-classificationinmachinelearning-recsys-02/blob/main/RecSys_02%20Wrap%20up%20report.pdf)


</br>

## 📂Architecture
```
📦level1-classificationinmachinelearning-recsys-02
 ┣ 📂dataloader
 ┃ ┣ 📜dataset_load.py
 ┃ ┣ 📜data_loader.py
 ┃ ┗ 📜README.md
 ┣ 📂dataset
 ┃ ┣ 📜apply_PCA.py
 ┃ ┣ 📜dataset_generator.py
 ┃ ┣ 📜data_generation.ipynb
 ┃ ┗ 📜MICE.py
 ┣ 📂eda
 ┃ ┣ 📜CCFplot.py
 ┃ ┣ 📜EDA.ipynb
 ┃ ┗ 📜README.md
 ┣ 📂model
 ┃ ┣ 📜binary_ensemble.py
 ┃ ┣ 📜focal_loss_LGBM.py
 ┃ ┣ 📜README.md
 ┃ ┣ 📜svm.py
 ┃ ┣ 📜tabnet.py
 ┃ ┣ 📜train.py
 ┃ ┗ 📜tree_model.py
 ┣ 📂utils
 ┃ ┣ 📜focal_loss.py
 ┃ ┣ 📜get_acc_auroc.py
 ┃ ┗ 📜README.md
 ┣ 📜.gitignore
 ┣ 📜data_generation_2.ipynb
 ┣ 📜ensemble.ipynb
 ┣ 📜hyperparameter_tuning.ipynb
 ┣ 📜README.md
 ┣ 📜train.py
 ┗ 📜trainer.ipynb
            
```

## ⚒️Development Environment
- 서버 스펙 : AI Stage GPU (Tesla V100)
- 협업 툴 : Github / Zoom / Slack / Notion / Google Drive
- 기술 스택 : Python / Scikit-Learn / Scikit-Optimize / Pandas / Numpy
