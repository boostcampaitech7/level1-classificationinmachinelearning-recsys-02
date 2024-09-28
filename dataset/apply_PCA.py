import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class MakePCA:
  def __init__(self, df) -> None:
    self.df = df

  def Find_PCA_n(self, visualize = False):
    '''
    주성분 개수 정하는 함수
    visualize : 시각화 선택
    '''
    self.train_df = self.df[self.df._type == 'train'].drop(columns = '_type')
    self.test_df = self.df[self.df._type == 'test'].drop(columns = '_type')

    # 1. 데이터 준비(수치형 데이터만 선택)
    X = self.train_df.drop('target',axis=1).select_dtypes(include=[float, int])
    X.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    X.fillna(X.mean(), inplace=True)

    # 2. 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. PCA 모델 생성, 데이터에 적용
    pca = PCA()  # 주성분 개수를 지정하지 않고 전체를 사용
    pca.fit(X_scaled)

    # 4. 누적 설명 분산 비율 계산
    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    # 5. 90%에 도달하는 주성분 개수 계산
    num_components_90 = (cumulative_variance >= 0.90).argmax() + 1

    # 6. 시각화 선택
    if visualize == True:
      plt.figure(figsize=(8, 6))
      plt.plot(cumulative_variance, marker='o', linestyle='--')
      plt.xlabel('Number of Principal Components')
      plt.ylabel('Cumulative Explained Variance')
      plt.title('Cumulative Explained Variance vs. Number of Components')

      plt.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
      plt.axvline(x=(cumulative_variance >= 0.90).argmax(), color='r', linestyle='--')  # 90%에 도달하는 지점
      plt.legend(loc='best')
      plt.grid(True)
      plt.show()

    return num_components_90
  

  def ApplyPCA(self, num, visualize = False):
    ''' 
    PCA 적용 함수
    num : 주성분 개수
    visualize : 시각화 선택
    '''
    # 트레인과 테스트 데이터 준비 (가정: train_df, test_df)
    X_train = self.train_df.drop('target',axis=1).select_dtypes(include=[float, int])
    X_test = self.test_df.drop('target',axis=1).select_dtypes(include=[float, int])

    # 무한대 및 NaN 값 처리
    X_train.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    X_test.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA 모델 생성 및 트레인 데이터에 적합
    pca = PCA(n_components=num)
    X_train_pca = pca.fit_transform(X_train_scaled)  

    # 테스트 데이터에 PCA 변환 적용
    X_test_pca = pca.transform(X_test_scaled)  

    # PCA 결과를 데이터프레임으로 변환
    pca_train_df = pd.DataFrame(data=X_train_pca, columns=[f'PC{i+1}' for i in range(num)])
    pca_test_df = pd.DataFrame(data=X_test_pca, columns=[f'PC{i+1}' for i in range(num)])

    # 합치기
    additional_info_df = self.df[['ID', '_type', 'target']]
    combined_pca_df = pd.concat([pca_train_df, pca_test_df], ignore_index=True)
    final_df = pd.concat([additional_info_df.reset_index(drop=True), combined_pca_df], axis=1)

    # 시각화 선택
    if visualize == True:
      pca_means = final_df[final_df['_type']=='train'].iloc[:, 2:].groupby('target').mean()  
      plt.figure(figsize=(15, 10))

      for i in range(1, num+1):  
          plt.subplot(5, 6, i)  
          plt.bar(pca_means.index, pca_means[f'PC{i}'])
          plt.title(f'Average of PC{i}')
          plt.xlabel('Target')
          plt.ylabel('Average Value')

      plt.tight_layout()  
      plt.show()

    return final_df
  

