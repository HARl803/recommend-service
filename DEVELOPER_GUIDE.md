# 조립식 PC 구성 추천 기능

## 1. 목표
- **목적**: 사용자 예산과 일부 부품 정보를 기반으로 나머지 부품을 추천하여 최적의 조립식 PC 구성을 제공하는 것.
- **기대 성능**: 추천 정확도 85% 이상, 처리 속도 1초 이내.
- **접근**: 단순한 필터링 기법을 사용하지 않고, 머신러닝 앙상블 기법과 딥러닝 분류 모델을 통해 높은 성능의 맞춤형 추천 시스템을 구현.

## 2. 프로젝트 접근의 계기
- **이전 프로젝트: 필터링과 GPT API 기반 추천 기능**
    - **필터링 기법**: 특정 조건을 기준으로 데이터를 필터링하여 추천을 제공하는 방식으로 사용되었으나, 복잡한 요구사항을 반영하는 데 한계가 있었음.
    - **GPT API 기반 추천**: 사전 학습된 정보를 바탕으로 추천을 제공하였으나, 새로운 데이터나 상황에 대한 적응력이 떨어짐.
- **필요성 인식**: 이 프로젝트에서는 이러한 한계를 극복하기 위해, 보다 정교하고 유연한 모델을 탐색하게 되었고, 앙상블 기법과 딥러닝 모델을 활용하는 방향으로 전환.

## 3. 데이터 특성 및 처리
### 3.1. 데이터 특성
- **조립식 컴퓨터 구성**: 
    - 조립식 컴퓨터는 총 8개의 부품으로 구성됨: CPU, Motherboard, Memory, GPU, SSD, Case, Power, Cooler.
    - 각 부품에 대해 부품 코드, 부품명, 부품 가격이 제공됨.
- **데이터셋 크기**:
    - **초기 데이터셋**: 10,000개.
    - **이후 데이터셋**: 500,000개 이상.

### 3.2. 데이터 수집
- **크롤링**: 
    - 웹 크롤링을 통해 조립식 PC 관련 부품 데이터를 수집.
    - 크롤링 작업은 자동화되어 있으며, 수집된 데이터는 JSON 형식으로 저장.
    - 데이터 신뢰성을 위해 검증 절차를 도입.
- **더미 데이터**:
    - 크롤링 이전에는 GPT-4를 활용하여 JSON 형식의 적절한 더미 데이터를 생성하여 사용.

### 3.3. 데이터 저장
- **Sqlite**:
    - 수집된 데이터를 SQLite 데이터베이스에 저장.
    - 필요한 경우 데이터를 덤프(dump)하여 백업 및 관리.

### 3.4. 데이터 가공 및 전처리

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from estimate.models import PC

# 코드 내부에 주석 추가
def process_data():
    # 데이터 로드: Django 모델에서 모든 PC 데이터를 가져옴
    pcs = PC.objects.all().values()
    df = pd.DataFrame(pcs)  # 데이터를 판다스 데이터프레임으로 변환

    # 데이터 가공: 각 부품의 비용을 합산하여 총 비용 계산
    part_columns = ['cpu', 'motherboard', 'memory', 'gpu', 'ssd', 'case', 'power', 'cooler']
    cost_columns = [f'{part}Cost' for part in part_columns]
    df['totalCost'] = df[cost_columns].sum(axis=1)

    # 데이터 가공: 예산 범위 설정 (costScope)
    df['costScope'] = pd.cut(
        df['totalCost'],
        bins=[-np.inf, 500000, 1000000, 1500000, 2000000, np.inf],  # 예산 범위 설정
        labels=['~50만원', '50~100만원', '100~150만원', '150~200만원', '200만원 이상']  # 각 범위에 대한 라벨 지정
    )
    ...

    # 데이터 인코딩: OneHotEncoder를 사용하여 범주형 데이터를 수치형 데이터로 변환
    before_columns = ['costScope'] + [f'{part}{attr}Before' for part in part_columns for attr in ['Code', 'Name', 'Cost']]
    encoder = OneHotEncoder(handle_unknown='ignore')  # OneHotEncoder 객체 생성
    X_encoded = encoder.fit_transform(df[before_columns])  # Before 데이터를 원핫인코딩

    return df, encoder
```

## 4. 모델 학습 및 평가
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import joblib

def train_model():
    df, encoder = process_data()  # 데이터 전처리 및 인코딩

    part_columns = ['cpu', 'motherboard', 'memory', 'gpu', 'ssd', 'case', 'power', 'cooler']
    X = df.drop(columns=[f'{part}{attr}After' for part in part_columns for attr in ['Code', 'Name', 'Cost']] + ['totalCost'])
    y = df[[f'{part}{attr}After' for part in part_columns for attr in ['Code', 'Name', 'Cost']] + ['totalCost']]

    X = X.fillna(-1)  # 결측치 처리

    # 종속 변수 인코딩
    label_encoders = {col: LabelEncoder() for col in y.columns}
    for col, le in label_encoders.items():
        y.loc[:, col] = le.fit_transform(y[col].astype(str))

    # 서비스 초기 : RandomForest 사용
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42))

    # 서비스 중기 : XGB 사용
    # model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=200, max_depth=15, random_state=42))

    # 서비스 발전 후 : MLP 사용
    # (코드 작성 예정)

    model.fit(X, y)  # 모델 학습

    kf = KFold(n_splits=5)  # 5-폴드 교차 검증
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    print(f'Cross-validation scores: {cv_scores}')  # 교차 검증 결과 출력
    ...

    # 모델 및 인코더 저장
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    joblib.dump((model, encoder, label_encoders), model_path)
```

## 5. 결과 및 향후 개선 방안
- **초기 서비스 결과**:
    - RandomForest를 사용하여 초기 단계에서도 높은 정확도의 추천 시스템을 구축.
    - 사용자의 예산과 부품 정보를 기반으로 맞춤형 추천을 제공.
- **중기 서비스 결과**:
    - 데이터가 증가하면서 XGBoost를 적용하여 예측 성능을 더욱 향상.
    - 사용자 맞춤형 추천의 정확성을 높임.
- **향후 계획**:
    - 서비스 발전 후에는 딥러닝 기반의 MLP 모델을 적용하여, 더욱 복잡한 데이터 패턴을 학습하고 고성능의 추천 시스템을 구현할 계획.
    - 실시간 추론 기능을 추가하여, 사용자 인터랙션 속도를 높일 예정.
    - 데이터의 지속적인 수집과 분석을 통해 모델을 고도화하고, 사용자의 요구에 더욱 정확하게 부응할 수 있도록 개선.

## 6. 키워드
- **하이퍼 파라미터**: 모델의 성능을 조정하기 위한 중요한 변수.
- **선형/비선형 관계**: 데이터의 관계를 분석할 때 사용하는 방법.
- **과적합**: 모델이 학습 데이터에 지나치게 적합하여 일반화 성능이 떨어지는 현상.
- **L1/L2 정규화**: 모델의 복잡도를 줄이기 위한 방법.
- **결정트리**: 의사결정 과정을 트리 구조로 나타내는 머신러닝 알고리즘.
- **시계열 데이터**: 시간의 흐름에 따라 수집된 데이터.
- **시퀀스 데이터**: 순서가 중요한 데이터, 예를 들어 텍스트 데이터.
- **그래프 구조 데이터**: 노드와 엣지로 이루어진 데이터를 나타냄.
