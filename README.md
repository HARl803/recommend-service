## 조립식 PC 구성 추천 기능

## **1. 목표**

- **목적**: 사용자 예산과 일부 부품 정보를 기반으로 나머지 부품을 추천하여 최적의 조립식 PC 구성을 제공하는 것.
- **접근**: 필터링 기법을 사용하지 않고, 머신러닝 앙상블 기법과 딥러닝 분류 모델을 통해 높은 성능의 맞춤형 추천 시스템을 구현.

## **2. 프로젝트 접근의 계기**

- **이전 프로젝트: 필터링과 GPT API 기반 추천 기능**
    - **필터링 기법**: 특정 조건을 기준으로 데이터를 필터링하여 추천을 제공하는 방식으로 사용되었으나, 복잡한 요구사항을 반영하는 데 한계가 있었음.
    - **GPT API 기반 추천**: 사전 학습된 정보를 바탕으로 추천을 제공하였으나, 새로운 데이터나 상황에 대한 적응력이 떨어짐.
- **필요성 인식**: 이러한 한계를 극복하기 위해, 보다 정교하고 유연한 모델을 탐색하게 되었고, 앙상블 기법과 딥러닝 모델을 활용하는 방향으로 전환.

## **3. 데이터 특성 및 처리**

### **3.1. 데이터 특성**

- **조립식 컴퓨터 구성**:
    - 조립식 컴퓨터는 총 8개의 부품으로 구성됨: CPU, Motherboard, Memory, GPU, SSD, Case, Power, Cooler.
    - 각 부품에 대해 부품 코드, 부품명, 부품 가격이 제공됨.
- **데이터셋 크기**:
    - **초기 데이터셋**: 10,000개.
    - **이후 데이터셋**: 500,000개 이상.

### **3.2. 데이터 수집**

- **크롤링**:
    - 웹 크롤링을 통해 조립식 PC 관련 부품 데이터를 수집.
    - 크롤링 작업은 자동화되어 있으며, 수집된 데이터는 JSON 형식으로 저장.
- 더미(크롤링 이전):
    - GPT 4o에게 JSON 형식을 주고, 적절한 더미 생성

### **3.3. 데이터 저장**

- **Sqlite**:
    - 수집된 데이터를 SQLite 데이터베이스에 저장.
    - 필요한 경우 데이터를 덤프(dump)하여 백업 및 관리.

### **3.4. 데이터 처리 및 코드 설명**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from estimate.models import PC

def process_data():
    # 데이터 로드
    pcs = PC.objects.all().values()  # Django 모델에서 모든 PC 데이터를 가져옴
    df = pd.DataFrame(pcs)  # 데이터를 판다스 데이터프레임으로 변환

    # 부품 및 비용 컬럼명 설정
    part_columns = ['cpu', 'motherboard', 'memory', 'gpu', 'ssd', 'case', 'power', 'cooler']
    cost_columns = [f'{part}Cost' for part in part_columns]

    # 데이터 가공: 각 부품의 비용을 합산하여 총 비용 계산
    df['totalCost'] = df[cost_columns].sum(axis=1)

    # 데이터 가공: 예산 범위 설정 (costScope)
    df['costScope'] = pd.cut(
        df['totalCost'],
        bins=[-np.inf, 500000, 1000000, 1500000, 2000000, np.inf],  # 예산 범위 설정
        labels=['~50만원', '50~100만원', '100~150만원', '150~200만원', '200만원 이상']  # 각 범위에 대한 라벨 지정
    )

    # 데이터 처리: 모든 조합의 Before 데이터 생성
    all_data = []
    base_data = df.to_dict(orient='records')  # 데이터프레임을 딕셔너리 리스트로 변환
    for row in base_data:
        # After 데이터 생성
        original_data = {f'{part}{attr}After': str(row[f'{part}{attr}']) for part in part_columns for attr in ['Code', 'Name', 'Cost']}
        original_data.update({k: str(v) for k, v in row.items()})
        for i in range(256):
            binary_str = format(i, '08b')  # 8비트 이진수 문자열 생성
            temp_data = {f'{part}{attr}Before': (str(row[f'{part}{attr}']) if binary_str[j] == '1' else 'nan')
                         for j, part in enumerate(part_columns)
                         for attr in ['Code', 'Name', 'Cost']}
            temp_data.update(original_data)
            all_data.append(temp_data)

    df = pd.DataFrame(all_data)  # 생성된 모든 데이터를 데이터프레임으로 변환

    # 필요한 컬럼만 선택하고, After 컬럼명을 변경
    selected_columns = ['costScope'] + \\
        [f'{part}{attr}Before' for part in part_columns for attr in ['Code', 'Name', 'Cost']] + \\
        [f'{part}{attr}After' for part in part_columns for attr in ['Code', 'Name', 'Cost']] + \\
        ['totalCost']

    df = df[selected_columns]  # 선택한 컬럼들로 데이터프레임 재구성

    # 데이터 인코딩: OneHotEncoder를 사용하여 범주형 데이터를 수치형 데이터로 변환
    before_columns = ['costScope'] + [f'{part}{attr}Before' for part in part_columns for attr in ['Code', 'Name', 'Cost']]
    encoder = OneHotEncoder(handle_unknown='ignore')  # OneHotEncoder 객체 생성
    X_encoded = encoder.fit_transform(df[before_columns])  # Before 데이터를 원핫인코딩

    # 인코딩된 데이터를 데이터프레임으로 변환 및 병합
    X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(before_columns))
    df = df.drop(before_columns, axis=1)  # 인코딩 전의 before 컬럼 제거
    df = pd.concat([df.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)  # 인코딩된 데이터프레임과 병합

    return df, encoder

```

- **목적**: 조립식 PC 구성 요소 데이터를 전처리하고, 부품 정보와 가격을 기반으로 데이터를 가공 및 인코딩하여 모델 학습에 적합한 형식으로 준비.
- **설명**:
    - **데이터 로드**: Django 모델에서 PC 부품 데이터를 가져와 Pandas 데이터프레임으로 변환.
    - **데이터 가공**: 부품 비용을 합산하여 `totalCost`를 계산하고, 예산 범위(`costScope`)를 설정함. 이 범위는 사용자의 예산에 따라 라벨을 지정함.
    - **모든 조합의 Before 데이터 생성**: 다양한 조합의 부품 구성을 고려하여 각 조합의 데이터를 생성함.
    - **데이터 인코딩**: OneHotEncoder를 사용하여 범주형 데이터를 수치형으로 변환하고, 이를 데이터프레임에 병합하여 학습에 적합한 형태로 준비.

## **4. 모델 학습 및 평가**

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

    # 서비스 중기 : XGB 사용 (코드 주석)
    # model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=200, max_depth=15, random_state=42))

    # 서비스 발전 후 : MLP 사용 (코드 작성 예정)

    model.fit(X, y)  # 모델 학습

    kf = KFold(n_splits=5)  # 5-폴드 교차 검증
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared

_error')
    print(f'Cross-validation scores: {cv_scores}')

    # 모델 및 인코더 저장
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    joblib.dump((model, encoder, label_encoders), model_path)

```

- **목적**: 데이터를 학습시키고, 모델을 평가하여 최적의 모델을 선택.
- **설명**:
    - `train_model` 함수는 데이터 전처리 및 인코딩 후, 모델 학습을 수행함.
    - 초기 단계에서는 `RandomForestRegressor`를 사용하여, 적은 데이터로도 높은 성능을 발휘할 수 있음.
    - 중기 이후에는 XGBoost 및 MLP와 같은 모델을 사용하여, 데이터가 많아질수록 성능을 극대화할 계획.
    - 학습된 모델과 인코더를 저장하여, 나중에 모델을 로드하고 사용 가능.

### **모델 실행 및 평가**

```python
# 모델 학습 및 저장을 위한 함수 호출
if __name__ == '__main__':
    train_model()

```

- **목적**: 전체 코드를 실행하여 모델을 학습시키고, 교차 검증 결과를 출력하며, 학습된 모델을 저장.

## **5. 결과 및 향후 개선 방안**

- **초기 서비스 결과**:
    - RandomForest를 사용하여 초기 단계에서도 높은 정확도의 추천 시스템을 구축.
    - 사용자의 예산과 부품 정보를 기반으로 맞춤형 추천을 제공.
- **중기 서비스 결과**:
    - 데이터가 증가하면서 XGBoost를 적용하여 예측 성능을 더욱 향상.
    - 사용자 맞춤형 추천의 정확성을 높임.
- **향후 계획**:
    - 서비스 발전 후에는 딥러닝 기반의 MLP 모델을 적용하여, 더욱 복잡한 데이터 패턴을 학습하고 고성능의 추천 시스템을 구현할 계획.
    - 데이터의 지속적인 수집과 분석을 통해 모델을 고도화하고, 사용자의 요구에 더욱 정확하게 부응할 수 있도록 개선.



---
### 알아 둘 키워드 (면접 대비)
- 하이퍼 파라미터
- 선형/비선형 관계
- 과적합
- L1/L2 정규화
- 가중치 계산하는법
- 결정트리
- 시계열 데이터
- 시퀀스 데이터
- 그래프 구조 데이터