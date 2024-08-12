import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Django 모델에서 데이터 가져오기
from estimate.models import PC

def process_data():
    # 데이터 로드
    pcs = PC.objects.all().values() # Django 모델에서 모든 PC 데이터를 가져옴
    df = pd.DataFrame(pcs) # 데이터를 판다스 데이터프레임으로 변환
    
    part_columns = ['cpu', 'motherboard', 'memory', 'gpu', 'ssd', 'case', 'power', 'cooler']
    # 부품 및 비용 컬럼명 설정
    cost_columns = [f'{part}Cost' for part in part_columns]
    
    # totalCost 계산
    df['totalCost'] = df[cost_columns].sum(axis=1) # 각 부품의 비용을 합산하여 총 비용 계산

    # costScope 계산
    df['costScope'] = pd.cut(
        df['totalCost'],
        bins=[-np.inf, 500000, 1000000, 1500000, 2000000, np.inf], # 예산 범위를 설정
        labels=['~50만원', '50~100만원', '100~150만원', '150~200만원', '200만원 이상'] # 각 범위에 대한 라벨 지정
    )

    # 모든 조합의 Before 데이터 생성
    all_data = []
    base_data = df.to_dict(orient='records') # 데이터프레임을 딕셔너리 리스트로 변환
    for row in base_data:
        # 각 행의 After 데이터를 생성
        original_data = {f'{part}{attr}After': str(row[f'{part}{attr}']) for part in part_columns for attr in ['Code', 'Name', 'Cost']}
        original_data.update({k: str(v) for k, v in row.items()})
        for i in range(256):
            binary_str = format(i, '08b') # 8비트 이진수 문자열 생성
            temp_data = {f'{part}{attr}Before': (str(row[f'{part}{attr}']) if binary_str[j] == '1' else 'nan') 
                         for j, part in enumerate(part_columns) 
                         for attr in ['Code', 'Name', 'Cost']} # Before 데이터 생성
            temp_data.update(original_data)
            all_data.append(temp_data)

    df = pd.DataFrame(all_data) # 생성된 모든 데이터를 데이터프레임으로 변환

    # 필요한 컬럼만 선택하고, After 컬럼명을 변경
    selected_columns = ['costScope'] + \
        [f'{part}{attr}Before' for part in part_columns for attr in ['Code', 'Name', 'Cost']] + \
        [f'{part}{attr}After' for part in part_columns for attr in ['Code', 'Name', 'Cost']] + \
        ['totalCost']

    df = df[selected_columns] # 선택한 컬럼들로 데이터프레임 재구성

    # 원핫인코딩
    before_columns = ['costScope'] + [f'{part}{attr}Before' for part in part_columns for attr in ['Code', 'Name', 'Cost']]
    encoder = OneHotEncoder(handle_unknown='ignore') # OneHotEncoder 객체 생성
    X_encoded = encoder.fit_transform(df[before_columns]) # Before 데이터를 원핫인코딩
    
    # 인코딩된 데이터를 데이터프레임으로 변환
    X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(before_columns))
    
    # 인코딩된 데이터를 기존 데이터프레임과 병합
    df = df.drop(before_columns, axis=1) # 인코딩 전의 before 컬럼 제거
    df = pd.concat([df.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1) # 인코딩된 데이터프레임과 병합

    return df, encoder