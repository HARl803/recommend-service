from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from . import train_model
import pandas as pd
import joblib
import os
from .serializers import PCEstimateSerializer

def index(request):
    pass

# 전처리 데이터 확인
def preview_data(request):
    df, encoder = train_model.process_data()
    sample_data = df.head(10).to_dict(orient='records')
    return JsonResponse(sample_data, safe=False)

# 인코더 및 모델 로드
model_path = os.path.join(os.path.dirname(__file__), 'model_randomforest.pkl')
model, encoder, label_encoders = joblib.load(model_path)


@api_view(['POST'])
def pc_estimate(request):
    serializer = PCEstimateSerializer(data=request.data)
    if serializer.is_valid():
        data = serializer.validated_data
        part_columns = ['cpu', 'motherboard', 'memory', 'gpu', 'ssd', 'case', 'power', 'cooler']

        # 입력 데이터를 딕셔너리로 생성
        input_data = {'costScope': data.get('costScope', 'nan')}
        for part in part_columns:
            input_data[f'{part}CodeBefore'] = data.get(f'{part}CodeBefore', 'nan')
            input_data[f'{part}NameBefore'] = data.get(f'{part}NameBefore', 'nan')
            input_data[f'{part}CostBefore'] = data.get(f'{part}CostBefore', 'nan')

        # 원핫 인코딩을 위한 데이터프레임 생성
        input_df = pd.DataFrame([input_data] * 10 )

        # 원핫 인코딩
        encoded_input = encoder.transform(input_df[['costScope'] + [f'{part}{attr}Before' for part in part_columns for attr in ['Code', 'Name', 'Cost']]])
        encoded_df = pd.DataFrame(encoded_input.toarray(), columns=encoder.get_feature_names_out(['costScope'] + [f'{part}{attr}Before' for part in part_columns for attr in ['Code', 'Name', 'Cost']]))
    
        # 'nan' 문자열이 포함된 열에 대해 처리
        for column in encoded_df.columns:
            if '_nan' in column:
                part_name = column.split('_')[0]
                encoded_df[column] = input_df.apply(lambda x: 1.0 if x[part_name] == 'nan' else 0.0, axis=1)

        # 원래 데이터프레임과 결합
        input_df = input_df.drop(columns=['costScope'] + [f'{part}{attr}Before' for part in part_columns for attr in ['Code', 'Name', 'Cost']])
        input_df = pd.concat([input_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # 결측치 처리
        input_df = input_df.fillna(-1)

        # 모델 넣기 전 인코딩 된 데이터 확인
        # return Response(input_df, status=status.HTTP_200_OK)

        # 모델 예측
        X = input_df.values
        y_preds = model.predict(X)

        # 종속변수 디코딩
        results = []
        for y_pred in y_preds:
            y_pred_df = pd.DataFrame(y_pred.reshape(1, -1), columns=[f'{part}{attr}After' for part in part_columns for attr in ['Code', 'Name', 'Cost']] + ['totalCost'])
            for col, le in label_encoders.items():
                y_pred_df[col] = le.inverse_transform(y_pred_df[col].astype(int))
            result = y_pred_df.to_dict(orient='records')[0]
            results.append(result)

        return Response(results, status=status.HTTP_200_OK) # 최대 5대의 PC 추천 결과 반환
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
