from django.shortcuts import render
from django.http import JsonResponse
from . import train_model

def index(request):
    pass

# 전처리 데이터 확인
def preview_data(request):
    df, encoder = train_model.process_data()
    sample_data = df.head(10).to_dict(orient='records')
    return JsonResponse(sample_data, safe=False)