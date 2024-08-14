### 프로젝트 개요
이 프로젝트는 조립식 PC 견적을 추천해주는 AI 기반 시스템을 구축하기 위한 것입니다.
Django 프레임워크를 활용하여 서버를 구현했으며, AI 모델은 다양한 PC 부품의 호환성과 가격 범주를 고려해 최적의 조합을 추천해줍니다.
이 시스템은 주기적으로 업데이트되는 데이터베이스를 기반으로 학습되며, 사용자에게 맞춤형 PC 구성을 제공하는 것을 목표로 합니다.

### 주요 기능
- **AI 기반 PC 견적 추천:** 사용자의 예산과 선호 부품을 입력받아, 빅데이터를 활용한 최적의 PC 구성을 추천합니다.
- **데이터 업데이트 및 재학습:** 주기적으로 업데이트되는 데이터베이스를 통해 모델을 지속적으로 학습시켜 최신 부품 정보를 반영합니다.
- **호환성 검증 및 커뮤니티 검증:** 추천된 PC 구성은 호환성 검증을 거친 후, 커뮤니티 공유를 통해 이중 검증을 받습니다.

### 기술 스택
- **Backend:** Django, Django REST Framework (DRF)
- **Frontend:** Vue.js
- **AI/ML:** Python, NumPy, Pandas, Scikit-learn
- **Database:** SQLite

### 설치 및 실행 방법

#### 1. 프로젝트 클론 및 설정
```bash
git clone https://github.com/HARl803/recommend-service.git
cd recommend-service/
git checkout develop
```

#### 2. 가상환경 설정
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

#### 3. 데이터베이스 생성 및 초기 데이터 로드
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py loaddata pc.json
```

#### 4. 모델 학습 (필요 시)
```bash
python estimate/train_model.py
```

#### 5. 서버 실행
```bash
python manage.py runserver
```

### API 명세서

#### AI 견적 추천 API
- **URL:** `http://127.0.0.1:8000/estimate/estimate/`
- **Method:** POST
- **Headers:** `{ 'Content-Type': 'application/json' }`
- **Request Body:**
    ```json
    {
        "costScope": "120~140만원",
        "cpuCodeBefore": "10000008",
        ...
    }
    ```
- **Response 예시:**
    ```json
    [
        {
            "cpuCodeAfter": "10000008",
            "cpuNameAfter": "인텔 코어 i5-11600K",
            ...
            "totalCost": "1885000"
        }
    ]
    ```

### 주요 특징
- **빅데이터 활용:** 검증된 PC 부품 조합 데이터베이스를 통해 사용자에게 최적의 견적을 제안합니다.
- **호환성 및 신뢰성:** AI 모델의 추천 결과는 자동화된 호환성 검증과 커뮤니티 피드백을 통해 신뢰성을 보장합니다.
- **확장 가능성:** 새로운 부품이 추가되거나, 가격 변동이 발생하면 시스템이 자동으로 학습하여 최신 정보를 반영합니다.

### 향후 발전 방향
- **고급 추천 기능:** 사용자의 선호도를 더 세밀하게 반영하는 고급 AI 추천 알고리즘을 개발할 계획입니다.
- **더 많은 데이터셋:** 데이터가 더 많아지고, 서버 성능이 발전 됨에 따라 적용되는 모델을 바꿀 예정입니다.

### 개발자 문서
프로젝트의 내부 실행 과정, 코드 구조, 및 기타 기술적 세부 사항은 [개발자 문서](./DEVELOPER_GUIDE.md)를 참조하세요.

### 기여 방법
이 프로젝트에 기여하고자 한다면, [GitHub 저장소](https://github.com/HARl803/recommend-service)를 방문해 Issues를 확인하고 Pull Request를 제출해 주세요. 모든 기여는 환영합니다!