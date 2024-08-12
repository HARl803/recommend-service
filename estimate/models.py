from django.db import models

class PC(models.Model):
    mentorID = models.IntegerField()                    # 멘토 ID
    pcID = models.IntegerField()                        # PC ID
    
    cpuCode = models.PositiveIntegerField()             # 부품 코드번호 숫자 8자리
    cpuName = models.CharField(max_length=100)          # 부품 이름 (한국어)
    cpuCost = models.PositiveIntegerField()             # 부품 가격 (KRW)
    
    motherboardCode = models.PositiveIntegerField()     # 부품 코드번호 숫자 8자리
    motherboardName = models.CharField(max_length=100)  # 부품 이름 (한국어)
    motherboardCost = models.PositiveIntegerField()     # 부품 가격 (KRW)
    
    memoryCode = models.PositiveIntegerField()          # 부품 코드번호 숫자 8자리
    memoryName = models.CharField(max_length=100)       # 부품 이름 (한국어)
    memoryCost = models.PositiveIntegerField()          # 부품 가격 (KRW)
    
    gpuCode = models.PositiveIntegerField()             # 부품 코드번호 숫자 8자리
    gpuName = models.CharField(max_length=100)          # 부품 이름 (한국어)
    gpuCost = models.PositiveIntegerField()             # 부품 가격 (KRW)
    
    ssdCode = models.PositiveIntegerField()             # 부품 코드번호 숫자 8자리
    ssdName = models.CharField(max_length=100)          # 부품 이름 (한국어)
    ssdCost = models.PositiveIntegerField()             # 부품 가격 (KRW)
    
    caseCode = models.PositiveIntegerField()            # 부품 코드번호 숫자 8자리
    caseName = models.CharField(max_length=100)         # 부품 이름 (한국어)
    caseCost = models.PositiveIntegerField()            # 부품 가격 (KRW)
    
    powerCode = models.PositiveIntegerField()           # 부품 코드번호 숫자 8자리
    powerName = models.CharField(max_length=100)        # 부품 이름 (한국어)
    powerCost = models.PositiveIntegerField()           # 부품 가격 (KRW)
    
    coolerCode = models.PositiveIntegerField()          # 부품 코드번호 숫자 8자리
    coolerName = models.CharField(max_length=100)       # 부품 이름 (한국어)
    coolerCost = models.PositiveIntegerField()          # 부품 가격 (KRW)
    
    class Meta:
        unique_together = ('mentorID', 'pcID')          # 멘토 ID와 PC ID의 복합 유일성 제약조건

