## 데이터 분석 및 모델링 과제 개요

이 과제는 **KDX 한국데이터거래소**에서 제공하는 데이터를 활용하여 **서울 지역 일별 카드 소비현황**을 예측한다.

실제 업무 환경에서는 **DW 및 Hadoop에서 SQL을 사용해 데이터를 추출**하는 것부터 시작한다. 그러나 해당 데이터를 사용할 수 없어 KDX 한국데이터거래소를 통해 제공되는 NH농협카드의 일별 소비 데이터를 사용한다.


### 환경 설정
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

print(f'numpy version: {np.__version__}')
print(f'pandas version: {pd.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'matplotlib version: {mpl.__version__}')

plt.style.use('ggplot')
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
```

---

### 데이터 로드
데이터는 **2020년 1월부터 2024년 6월**까지의 일별 소비 데이터를 포함한다. 각 월별 CSV 파일로 제공되며 데이터를 하나의 DataFrame으로 통합한다.

```python
# 데이터 로드 및 통합
bas_ym = pd.date_range(start='20200101', end='20240630', freq='MS').strftime('%Y%m').tolist()

df = pd.DataFrame()

for i, var in enumerate(bas_ym):
    data_path = f'data/[NH농협카드] 일자별 소비현황_서울_{var}.csv'
    
    encodings = ['utf-8-sig', 'euc-kr', 'cp949']
    for encoding in encodings:
        try:
            tmp_df = pd.read_csv(data_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Failed to read {data_path} with available encodings.")
    
    # Ensure no duplicate columns
    tmp_df = tmp_df.loc[:, ~tmp_df.columns.duplicated()]
    
    df = pd.concat([df, tmp_df], axis=0)
    
print(df.shape)
print(df.head())
```

#### 주요 처리 단계:
1. **파일 읽기**: 월별 데이터를 읽어와 결합합니다.
2. **인코딩 처리**: 다양한 인코딩 형식을 시도(`utf-8-sig`, `euc-kr`, `cp949`)하여 파일을 안정적으로 로드합니다.
3. **중복 열 제거**: 데이터 중복을 방지하기 위해 열 중복을 제거합니다.
4. **데이터 통합**: 모든 파일 데이터를 하나의 DataFrame으로 결합합니다.

### 데이터 설명

#### 1. 데이터 출처
- **KDX 한국데이터거래소**
    - **데이터명**: [NH농협카드] 일자별 소비현황_서울
    - **URL**: [KDX 데이터 상세](https://kdx.kr/data/product-list?specs_id=MA38230007&corp_id=CORP000024&category_id=CA000004)

#### 2. 데이터 개요
- **분석 기간**: 2020년 1월 ~ 2024년 6월
- **지역 범위**: 서울특별시
- **내용**: NH농협카드의 일별 소비 금액 및 관련 지표
- **형식**: 월별 CSV 파일로 제공


### 분석 결과

![picture 0](images/49a21fa26365104b0e4fc04bb6c524810d922c40fcf9009ca24359372ee791f6.png)  

![picture 1](images/6bfa6151f96af81879cc16e9e8f5290e64bb06e49d9911475a832a5059f71c43.png)  

![picture 2](images/039fa079010e43b530c67b20165559757c48c2bb519ea171b9f6b5ca13415c54.png)  

![picture 3](images/cd0d2f260ca3d2ac3bb27ccdd8decc1eec911bdfb490ec2906a38da92f108489.png)  

![picture 4](images/fb049c4de0ed8870a00c8c76ef8a266a69118fcc06f2b242d7e2e0ad82780aeb.png)  
