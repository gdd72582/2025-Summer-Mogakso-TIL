# 6주차: AICE Associate 샘플문항 풀이
## 내비게이션 주행데이터를 이용한 도착시각 예측 문제

---

## 📋 시험 환경 정보

### 허용된 오픈북 사이트
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- XGBoost

### 복사 가능한 단어 리스트
| 문제 | 유형 | 복사 가능 항목 |
|------|------|----------------|
| 3 | 변수 | A0007IT.json, signal.csv, df_a, df_b |
| 4 | 컬럼명 | Address1 |
| 5 | 컬럼명 | Time_Driving, Speed_Per_Hour |
| 6 | 변수명 | RID, df_temp |
| 7 | 변수명 | df_na |
| 8 | 컬럼명 | Time_Departure, Time_Arrival, df_del |
| 9 | 변수 | df_preset |

---

## 📊 데이터 컬럼 설명

### A0007IT.json 파일 컬럼
- **Time_Departure**: 출발시각
- **Time_Arrival**: 도착시각
- **Distance**: 이동 거리 (m)
- **Time_Driving**: 실주행시간 (초)
- **Speed_Per_Hour**: 평균시속
- **Address1**: 주소1
- **Address2**: 주소2
- **Weekday**: 출발시각의 요일
- **Hour**: 출발시각의 시각
- **Day**: 출발시각의 날짜

### signal.csv 파일 컬럼
- **RID**: 경로ID
- **Signaltype**: 경로의 신호등 갯수

---

## 🔧 1단계: 라이브러리 임포트 및 데이터 로딩

### 문제 1: scikit-learn 패키지 임포트
**문제:** scikit-learn 패키지는 머신러닝 교육을 위한 최고의 파이썬 패키지입니다. scikit-learn를 별칭(alias) sk로 임포트하는 코드를 작성하고 실행하세요.

```python
import sklearn as sk
```

### 문제 2: Pandas 패키지 임포트
**문제:** Pandas는 데이터 분석을 위해 널리 사용되는 파이썬 라이브러리입니다. Pandas를 사용할 수 있도록 별칭(alias)을 pd로 해서 불러오세요.

```python
import pandas as pd
```

### 문제 3: 데이터 파일 읽기 및 병합
**문제:** 모델링을 위해 분석 및 처리할 데이터 파일을 읽어오려고 합니다. Pandas함수로 2개 데이터 파일을 읽고 합쳐서 1개의 데이터프레임 변수명 df에 할당하는 코드를 작성하세요.

**세부 요구사항:**
- A0007IT.json 파일을 읽어 데이터 프레임 변수명 df_a에 할당
- signal.csv 파일을 읽어 데이터 프레임 변수명 df_b에 할당
- df_a와 df_b 데이터프레임을 판다스의 merge 함수를 활용하여 합쳐 데이터프레임 변수명 df에 저장
- 합칠때 사용하는 키(on): 'RID'
- 합치는 방법(how): 'inner'

```python
# JSON 파일 읽기
df_a = pd.read_json('A0007IT.json')

# CSV 파일 읽기
df_b = pd.read_csv('signal.csv')

# 두 데이터프레임 병합 (inner join on RID)
df = pd.merge(df_a, df_b, on='RID', how='inner')

# 결과 확인
print("병합된 데이터 크기:", df.shape)
print("컬럼 목록:", list(df.columns))
```

---

## 📈 2단계: 데이터 탐색 및 시각화

### 문제 4: Address1 분포 분석
**문제:** Address1(주소1)에 대한 분포도를 알아 보려고 합니다. Address1(주소1)에 대해 countplot그래프로 만드는 코드와 답안을 작성하세요.

**세부 요구사항:**
- Seaborn을 활용
- 첫번째, Address1(주소1)에 대해서 분포를 보여주는 countplot 그래프 그리기
- 두번째, 지역명이 없는 '-'에 해당되는 row(행)을 삭제
- 출력된 그래프를 보고 해석한 것으로 옳지 않은 선택지를 골라 '답안04' 변수에 저장

**해석 옵션:**
1. Countplot 그래프에서 Address1(주소1) 분포를 확인시 '경기도' 분포가 제일 크다.
2. Address1(주소1) 분포를 보면 '인천광역시' 보다 '서울특별시'가 더 크다.
3. 지역명이 없는 '-'에 해당되는 row(행)가 2개 있다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# matplotlib 한글 폰트 설정
import matplotlib.font_manager as fm
plt.rc('font', family='NanumGothicCoding')

# 1. Address1 분포 countplot 그리기
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Address1')
plt.title('Address1 분포')
plt.xticks(rotation=45)
plt.show()

# 2. '-' 값이 있는 행 확인
print("'-' 값이 있는 행의 개수:", len(df[df['Address1'] == '-']))

# 3. '-' 값이 있는 행 삭제
df_clean = df[df['Address1'] != '-']

# 4. 정리된 데이터로 다시 countplot 그리기
plt.figure(figsize=(12, 6))
sns.countplot(data=df_clean, x='Address1')
plt.title('Address1 분포 (정리된 데이터)')
plt.xticks(rotation=45)
plt.show()

# 5. 각 지역별 개수 확인
address_counts = df_clean['Address1'].value_counts()
print("각 지역별 개수:")
print(address_counts)

# 그래프 분석 결과에 따라 답안 선택
답안04 = 3  # 예시: 3번이 틀렸다고 가정
```

### 문제 5: 실주행시간과 평균시속 분포 분석
**문제:** 실주행시간과 평균시속의 분포를 같이 확인하려고 합니다. Time_Driving(실주행시간)과 Speed_Per_Hour(평균시속)을 jointplot그래프로 만드세요.

**세부 요구사항:**
- Seaborn을 활용
- X축에는 Time_Driving(실주행시간)을 표시하고 Y축에는 Speed_Per_Hour(평균시속)을 표시

```python
# Time_Driving과 Speed_Per_Hour의 jointplot 그리기
plt.figure(figsize=(10, 8))
sns.jointplot(data=df, x='Time_Driving', y='Speed_Per_Hour', 
              kind='scatter', height=8, ratio=3)
plt.suptitle('실주행시간과 평균시속의 관계', y=1.02)
plt.show()

# 추가 분석: 상관관계 확인
correlation = df['Time_Driving'].corr(df['Speed_Per_Hour'])
print(f"실주행시간과 평균시속의 상관계수: {correlation:.3f}")

# 이상치 확인 (시속 300 이상)
outliers = df[df['Speed_Per_Hour'] > 300]
print(f"시속 300 이상인 이상치 개수: {len(outliers)}")
```

---

## 🧹 3단계: 데이터 전처리

### 문제 6: 이상치 제거 및 불필요 컬럼 삭제
**문제:** 위의 jointplot 그래프에서 시속 300이 넘는 이상치를 발견할수 있습니다. 가이드에 따라서 전처리를 수행하고 저장하세요.

**세부 요구사항:**
- 대상 데이터프레임: df
- jointplot 그래프를 보고 시속 300 이상되는 이상치를 찾아 해당 행(Row)을 삭제
- 불필요한 'RID' 컬럼을 삭제
- 전처리 반영 후에 새로운 데이터프레임 변수명 df_temp에 저장

```python
# 1. 시속 300 이상인 이상치 제거
df_temp = df[df['Speed_Per_Hour'] <= 300]

# 2. RID 컬럼 삭제
df_temp = df_temp.drop('RID', axis=1)

# 3. 결과 확인
print("이상치 제거 후 데이터 크기:", df_temp.shape)
print("Speed_Per_Hour 범위:", df_temp['Speed_Per_Hour'].min(), "~", df_temp['Speed_Per_Hour'].max())
```

### 문제 7: 결측치 처리
**문제:** 모델링 성능을 제대로 얻기 위해서 결측치 처리는 필수입니다. 아래 가이드를 따라 결측치 처리하세요.

**세부 요구사항:**
- 대상 데이터프레임: df_temp
- 결측치를 확인하는 코드를 작성
- 결측치가 있는 행(row)을 삭제
- 전처리 반영된 결과를 새로운 데이터프레임 변수명 df_na에 저장
- 결측치 개수를 '답안07' 변수에 저장

```python
# 1. 결측치 확인
print("각 컬럼별 결측치 개수:")
print(df_temp.isnull().sum())

# 전체 결측치 개수 계산
total_missing = df_temp.isnull().sum().sum()
print(f"전체 결측치 개수: {total_missing}")

# 2. 결측치가 있는 행 삭제
df_na = df_temp.dropna()

# 3. 결과 확인
print("결측치 제거 후 데이터 크기:", df_na.shape)

# 4. 결측치 개수를 변수에 저장
답안07 = total_missing
print(f"답안07 = {답안07}")
```

### 문제 8: 불필요한 변수 삭제
**문제:** 모델링 성능을 제대로 얻기 위해서 불필요한 변수는 삭제해야 합니다. 아래 가이드를 따라 불필요 데이터를 삭제 처리하세요.

**세부 요구사항:**
- 대상 데이터프레임: df_na
- 'Time_Departure', 'Time_Arrival' 2개 컬럼을 삭제
- 전처리 반영된 결과를 새로운 데이터프레임 변수명 df_del에 저장

```python
# Time_Departure, Time_Arrival 컬럼 삭제
df_del = df_na.drop(['Time_Departure', 'Time_Arrival'], axis=1)

# 결과 확인
print("컬럼 삭제 후 데이터 크기:", df_del.shape)
print("남은 컬럼들:", list(df_del.columns))
```

### 문제 9: 원-핫 인코딩
**문제:** 원-핫 인코딩(One-hot encoding)은 범주형 변수를 1과 0의 이진형 벡터로 변환하기 위하여 사용하는 방법입니다. 원-핫 인코딩으로 아래 조건에 해당하는 컬럼 데이터를 변환하세요.

**세부 요구사항:**
- 대상 데이터프레임: df_del
- 원-핫 인코딩 대상: object 타입의 전체 컬럼
- 활용 함수: Pandas의 get_dummies
- 해당 전처리가 반영된 결과를 데이터프레임 변수 df_preset에 저장

```python
# object 타입 컬럼 확인
object_columns = df_del.select_dtypes(include=['object']).columns
print("원-핫 인코딩 대상 컬럼:", list(object_columns))

# 원-핫 인코딩 수행
df_preset = pd.get_dummies(df_del, columns=object_columns)

# 결과 확인
print("원-핫 인코딩 후 데이터 크기:", df_preset.shape)
print("원-핫 인코딩 후 컬럼 수:", len(df_preset.columns))
```

### 문제 10: 데이터셋 분리 및 스케일링
**문제:** 훈련과 검증 각각에 사용할 데이터셋을 분리하려고 합니다. Time_Driving(실주행시간) 컬럼을 label값 y로, 나머지 컬럼을 feature값 X로 할당한 후 훈련데이터셋과 검증데이터셋으로 분리하세요. 추가로, 가이드 따라서 훈련데이터셋과 검증데이터셋에 스케일링을 수행하세요.

**세부 요구사항:**
- 대상 데이터프레임: df_preset
- 훈련데이터셋 label: y_train
- 훈련데이터셋 Feature: X_train
- 검증데이터셋 label: y_valid
- 검증데이터셋 Feature: X_valid
- 훈련과 검증 데이터셋 비율: 80:20
- random_state: 42
- Scikit-learn의 train_test_split 함수 사용
- RobustScaler 스케일링 사용

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# 1. Feature와 Label 분리
X = df_preset.drop('Time_Driving', axis=1)  # Time_Driving 제외한 모든 컬럼
y = df_preset['Time_Driving']  # Time_Driving 컬럼

# 2. 훈련/검증 데이터셋 분리
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. RobustScaler 스케일링
scaler = RobustScaler()

# 훈련 데이터에 fit_transform 적용
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# 검증 데이터에 transform 적용
X_valid_scaled = scaler.transform(X_valid)
X_valid = pd.DataFrame(X_valid_scaled, columns=X_valid.columns)

# 결과 확인
print("훈련 데이터 크기:", X_train.shape)
print("검증 데이터 크기:", X_valid.shape)
print("훈련 라벨 크기:", y_train.shape)
print("검증 라벨 크기:", y_valid.shape)
```

---

## 🤖 4단계: 머신러닝 모델링

### 문제 11: 머신러닝 모델 생성
**문제:** Time_Driving(실주행시간)을 예측하는 머신러닝 모델을 만들려고 합니다. 결정 트리와 랜덤 포레스트는 독립변수 공간을 순차적으로 다양한 규칙을 적용하여 분할하는 방법으로 분류와 회귀 분석에 모두 사용할 수 있습니다. 아래 가이드에 따라 결정 트리와 랜덤 포레스트 모델을 생성하고 훈련하세요.

**세부 요구사항:**
- 결정 트리(decision tree):
  - 최대 트리 깊이(max_depth): 5
  - 노드 분할을 위한 최소 샘플 수(min_samples_split): 3
  - random_state: 120
  - 결정 트리 모델을 dt 변수에 저장
- 랜덤 포레스트(RandomForest):
  - 최대 트리 깊이(max_depth): 5
  - 노드 분할을 위한 최소 샘플 수(min_samples_split): 3
  - random_state: 120

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 결정 트리 모델 생성 및 훈련
dt = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=3,
    random_state=120
)
dt.fit(X_train, y_train)

# 2. 랜덤 포레스트 모델 생성 및 훈련
rf = RandomForestRegressor(
    max_depth=5,
    min_samples_split=3,
    random_state=120
)
rf.fit(X_train, y_train)

# 3. 모델 성능 평가
# 결정 트리 성능
dt_train_pred = dt.predict(X_train)
dt_valid_pred = dt.predict(X_valid)

dt_train_mse = mean_squared_error(y_train, dt_train_pred)
dt_valid_mse = mean_squared_error(y_valid, dt_valid_pred)
dt_train_r2 = r2_score(y_train, dt_train_pred)
dt_valid_r2 = r2_score(y_valid, dt_valid_pred)

print("=== 결정 트리 성능 ===")
print(f"훈련 MSE: {dt_train_mse:.2f}")
print(f"검증 MSE: {dt_valid_mse:.2f}")
print(f"훈련 R²: {dt_train_r2:.3f}")
print(f"검증 R²: {dt_valid_r2:.3f}")

# 랜덤 포레스트 성능
rf_train_pred = rf.predict(X_train)
rf_valid_pred = rf.predict(X_valid)

rf_train_mse = mean_squared_error(y_train, rf_train_pred)
rf_valid_mse = mean_squared_error(y_valid, rf_valid_pred)
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_valid_r2 = r2_score(y_valid, rf_valid_pred)

print("\n=== 랜덤 포레스트 성능 ===")
print(f"훈련 MSE: {rf_train_mse:.2f}")
print(f"검증 MSE: {rf_valid_mse:.2f}")
print(f"훈련 R²: {rf_train_r2:.3f}")
print(f"검증 R²: {rf_valid_r2:.3f}")
```

### 문제 12: 모델 성능 평가 (MAE)
**문제:** 위 의사결정나무(decision tree)와 랜덤포레스트(RandomForest) 모델의 성능을 평가하려고 합니다. 아래 가이드에 따라 예측 결과의 mae(Mean Absolute Error)를 구하고 평가하세요.

**세부 요구사항:**
- 성능 평가는 검증 데이터셋을 활용
- 11번 문제에서 만든 의사결정나무(decision tree) 모델로 y값을 예측하여 y_pred_dt에 저장
- 검증 정답(y_valid)과 예측값(y_pred_dt)의 mae를 구하고 dt_mae 변수에 저장
- 11번 문제에서 만든 랜덤포레스트(Random Forest) 모델로 y값을 예측하여 y_pred_rf에 저장
- 검증 정답(y_valid)과 예측값(y_pred_rf)의 mae를 구하고 rf_mae 변수에 저장
- 2개의 모델에 대한 mae 성능평가 결과를 확인하여 성능좋은 모델 이름을 '답안12' 변수에 저장

```python
from sklearn.metrics import mean_absolute_error

# 1. 결정 트리 모델 예측
y_pred_dt = dt.predict(X_valid)

# 2. 결정 트리 MAE 계산
dt_mae = mean_absolute_error(y_valid, y_pred_dt)

# 3. 랜덤 포레스트 모델 예측
y_pred_rf = rf.predict(X_valid)

# 4. 랜덤 포레스트 MAE 계산
rf_mae = mean_absolute_error(y_valid, y_pred_rf)

# 5. 결과 출력
print("=== 모델 성능 비교 (MAE) ===")
print(f"결정 트리 MAE: {dt_mae:.2f}")
print(f"랜덤 포레스트 MAE: {rf_mae:.2f}")

# 6. 더 좋은 성능의 모델 선택
if dt_mae < rf_mae:
    답안12 = 'decisiontree'
    print("더 좋은 모델: 결정 트리")
else:
    답안12 = 'randomforest'
    print("더 좋은 모델: 랜덤 포레스트")

print(f"답안12 = '{답안12}'")
```

---

## 🧠 5단계: 딥러닝 모델링

### 문제 13: 딥러닝 모델 생성 및 학습
**문제:** Time_Driving(실주행시간)을 예측하는 딥러닝 모델을 만들려고 합니다. 아래 가이드에 따라 모델링하고 학습을 진행하세요.

**세부 요구사항:**
- Tensorflow framework를 사용하여 딥러닝 모델을 만드세요
- 히든레이어(hidden layer) 2개이상으로 모델을 구성하세요
- dropout 비율 0.2로 Dropout 레이어 1개를 추가해 주세요
- 손실함수는 MSE(Mean Squared Error)를 사용하세요
- 하이퍼파라미터 epochs: 30, batch_size: 16 으로 설정해 주세요
- 각 에포크마다 loss와 metrics 평가하기 위한 데이터로 x_valid, y_valid 사용하세요
- 학습정보는 history 변수에 저장해 주세요

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 딥러닝 모델 구성
model = Sequential([
    # 입력층 (특성 수에 맞춤)
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    
    # 첫 번째 히든 레이어
    Dense(32, activation='relu'),
    
    # Dropout 레이어 (0.2 비율)
    Dropout(0.2),
    
    # 두 번째 히든 레이어
    Dense(16, activation='relu'),
    
    # 출력층 (회귀 문제이므로 1개 뉴런, 활성화함수 없음)
    Dense(1)
])

# 2. 모델 컴파일
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # 추가 메트릭으로 MAE도 확인
)

# 3. 모델 구조 확인
print("=== 딥러닝 모델 구조 ===")
model.summary()

# 4. Early Stopping 콜백 설정 (과적합 방지)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 5. 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping],
    verbose=1
)

# 6. 학습 결과 확인
print("\n=== 학습 완료 ===")
print(f"총 에포크: {len(history.history['loss'])}")
print(f"최종 훈련 MSE: {history.history['loss'][-1]:.2f}")
print(f"최종 검증 MSE: {history.history['val_loss'][-1]:.2f}")
```

### 문제 14: 딥러닝 모델 성능 시각화
**문제:** 위 딥러닝 모델의 성능을 평가하려고 합니다. Matplotlib 라이브러리 활용해서 학습 mse와 검증 mse를 그래프로 표시하세요.

**세부 요구사항:**
- 1개의 그래프에 학습 mse과 검증 mse 2가지를 모두 표시하세요
- 위 2가지 각각의 범례를 'mse', 'val_mse'로 표시하세요
- 그래프의 타이틀은 'Model MSE'로 표시하세요
- X축에는 'Epochs'라고 표시하고 Y축에는 'MSE'라고 표시하세요

```python
import matplotlib.pyplot as plt

# 1. 학습 과정 시각화
plt.figure(figsize=(10, 6))

# 학습 MSE와 검증 MSE 플롯
plt.plot(history.history['loss'], label='mse', color='blue')
plt.plot(history.history['val_loss'], label='val_mse', color='red')

# 그래프 설정
plt.title('Model MSE', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Y축 범위 설정 (0부터 시작)
plt.ylim(bottom=0)

plt.tight_layout()
plt.show()

# 2. 추가 분석: 모델 성능 비교
print("\n=== 최종 모델 성능 비교 ===")

# 딥러닝 모델 예측
y_pred_dl = model.predict(X_valid).flatten()
dl_mae = mean_absolute_error(y_valid, y_pred_dl)
dl_mse = mean_squared_error(y_valid, y_pred_dl)

print(f"결정 트리 MAE: {dt_mae:.2f}")
print(f"랜덤 포레스트 MAE: {rf_mae:.2f}")
print(f"딥러닝 모델 MAE: {dl_mae:.2f}")
print(f"딥러닝 모델 MSE: {dl_mse:.2f}")

# 3. 예측값 vs 실제값 시각화
plt.figure(figsize=(12, 4))

# 결정 트리
plt.subplot(1, 3, 1)
plt.scatter(y_valid, y_pred_dt, alpha=0.6)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title('결정 트리')

# 랜덤 포레스트
plt.subplot(1, 3, 2)
plt.scatter(y_valid, y_pred_rf, alpha=0.6)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title('랜덤 포레스트')

# 딥러닝
plt.subplot(1, 3, 3)
plt.scatter(y_valid, y_pred_dl, alpha=0.6)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title('딥러닝')

plt.tight_layout()
plt.show()
```

---

## 📊 프로젝트 요약

### 전체 워크플로우
1. **데이터 준비** (문제 1-3): 라이브러리 임포트, 데이터 로딩 및 병합
2. **데이터 탐색** (문제 4-5): 시각화를 통한 데이터 분포 분석, 이상치 탐지
3. **데이터 전처리** (문제 6-10): 이상치 제거, 결측치 처리, 불필요 변수 삭제, 범주형 변수 인코딩, 데이터셋 분리 및 스케일링
4. **머신러닝 모델링** (문제 11-12): 결정 트리, 랜덤 포레스트 모델 생성 및 성능 평가
5. **딥러닝 모델링** (문제 13-14): 신경망 모델 생성, 학습, 성능 시각화

### 핵심 학습 포인트
- **데이터 전처리**: 이상치 제거, 결측치 처리, 스케일링의 중요성
- **모델 비교**: 다양한 알고리즘의 성능 비교 및 평가
- **시각화**: 데이터 탐색과 모델 성능 분석을 위한 그래프 활용
- **실무 적용**: 실제 내비게이션 데이터를 활용한 예측 모델 구축

### 참고 자료
- [AICE Associate 자격증 준비 가이드](https://datawithu.tistory.com/m/34)
- [AICE ASSOCIATE 자격증 취득을 위한 총 정리](https://velog.io/@sy508011/AICE-ASSOCIATE-자격증-취득을-위한-총-정리)

