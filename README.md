# 박막 두께 예측 프로젝트

반사율 스펙트럼으로부터 네 개 층의 두께를 예측하는 회귀 프로젝트입니다. 반사율은 반사광 세기를 입사광 세기로 나눈 값입니다.

## 데이터

| 항목 | 내용 |
|---|---|
| 학습 데이터 | 810,000개 샘플, 230개 열 |
| 테스트 데이터 | 10,000개 샘플, 227개 열 |
| 입력 | 반사율 채널 `0`~`225`, 총 226개 |
| 예측 대상 | `layer_1`~`layer_4` |
| 학습 데이터의 두께 값 범위 | 각 층 10~300 |

채널 번호에 대응하는 실제 파장과 두께의 단위는 제공된 CSV만으로 확인되지 않았습니다.

## 현재 결과

### Ridge + PCA 모델

`model_training_final.py`는 전체 학습 데이터를 표준화하고 PCA로 226개 특성을 120개로 축소한 뒤, 각 층에 Ridge 회귀 모델을 학습합니다. 저장된 PCA의 설명 분산비는 약 99.92%입니다.

| 층 | 학습 RMSE | 학습 MAE | 학습 R² |
|---|---:|---:|---:|
| layer_1 | 86.19 | 75.22 | 0.0084 |
| layer_2 | 86.39 | 74.89 | 0.0038 |
| layer_3 | 86.41 | 75.04 | 0.0033 |
| layer_4 | 84.28 | 73.44 | 0.0519 |
| 평균 | 85.82 | 74.65 | 0.0168 |

위 수치는 이전 실행에서 출력된 **학습 데이터 기준** 결과입니다. R²가 0에 가까워 평균값 예측보다 크게 개선되지 않았으며, 독립적인 검증 또는 테스트 성능은 확인되지 않았습니다.

스크립트는 앞의 100,000개 샘플에 대해 같은 데이터를 학습과 평가에 모두 사용해 `alpha`를 선택합니다. 저장된 모델의 `alpha`는 모두 0.1이지만, 이를 독립 검증으로 확인된 최적값이라고 볼 수 없습니다.

### 기본 Lasso 모델

`model_training.py`는 학습 데이터에서 무작위로 50,000개를 뽑아 Ridge와 Lasso를 비교한 뒤, 각 층에 선택된 모델을 재학습합니다. 저장된 모델은 네 층 모두 Lasso입니다.

| 층 | 학습 RMSE | 학습 MAE | 학습 R² | 검증 RMSE |
|---|---:|---:|---:|---:|
| layer_1 | 86.15 | 75.15 | 0.0119 | 86.55 |
| layer_2 | 86.30 | 74.80 | 0.0073 | 86.61 |
| layer_3 | 86.28 | 74.94 | 0.0073 | 86.90 |
| layer_4 | 84.07 | 73.23 | 0.0509 | 84.35 |

이 검증 수치는 이전 실행 결과이며, 모델 선택에도 사용되었습니다. 표준화가 학습·검증 분리 전에 수행되어 검증 데이터의 정보가 전처리에 포함된 한계가 있습니다. 두 모델의 학습 수치를 서로 다른 데이터에서 계산했으므로 이를 근거로 어느 제출 파일이 더 좋다고 판단할 수 없습니다.

## 파일

- `data/train.csv`: 두께와 반사율이 포함된 학습 데이터
- `data/test.csv`: 예측할 반사율과 ID
- `data/sample_submission.csv`: 제출 형식 예시
- `submission.csv`: 기본 모델의 테스트 예측
- `submission_final.csv`: Ridge + PCA 모델의 테스트 예측
- `models.pkl`, `scaler.pkl`: 기본 모델과 표준화 도구
- `models_final.pkl`: Ridge 모델, 표준화 도구, PCA가 함께 저장된 파일
- `model_training.py`, `model_training_final.py`: 모델 학습 및 예측 스크립트
- `analysis.py`: 데이터 통계 및 그래프 생성
- `feature_analysis.py`: 첫 번째 층의 Ridge 계수를 이용한 특성 분석
- `output_01_target_distribution.png`: 층별 두께 분포
- `output_02_spectrum_samples.png`: 학습 데이터의 처음 10개 반사율 스펙트럼
- `output_03_spectrum_statistics.png`: 채널별 반사율 평균과 표준편차
- `.venv/`: 프로젝트 가상환경



## 사용 방법

학습 및 분석에는 `numpy`, `pandas`, `scikit-learn`이 필요하고, 분석 스크립트에는 `matplotlib`, `seaborn`도 필요합니다.

프로젝트 폴더에서 다음 명령으로 재실행할 수 있습니다.

```powershell
python model_training_final.py  # 전체 데이터로 Ridge + PCA 학습
python model_training.py        # 50,000개 샘플로 기본 모델 학습
python analysis.py              # 데이터 분석 및 그래프 생성
```

학습 스크립트를 다시 실행하면 해당 제출 CSV와 모델 파일을 덮어씁니다. 전체 데이터 학습에는 시간과 메모리가 많이 필요합니다.

저장된 Ridge + PCA 모델로 예측할 때는 입력 열을 학습 데이터와 같은 `0`~`225` 순서로 맞춰야 합니다.

```python
import pickle
import pandas as pd

test = pd.read_csv("data/test.csv")
columns = [str(i) for i in range(226)]

with open("models_final.pkl", "rb") as f:
    artifacts = pickle.load(f)

X_scaled = artifacts["scaler"].transform(test[columns].to_numpy())
X_pca = artifacts["pca"].transform(X_scaled)
predictions = {
    layer: model.predict(X_pca)
    for layer, model in artifacts["models"].items()
}
result = pd.DataFrame(predictions)
```


## 다음 개선 방향

먼저 학습·검증 데이터를 분리하고 전처리를 학습 분할에만 적용한 뒤, 동일한 검증 데이터와 평가 지표로 모델을 비교해야 합니다. 반사율과 두께의 비선형 관계를 고려한 모델도 검토할 수 있습니다.
