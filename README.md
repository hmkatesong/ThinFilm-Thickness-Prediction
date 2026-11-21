# 반사율 스펙트럼 피처 엔지니어링 (윤서 브랜치)

DACON 235554 박막 두께 데이터로 한 실험이다. 반사율 226개로 네 층 두께를 맞히는 문제에서, 곡선을 가공한 피처(FE)를 넣으면 원시 226개만 넣을 때보다 MAE가 얼마나 줄어드는지 봤다. 모델과 분할은 그대로 두고 입력만 바꿔 비교했다.

## 결과 요약

MAE 단위는 nm이고 네 층 평균이다.

| 모델 (valid) | 원시 226개 | 원시 + FE | 감소 |
|---|---|---|---|
| LightGBM, 시드 3개 | 35.35 ± 0.06 | 26.85 ± 0.05 | -24.0% |
| MLP | 25.4 | 21.3 | -16% |
| Ridge | 72.5 | 56.6 | -22% |

±는 시드 3개 MAE의 표준편차(ddof=0)다. 분할은 seed 0으로 고정하고 모델 시드만 바꿨다. MLP와 Ridge 행은 시드 0 하나만 돌린 값이다. LightGBM 감소율은 valid 2만 개를 부트스트랩으로 다시 뽑아 봐도 95% 구간이 23.4~24.3%였다.

test 10만 개는 모델마다 한 번씩만 채점했다. 외우기 기준선은 가장 가까운 학습 곡선의 두께를 그대로 답하는 1-NN이다.

| 모델 | test MAE |
|---|---|
| 외우기 기준선 (1-NN) | 26.44 |
| LightGBM 원시 (E26, 사후 확인) | 35.38 |
| LightGBM + FE | 26.92 |
| MLP + FE, 시드 3개 각각 | 21.18 ± 0.35 |
| MLP + FE, 시드 3개 예측 평균 | 16.94 |

- test에서도 LightGBM은 FE로 23.9% 줄었다(95% 구간 23.7~24.1%). E26은 실험 25개를 끝낸 뒤 이걸 확인하려고 하나 더 돌린 것이다.
- LightGBM + FE는 외우기보다 0.5 nm 나쁘다(짝비교 표준오차 0.1). 외우기를 넘은 건 MLP뿐이다.
- 시드 예측 평균은 처음 계획에 없었고 test에서 먼저 봤다. valid에서도 21.19에서 16.89로 똑같이 줄어서 결과에 넣었다(E25).
- 외우기 점수는 학습에 없던 두께인 층이 몇 개냐에 따라 크게 갈린다. train 두께는 10, 30, ..., 290 nm라서 네 층 합이 항상 20 nm의 배수다. 학습에 없던 층이 홀수 개면 합이 같은 train 곡선이 하나도 없고, 1-NN이 고른 곡선도 전부 합이 어긋났다. 1-NN은 홀수일 때 33.9, 짝수일 때 17.6이다. 합이 안 맞는 게 이 차이의 주된 원인으로 보지만, 원인만 따로 떼어 실험하지는 않았다. MLP + FE(시드 평균)는 17.1과 16.7로 큰 차이가 없다. 같은 짝수 안에서도 2개(16.5)와 4개(25.2)는 차이가 나는데, 이건 따로 보지 않았다(`results/error_by_unseen.csv`, `results/error_by_parity.csv`).

피처별 기여도는 `results/ablation.csv`와 `results/fig_ablation.png`에 있다. 실험 26개 기록은 `experiment_log.csv`에 있다.

## 분할

동훈님 제안의 건너뛰기 분할을 그대로 썼다.
- train: 네 층 모두 10, 30, ..., 290 nm인 조합 50,625개
- valid 20,000개, test 100,000개: 20, 40, ..., 280 nm인 층이 하나 이상 있고 300 nm는 없는 조합. 서로 겹치지 않게 뽑았다(seed 0)

valid와 test에는 학습 때 한 번도 안 나온 두께 값이 들어 있다. 그래서 1-NN처럼 학습 곡선의 두께를 그대로 답하면 그 층에서 최소 10 nm씩 틀린다. 모델이 곡선과 두께의 관계를 배웠는지 보려고 고른 분할이다.

- 학습 크기를 5만으로 맞추면 무작위 분할도 1-NN 점수가 거의 같다. 무작위 26.4(E06), 건너뛰기 26.2(E02)다.
- 8:1:1 크기 무작위(학습 64.8만, valid 2만)에서는 1-NN이 10.7까지 내려간다(E11). 같은 무작위 분할에서 학습만 5만(E06, 26.4)과 64.8만(E11, 10.7)으로 바꾼 비교라, 이 차이는 분할 방식보다 학습 크기에서 나온다.
- 이 분할의 valid는 DACON test보다 학습 곡선에서 훨씬 멀다. 가장 가까운 학습 곡선까지 거리(중앙값)가 DACON test는 0.48, 건너뛰기 valid는 0.93이다. 그래서 DACON보다 어려운 시험일 거라고 보고, 여기 점수나 모델 순위가 DACON 리더보드에서도 같다고 보지 않는다. DACON 조건에서 모델을 돌려 보지는 않았다.

## 피처 (`src/features.py`)

| 그룹 | 내용 | 개수 |
|---|---|---|
| sg | Savitzky-Golay 1차, 2차 미분 (창 11, 3차) | 452 |
| fft | 평균을 뺀 평활 곡선에 Hann 창을 씌운 FFT. 저주파 30개의 크기, 실수부, 허수부 | 90 |
| peak | 평활 곡선에서 find_peaks(prominence 0.03). 봉우리와 골 개수, 각각 처음 4개의 위치와 높이, 극값 사이 평균 간격(반주기) | 19 |
| band | 평활 곡선을 8구간으로 나눈 평균, 표준편차와 전체 평균, 표준편차, 최소, 최대, argmin, argmax | 22 |
| pca | 평활 곡선의 PCA 20개 (train으로 fit) | 20 |

평활 곡선은 SG(창 11, 3차)로 매끈하게 만든 곡선이다. 최선 조합은 raw + fft + peak + band + pca(377개)다. sg까지 넣으면 0.16 nm 좋아지지만 시간이 2.3배 걸려서 뺐다.

## 설치와 데이터

```bash
pip install -r requirements.txt
```

DACON 235554에서 `train.csv`, `test.csv`를 받아 레포 안 `data/` 폴더에 넣는다(git에는 안 올라간다). 다른 곳에 두었으면 환경변수 `THINFILM_DATA`에 그 폴더를 적는다. 처음 실행할 때 `cache/`에 npy 파일을 만든다(30초쯤, 750 MB).

## 실행

```bash
python -X utf8 -m pytest -q tests
python -X utf8 src/run_exp.py --exp R14 --model lgbm --feats raw,fft,peak,band,pca
python -X utf8 src/plot_ablation.py
```

- `run_exp.py`는 실행할 때마다 `experiment_log.csv`에 한 줄을 붙인다. 이미 있는 이름(E01~E26)은 거부하니까, 다시 돌려 볼 때는 R14처럼 새 이름을 쓰면 된다.
- 로그의 `시드평균예측MAE`는 시드별 예측을 먼저 평균하고 잰 MAE다. 시드가 1개면 `MAE평균`과 같다.
- LightGBM 학습률은 `--lr`로 바꾼다. E04, E05는 0.05였고 나머지는 기본값 0.1이다. E11은 `--split random --n_train 648000`으로 돌렸다.
- 오류 분석(`src/analyze.py`)은 test 예측 파일이 있어야 돈다. `results/preds/`는 git에 없어서, 아래 세 줄을 먼저 돌리고 `python -X utf8 src/analyze.py R22 R24 R23`을 실행한다. MLP는 5분, LightGBM은 9분쯤 걸린다. 이름이 E22 E24 E23이 아니면 결과 파일 이름 끝에 `_R22_R24_R23`이 붙어서, 올려 둔 표를 덮어쓰지 않는다.

```bash
python -X utf8 src/run_exp.py --exp R22 --model mlp --feats raw,fft,peak,band,pca --seeds 0,1,2 --eval test
python -X utf8 src/run_exp.py --exp R23 --model knn --eval test
python -X utf8 src/run_exp.py --exp R24 --model lgbm --feats raw,fft,peak,band,pca --eval test
```

- 모델은 mean, knn, ridge, lgbm, mlp 다섯 가지다(`src/models.py`). CPU만 쓴다. LightGBM은 시드 1개에 4~18분, 시드 3개면 25분쯤 걸린다. MLP는 시드 1개에 2분쯤이다.
- LightGBM early stopping은 valid MAE(l1)만 보게 했다(`first_metric_only=True`). E01~E25는 이 옵션을 넣기 전에 돌렸다. 5,000그루 상한까지 간 층은 결과가 같고, 실제로 옵션을 넣고 돌린 E26의 나무 수가 E07과 똑같았다. 상한 전에 멈췄던 E15 layer_1, E16 layer_2와 layer_4만 다시 돌리면 조금 다를 수 있다.

## 한계

- 이 분할로는 "20 nm 간격으로 배우고 그 사이 값 맞히기"까지만 잰다. 학습이 5만 개라 DACON 리더보드 점수와 직접 비교할 수 없다. 학습이 촘촘할 때도 FE 효과와 모델 순위가 같은지는 확인하지 않았다.
- LightGBM은 학습률 0.05와 0.1 모두 5,000그루 상한에서 아직 좋아지는 중이었다. -24%는 같은 5,000그루 안에서 잰 차이다. 원시는 0.05일 때가 0.8 nm 더 좋았는데, 그걸 기준으로 해도 FE 감소율은 22%다.
- 피처 설정값은 흔히 쓰는 값으로 한 번 정하고 튜닝하지 않았다. Ridge alpha는 후보 중 가장 큰 값(100)이 골라졌다.
- 예측의 2%쯤이 두께 범위(10~300 nm)를 벗어난다. [10, 300]으로 잘라 주면 MLP + FE 시드 평균이 16.94에서 16.74가 되는데, 결과에는 넣지 않았다.
- 학습 오차는 기록하지 않았다. valid와 test 점수가 같다는 건 valid로 설정을 고른 영향이 작다는 뜻이고, 과적합 여부까지 말해 주지는 않는다.
