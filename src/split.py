"""검증 분할.

skip (주 분할): 두께를 층마다 번갈아 '공부용'(10,30,...,290)과 '시험용'(20,40,...,280)으로 나눈다.
  train = 네 층 모두 공부용 두께 (15^4 = 50,625개)
  valid/test = 시험용 두께가 한 층 이상 있고 300 nm는 없는 조합에서 서로 겹치지 않게 무작위 추출
  -> 검증 샘플의 두께 조합은 학습 때 한 번도 안 나온 값을 포함한다 (격자 사이 값 맞히기).
random (대조): 전체에서 무작위로 뽑는다. 기본은 skip과 같은 학습 크기(50,625).
  같은 크기면 외우기(1-NN) 점수가 skip과 거의 같다 (E06 26.4 vs E02 26.2).
  --n_train 648000(8:1:1)이면 곡선이 거의 같은 이웃 조합이 학습에 대부분 들어가 10.7이 된다 (E11).
"""
import numpy as np

STUDY = np.arange(10, 300, 20)  # 10, 30, ..., 290
N_TRAIN_SKIP = 15 ** 4


def skip_split(y, n_valid=20_000, n_test=100_000, seed=0):
    y = np.rint(y).astype(int)
    is_study = np.isin(y, STUDY)
    train = np.where(is_study.all(axis=1))[0]
    pool = np.where(~is_study.all(axis=1) & (y < 300).all(axis=1))[0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool, n_valid + n_test, replace=False)
    return train, np.sort(pick[:n_valid]), np.sort(pick[n_valid:])


def random_split(n_rows, n_train=N_TRAIN_SKIP, n_valid=20_000, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(n_rows, n_train + n_valid, replace=False)
    return np.sort(pick[:n_train]), np.sort(pick[n_train:])


def n_unseen_layers(y):
    """샘플마다 시험용 두께(학습에 없던 값)인 층 개수 0~4."""
    return (~np.isin(np.rint(y).astype(int), STUDY)).sum(axis=1)
