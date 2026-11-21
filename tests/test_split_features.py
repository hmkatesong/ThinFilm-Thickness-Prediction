"""분할 규칙과 피처의 기본 성질 검사. 실제 데이터 대신 같은 격자 구조의 가짜 y를 만든다."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from features import ALL_GROUPS, FeatureBuilder  # noqa: E402
from split import STUDY, n_unseen_layers, random_split, skip_split  # noqa: E402

GRID = np.arange(10, 310, 10)


@pytest.fixture(scope="module")
def y_full():
    return np.array(list(itertools.product(GRID, repeat=4)), dtype=np.float32)


def test_skip_split_rules(y_full):
    tr, va, te = skip_split(y_full, n_valid=2000, n_test=5000)
    assert len(tr) == 15 ** 4
    assert np.isin(y_full[tr], STUDY).all()  # train은 공부용 두께만
    for idx in (va, te):
        assert (n_unseen_layers(y_full[idx]) >= 1).all()  # 시험용 두께가 한 층 이상
        assert (y_full[idx] < 300).all()  # 300 nm 제외
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)


def test_skip_split_unseen_values(y_full):
    """valid의 시험용 두께 값은 train 어느 층에도 나오지 않는다."""
    tr, va, _ = skip_split(y_full, n_valid=2000, n_test=10)
    seen = set(np.unique(y_full[tr]))
    unseen_vals = set(np.unique(y_full[va])) - set(STUDY)
    assert unseen_vals and not unseen_vals & seen


def test_random_split_disjoint():
    tr, va = random_split(810_000, n_valid=1000)
    assert len(tr) == 15 ** 4 and len(va) == 1000 and not set(tr) & set(va)


def test_features_shape_and_row_independence():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 6 * np.pi, 226)
    X = (0.4 + 0.3 * np.sin(t[None] * rng.uniform(0.5, 2, (50, 1)))).astype(np.float32)
    X += rng.normal(0, 0.009, X.shape).astype(np.float32)
    fb = FeatureBuilder(ALL_GROUPS).fit(X)
    F = fb.transform(X)
    assert F.shape[0] == 50
    # PCA 외 피처는 다른 행과 무관해야 한다: 한 행만 따로 변환해도 같은 값
    one = fb.transform(X[:1])
    np.testing.assert_allclose(one, F[:1], rtol=1e-5, atol=1e-5)
    # NaN은 봉우리 칸(없을 때 비움)에서만 나온다
    nb = FeatureBuilder(["raw", "sg", "fft", "band", "pca"]).fit(X).transform(X)
    assert np.isfinite(nb).all()


def test_fft_counts_oscillations():
    """출렁임이 촘촘한 곡선일수록 FFT 크기 최대 위치(주기 번호)가 커진다."""
    t = np.linspace(0, 1, 226)
    X = np.stack([0.4 + 0.3 * np.sin(2 * np.pi * k * t) for k in (2, 5, 9)]).astype(np.float32)
    mag = FeatureBuilder(["fft"]).fit(X).transform(X)[:, :30]
    peaks = mag.argmax(axis=1) + 1
    assert list(peaks) == sorted(peaks) and peaks[0] < peaks[-1]


def test_unknown_group_rejected():
    with pytest.raises(ValueError):
        FeatureBuilder(["raw", "nope"])


def test_train_thickness_sum_parity(y_full):
    """train 두께는 모두 10 nm의 홀수 배라 네 층 합이 20 nm의 배수다.
    처음 보는 층 수가 홀수면 합이 10 nm만큼 어긋나 합이 같은 train 조합이 없다 (외우기 홀짝 효과)."""
    tr, va, _ = skip_split(y_full, n_valid=5000, n_test=10)
    assert (np.rint(y_full[tr].sum(axis=1)).astype(int) % 20 == 0).all()
    k = n_unseen_layers(y_full[va])
    s = np.rint(y_full[va].sum(axis=1)).astype(int) % 20
    assert ((s == 10) == (k % 2 == 1)).all()


def test_raw_dir_env(tmp_path, monkeypatch):
    """환경변수 THINFILM_DATA가 가리키는 폴더를 먼저 쓴다."""
    import data
    (tmp_path / "train.csv").write_text("x")
    monkeypatch.setenv("THINFILM_DATA", str(tmp_path))
    assert data.raw_dir() == tmp_path


def test_raw_dir_env_missing_raises(tmp_path, monkeypatch):
    """환경변수 폴더에 train.csv가 없으면 다른 후보로 넘어가지 않고 바로 멈춘다."""
    import data
    monkeypatch.setenv("THINFILM_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.raw_dir()
