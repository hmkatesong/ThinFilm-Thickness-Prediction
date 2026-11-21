# 2단계: 건너뛰기(20 nm 부분격자) 분할
# Train: 네 층 모두 10,30,...,290 nm / Valid: 내부 점의 10% / Test: 내부 점의 나머지 / edge: 300 nm 포함(외삽)
# 인덱스는 grid를 펼친 값 i1*27000 + i2*900 + i3*30 + i4
import os, sys
import numpy as np
from common import WORK
sys.stdout.reconfigure(encoding="utf-8")


def make_split(seed=0, valid_frac=0.1):
    i = np.stack(np.unravel_index(np.arange(810000), (30,) * 4), 1)
    off = (i % 2).sum(1)                 # 학습 두께가 아닌 층 수 (0~4)
    edge = (i == 29).any(1)              # 한 층이라도 300 nm
    train = np.flatnonzero((off == 0) & ~edge)
    inner = np.flatnonzero((off > 0) & ~edge)
    pick = np.random.default_rng(seed).permutation(inner)
    n_valid = int(round(len(inner) * valid_frac))
    return {"train": train, "valid": np.sort(pick[:n_valid]), "test": np.sort(pick[n_valid:]),
            "edge": np.flatnonzero(edge), "off": off.astype(np.int8)}


if __name__ == "__main__":
    sp = make_split()
    assert len(sp["train"]) == 50625 and len(sp["edge"]) == 102719
    assert len(sp["valid"]) + len(sp["test"]) == 656656
    assert not np.intersect1d(sp["valid"], sp["test"]).size
    assert not np.intersect1d(sp["train"], sp["test"]).size
    print("train", len(sp["train"]), "valid", len(sp["valid"]), "test", len(sp["test"]), "edge", len(sp["edge"]))
    print("Test의 학습 밖 층 수(0~4) 분포:", np.bincount(sp["off"][sp["test"]], minlength=5).tolist())
    np.savez(os.path.join(WORK, "split.npz"), **sp)
